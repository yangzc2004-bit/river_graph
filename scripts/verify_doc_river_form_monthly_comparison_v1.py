"""Replay same-calendar morphology comparisons and all background fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_form_monthly_comparison_v1 import (
    CELLS,
    DATASET,
    ROOT,
    build_monthly_records,
    digest,
    hydro_balance,
)

from river_graph.analysis.river_form_monthly import (
    ARMS,
    background_prediction,
    crossfit_backgrounds,
    information_scores,
    paired_records,
    response_evidence,
    response_summary,
    station_weights,
)
from river_graph.experiments.provenance import sha256_file


def equivalent(frame, path):
    saved = pd.read_csv(path, dtype={c: str for c in ("station", "station_a", "station_b", "huc4", "excluded_huc4")})
    for c in ("first_month", "last_month"):
        if c in saved:
            saved[c] = pd.to_datetime(saved[c])
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), saved.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    out = ROOT/"analysis"
    config = json.loads((ROOT/"config.json").read_text())
    sidecar = json.loads((out/"monthly_predictions.provenance.json").read_text())
    assert sidecar["config_hash"] == digest(config)
    assert sidecar["dataset_hash"] == sha256_file(DATASET)
    assert sidecar["source_cells_sha256"] == sha256_file(CELLS)
    assert sidecar["prediction_sha256"] == sha256_file(out/"monthly_predictions.parquet")
    assert sidecar["runtime_snapshot_hash"] == digest(sidecar["runtime_sources"])
    for path, expected in sidecar["runtime_sources"].items():
        assert sha256_file(Path(path)) == expected
        assert sha256_file(ROOT/"code_snapshot"/path) == expected
    frame, pairs = build_monthly_records()
    pd.testing.assert_frame_equal(frame, pd.read_parquet(out/"station_month_inputs.parquet"), check_exact=True)
    assert frame.station.nunique() == sources["n_stations"] == 297
    assert frame.huc4.nunique() == sources["n_huc4"] == 62
    assert len(frame) == sources["n_source_months"] == 18688
    assert sidecar["mask_hash"] == hashlib.sha256(frame[["station", "month_index", "huc4"]].to_csv(index=False).encode()).hexdigest()
    assert not frame.duplicated(["station", "month_index"]).any()
    assert frame.groupby("comid").huc4.nunique().eq(1).all()
    np.testing.assert_allclose(pd.Series(station_weights(frame)).groupby(frame.station).sum(), 1.)

    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    allowed = np.zeros(np.asarray(data["y"]).shape, bool)
    allowed.ravel()[np.load(CELLS)] = True
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    sites = np.array([index[s] for s in frame.station])
    months = frame.month_index.to_numpy()
    assert allowed[sites, months].all()
    np.testing.assert_array_equal(frame.y_true, np.asarray(data["y"])[sites, months])
    hidden = dict(data)
    changed = np.asarray(data["y"]).copy()
    changed[~allowed] = 123456.
    hidden["y"] = changed
    alternate, alternate_pairs = build_monthly_records(hidden)
    pd.testing.assert_frame_equal(frame, alternate, check_exact=True)
    pd.testing.assert_frame_equal(pairs, alternate_pairs, check_exact=True)

    predictions, states = crossfit_backgrounds(frame)
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(out/"monthly_predictions.parquet"), check_exact=True)
    assert states == json.loads((out/"fitted_states.json").read_text())
    assert len(predictions) == len(frame)*len(ARMS)
    assert not predictions.duplicated(["station", "month_index", "arm"]).any()
    folds = predictions[predictions.arm.eq("environment")].set_index(["station", "month_index"]).fold
    assert predictions.groupby("huc4").fold.nunique().eq(1).all()
    checks = 0
    for state in states:
        assert set(state["held_huc4"]).isdisjoint(state["training_huc4"])
        query = frame[frame.huc4.isin(state["held_huc4"])].copy()
        expected = background_prediction(query, state)
        query.y_true = 999999.
        query["future_doc"] = -999999.
        np.testing.assert_array_equal(expected, background_prediction(query, state))
        checks += 1
    # A whole nested re-fit checks that held-region labels do not influence the
    # fitted states, in addition to the inference-only query perturbations.
    first = next(s for s in states if s["fold"] == 0)["held_huc4"]
    altered = frame.copy()
    altered.loc[altered.huc4.isin(first), "y_true"] *= 1000
    altered_predictions, altered_states = crossfit_backgrounds(altered)
    cols = ["station", "month_index", "arm", "log_pred", "high_probability"]
    pd.testing.assert_frame_equal(predictions[predictions.fold.eq(0)][cols],
                                  altered_predictions[altered_predictions.fold.eq(0)][cols], check_exact=True)
    assert [s for s in states if s["fold"] == 0] == [s for s in altered_states if s["fold"] == 0]

    records, ledger = paired_records(frame, pairs, predictions)
    pd.testing.assert_frame_equal(records, pd.read_parquet(out/"paired_month_records.parquet"), check_exact=True)
    assert not records.duplicated(["pair_id", "month_index"]).any()
    lookup = frame.set_index(["station", "month_index"])
    for suffix in ("a", "b"):
        keys = pd.MultiIndex.from_arrays([records["station_"+suffix], records.month_index])
        np.testing.assert_array_equal(records["y_true_"+suffix], lookup.loc[keys, "y_true"])
        np.testing.assert_array_equal(records.date, lookup.loc[keys, "date"])
        # Every adjustment background is from the region-excluding fold.
        assert (folds.loc[keys].to_numpy() >= 0).all()
    assert ledger[ledger.included & ledger.class_a.eq(1) & ledger.class_b.eq(3)].shape[0] == 20
    assert ledger[ledger.complete_hydro_included & ledger.class_a.eq(1) & ledger.class_b.eq(3)].shape[0] == 17

    evidence = response_evidence(records)
    contrasts, sensitivity = response_summary(evidence, sources["bootstrap_draws"])
    station, scores, gains = information_scores(predictions, sources["bootstrap_draws"])
    tables = {"pair_inclusion_ledger": ledger, "paired_response_evidence": evidence,
              "paired_response_contrasts": contrasts, "omitted_huc4_sensitivity": sensitivity,
              "hydro_balance": hydro_balance(records), "station_model_errors": station,
              "information_scores": scores, "information_gains": gains}
    for name, rebuilt in tables.items():
        equivalent(rebuilt, out/f"{name}.csv")
    assert contrasts[contrasts.n_huc4.eq(1)][["ci_low", "ci_high"]].isna().all().all()
    for suffix in ("", "_cn"):
        manifest = json.loads((ROOT/"figures"/f"manifest{suffix}.json").read_text())
        assert manifest["generator_sha256"] == sha256_file(Path("scripts/plot_doc_river_form_monthly_comparison_v1.py"))
        for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected, path
    record = {"status": "passed", "n_source_months_replayed": len(frame), "n_prediction_rows_replayed": len(predictions),
              "background_states_replayed": len(states), "held_label_inference_checks": checks,
              "held_label_refit_perturbation": "passed", "nonpermitted_doc_perturbation": "passed",
              "same_calendar_pair_alignment": "passed", "all_pairs_unchanged": True,
              "singleton_region_intervals": "unavailable", "bootstrap_draws": sources["bootstrap_draws"],
              "tables_replayed": len(tables), "figure_manifests_checked": True, "existing_neural_training": False}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
