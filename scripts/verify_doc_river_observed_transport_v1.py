"""Replay observed DOC arrival predictions and their receiver-balanced analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_observed_transport_v1 import (
    DATASET,
    OLD,
    ROOT,
    build_records,
    paired_gains,
    receiver_coverage,
    receiver_metrics,
    summarize,
    system_sensitivity,
)

from river_graph.analysis.river_observed_transport import (
    OPERATORS,
    calibrated_prediction,
    concentration_proxy,
    connection_metrics,
    nested_predictions,
)
from river_graph.experiments.provenance import sha256_file


def equivalent(rebuilt, path):
    saved = pd.read_csv(path, dtype={c: str for c in ("target", "source_a", "source_b", "huc4", "group")})
    if "group" in rebuilt:
        rebuilt = rebuilt.copy()
        rebuilt.group = rebuilt.group.astype(str)
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), saved.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-11)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    a = ROOT/"analysis"
    sidecar = json.loads((a/"connection_predictions.provenance.json").read_text())
    assert sidecar["config_hash"] == digest(json.loads((ROOT/"config.json").read_text()))
    assert sidecar["dataset_hash"] == sha256_file(DATASET)
    assert sidecar["source_cells_sha256"] == sha256_file(OLD/"source_cells.npy")
    assert sidecar["prediction_sha256"] == sha256_file(a/"connection_predictions.parquet")
    assert sidecar["runtime_snapshot_hash"] == digest(sidecar["runtime_sources"])
    for path, expected in sidecar["runtime_sources"].items():
        assert sha256_file(ROOT/"code_snapshot"/path) == expected
        assert sha256_file(Path(path)) == expected

    frame, ledger, window, max_path = build_records()
    pd.testing.assert_frame_equal(frame, pd.read_parquet(a/"input_records.parquet"), check_exact=True)
    mask_payload = frame[["pair_id", "target", "source_a", "source_b", "month_index", "component"]].to_csv(index=False).encode()
    assert sidecar["mask_hash"] == hashlib.sha256(mask_payload).hexdigest()
    assert frame.pair_id.nunique() == sources["pairs"] == 59
    assert frame.target.nunique() == sources["receivers"] == 22
    assert frame.component.nunique() == sources["components"] == 11
    assert len(frame) == sources["pair_months"] == 3026
    assert len(frame.drop_duplicates(["target", "month_index"])) == sources["unique_receiver_months"] == 1092
    assert frame.groupby("target").component.nunique().eq(1).all()
    assert frame.groupby("pair_id").size().ge(24).all()

    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    truth = np.asarray(data["y"])
    allowed = np.zeros(truth.shape, bool)
    allowed.ravel()[np.load(OLD/"source_cells.npy")] = True
    for branch in ("a", "b"):
        station = np.array([index[s] for s in frame[f"source_{branch}"]])
        month = frame.month_index.to_numpy()
        assert allowed[station, month].all() and allowed[station, month-1].all()
        np.testing.assert_array_equal(frame[f"doc_{branch}_now"], truth[station, month])
        np.testing.assert_array_equal(frame[f"doc_{branch}_previous"], truth[station, month-1])
    hidden = dict(data)
    changed = truth.copy()
    changed[~allowed] = 123456.
    hidden["y"] = changed
    hidden_frame, hidden_ledger, hidden_window, hidden_max = build_records(hidden)
    pd.testing.assert_frame_equal(frame, hidden_frame, check_exact=True)
    pd.testing.assert_frame_equal(ledger, hidden_ledger, check_exact=True)
    pd.testing.assert_frame_equal(window, hidden_window, check_exact=True)
    assert hidden_max == max_path

    predictions, trials, states = nested_predictions(frame, max_path)
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(a/"connection_predictions.parquet"), check_exact=True)
    assert states == json.loads((a/"fitted_states.json").read_text())
    assert len(predictions) == len(frame)*len(OPERATORS)
    assert not predictions.duplicated(["pair_id", "month_index", "operator"]).any()
    assert np.isfinite(predictions.y_pred).all() and predictions.y_pred.ge(0).all()

    # Receiving truth is a scoring channel. Keep the legitimately measured
    # upstream input channels fixed, including when a receiver is another source.
    perturbation_checks = 0
    for state in states:
        held = state["held_component"]
        train, test = frame[frame.component.ne(held)], frame[frame.component.eq(held)].copy()
        assert held not in state["training_components"]
        assert sorted(train.component.unique()) == state["training_components"]
        assert state["q90_threshold"] == float(train.drop_duplicates(["target", "month_index"]).y_true.quantile(.9))
        test.y_true = 999999.
        replay = calibrated_prediction(test, concentration_proxy(test, state["operator"], state["fraction"], max_path), state)
        saved = predictions[predictions.component.eq(held) & predictions.operator.eq(state["operator"])].y_pred
        np.testing.assert_array_equal(replay, saved)
        perturbation_checks += 1
    held = int(frame.component.min())
    altered = frame.copy()
    altered.loc[altered.component.eq(held), "y_true"] *= 1000
    changed_predictions, _, changed_states = nested_predictions(altered, max_path)
    compare = ["pair_id", "month_index", "operator", "fraction", "y_pred", "q90_threshold"]
    pd.testing.assert_frame_equal(predictions[predictions.component.eq(held)][compare],
                                  changed_predictions[changed_predictions.component.eq(held)][compare], check_exact=True)
    assert [s for s in states if s["held_component"] == held] == [s for s in changed_states if s["held_component"] == held]

    connections = connection_metrics(predictions)
    receivers = receiver_metrics(connections)
    tables = {"inclusion_ledger": ledger, "window_availability": window, "selection_trials": trials,
              "connection_metrics": connections, "receiver_metrics": receivers,
              "receiver_coverage": receiver_coverage(predictions), "system_sensitivity": system_sensitivity(receivers),
              "operator_summary": summarize(receivers, sources["bootstrap_draws"]),
              "paired_gains": paired_gains(receivers, sources["bootstrap_draws"])}
    for name, rebuilt in tables.items():
        equivalent(rebuilt, a/f"{name}.csv")
    gains = tables["paired_gains"]
    singleton = gains[gains.n_blocks.eq(1)]
    assert singleton[["ci_low", "ci_high", "gain_ci_low_pct", "gain_ci_high_pct"]].isna().all().all()
    intervals = tables["operator_summary"]
    assert intervals[intervals.n_blocks.eq(1)][["ci_low", "ci_high"]].isna().all().all()
    for name in ("manifest.json", "manifest_cn.json"):
        manifest = json.loads((ROOT/"figures"/name).read_text())
        assert sha256_file(Path("scripts/plot_doc_river_observed_transport_v1.py")) == manifest["generator_sha256"]
        for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected, path
    record = {"status": "passed", "source_files_checked": len(sources["source_hashes"]),
              "prediction_rows_replayed": len(predictions), "outer_system_folds": frame.component.nunique(),
              "query_truth_perturbation_checks": perturbation_checks,
              "held_truth_parameter_selection_perturbation": "passed",
              "nonpermitted_doc_perturbation": "passed", "source_month_alignment": "current and immediately previous calendar bin",
              "bootstrap_draws": sources["bootstrap_draws"], "tables_replayed": len(tables),
              "source_snapshot_and_prediction_sidecar": "passed", "figure_manifests_checked": True,
              "existing_neural_models_retrained": False}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
