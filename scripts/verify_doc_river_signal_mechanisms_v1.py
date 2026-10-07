"""Replay signal diagnostics and inspect source roles and unchanged fitted models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_signal_mechanisms_v1 import (
    CELLS,
    CODE,
    DATASET,
    INPUT,
    OLD,
    ORIGIN,
    ROOT,
    STATES,
    analyze,
    digest,
    measured_flows,
)

from river_graph.analysis.river_signal_mechanisms import (
    arrival_opportunity,
    mixing_records,
)
from river_graph.experiments.provenance import sha256_file


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path)
    config = json.loads((ROOT/"config.json").read_text())
    assert digest(config) == sources["config_hash"]
    sidecar = json.loads((ROOT/"analysis/arrival_opportunity.provenance.json").read_text())
    assert sidecar["config_hash"] == digest(sidecar["config"]) == sources["config_hash"]
    assert sidecar["dataset_hash"] == sha256_file(DATASET)
    assert sidecar["source_cells_sha256"] == sha256_file(CELLS)
    assert sidecar["mask_hash"] == json.loads(ORIGIN.read_text())["mask_hash"]
    assert sidecar["runtime_snapshot_hash"] == digest(sidecar["runtime_sources"])
    assert sidecar["prediction_sha256"] == sha256_file(ROOT/"analysis/arrival_opportunity.parquet")
    assert sidecar["input_records_sha256"] == sha256_file(INPUT) and sidecar["saved_states_sha256"] == sha256_file(STATES)
    for path, expected in sidecar["runtime_sources"].items():
        assert sha256_file(Path(path)) == expected
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    frame = pd.read_parquet(INPUT)
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    y = np.asarray(data["y"])
    permitted = np.zeros(y.shape, bool)
    permitted.ravel()[np.load(CELLS)] = True
    for station, column, lag in (("source_a", "doc_a_now", 0), ("source_b", "doc_b_now", 0),
        ("target", "y_true", 0), ("source_a", "doc_a_previous", 1), ("source_b", "doc_b_previous", 1)):
        nodes, months = frame[station].map(index).to_numpy(int), frame.month_index.to_numpy(int)-lag
        assert (months >= 0).all() and permitted[nodes, months].all()
        np.testing.assert_array_equal(y[nodes, months], frame[column])
    np.testing.assert_array_equal(pd.DatetimeIndex(data["months"])[frame.month_index], frame.date)
    states = json.loads(STATES.read_text())
    tables, products, fits = analyze(frame, data, states, config["max_path_km"], config["bootstrap_draws"])
    for name, table in tables.items():
        saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype={"target": str, "huc4": str})
        pd.testing.assert_frame_equal(table.reset_index(drop=True), saved, check_dtype=False,
                                      check_exact=False, atol=1e-10, rtol=1e-10)
    for name, product in products.items():
        pd.testing.assert_frame_equal(product, pd.read_parquet(ROOT/"analysis"/f"{name}.parquet"), check_exact=True)
    assert fits == json.loads((ROOT/"analysis/calendar_fits.json").read_text())
    pred = pd.read_parquet(OLD/"analysis/connection_predictions.parquet")
    saved_mean = pred[pred.operator.eq("mean_delay")].set_index(["pair_id", "month_index"])
    arrival = products["arrival_opportunity"].set_index(["pair_id", "month_index"])
    np.testing.assert_allclose(arrival.mean_input_prediction, saved_mean.loc[arrival.index, "y_pred"], atol=1e-12)
    altered = frame.copy()
    altered["y_true"] = 999999.
    changed = arrival_opportunity(altered, config["max_path_km"])
    independent = ["arrival_gap", "arrival_gap_relative", "mean_delay_proxy", "branch_arrival_proxy"]
    pd.testing.assert_frame_equal(products["arrival_opportunity"][independent], changed[independent], check_exact=True)
    _, source_changed, _ = mixing_records(altered)
    pd.testing.assert_frame_equal(products["calendar_anomalies"][["source_a_anomaly", "source_b_anomaly", "mixture_anomaly"]],
        source_changed[["source_a_anomaly", "source_b_anomaly", "mixture_anomaly"]], check_exact=True)
    altered_data = dict(data)
    altered_data["y"] = np.full_like(y, 999999.)
    for a, b in zip(measured_flows(frame, data), measured_flows(frame, altered_data), strict=True):
        np.testing.assert_array_equal(a, b)
    # A later hydro measurement cannot alter earlier screening records.
    cutoff = int(np.median(frame.month_index))
    earlier = frame[frame.month_index.le(cutoff)]
    altered_data["x"] = np.asarray(data["x"]).copy()
    altered_data["x"][:, cutoff+1:] = 999999.
    for a, b in zip(measured_flows(earlier, data), measured_flows(earlier, altered_data), strict=True):
        np.testing.assert_array_equal(a, b)
    s = tables["mixing_summary"]
    assert s[s.population.isin(["form_1", "form_2"])][["ci_low", "ci_high"]].isna().all().all()
    for suffix in ("", "_cn"):
        manifest = json.loads((ROOT/"figures"/f"manifest{suffix}.json").read_text())
        assert manifest["generator_sha256"] == sha256_file(Path("scripts/plot_doc_river_signal_mechanisms_v1.py"))
        for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected, path
    result = {"status": "passed", "source_cells_and_dates": "matched original dataset and permitted roles",
        "native_mixing_and_arrival_identities": "passed", "receiver_doc_independent_inputs": "passed",
        "future_hydro_independent_screen": "passed", "saved_mean_delay_predictions": "matched without refit",
        "n_connections": len(tables["mixing_connections"]), "n_receivers": len(tables["mixing_receivers"]),
        "n_replayed_tables": len(tables), "n_replayed_products": len(products), "bootstrap_draws": config["bootstrap_draws"],
        "single_system_class_intervals": "unavailable", "figure_manifests": "passed", "new_model_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
