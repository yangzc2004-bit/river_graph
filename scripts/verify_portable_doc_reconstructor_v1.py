"""Replay portable DOC inference without fitting or using new-site DOC labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from portable_doc_reconstructor_v1 import PROCEDURES, PortableDOCReconstructor
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT
from run_doc_geographical_confirmation_v1 import ROOT as GEO_ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_portable_inference_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=GEO_ROOT/"runs/huc4_1013_seed42")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, output = args.source_run, args.root
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/"config.json").read_text())
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        cells = saved["test"].copy()
    months = dataset["y"].shape[1]
    ids = np.unique(cells//months)
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], dataset["y"].shape)
    inputs = {key: np.asarray(dataset[key])[ids] for key in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=np.asarray(dataset["months"]), daily_features=daily[ids])
    if any(key in inputs for key in ("y", "y_mask", "ph", "spec_conductance")):
        raise ValueError("portable replay must not pass water-quality labels")
    target_rows = np.searchsorted(ids, cells//months)
    table = pd.read_parquet(run/"predictions.parquet")
    records = []
    for procedure in PROCEDURES:
        predictor = PortableDOCReconstructor.from_geographical_run(run, procedure=procedure)
        prediction = predictor.predict(inputs)
        expected = table[table.model_name.eq(procedure)]
        np.testing.assert_array_equal(expected.cell, cells)
        replay = prediction[target_rows, cells % months]
        np.testing.assert_allclose(replay, expected.y_pred, rtol=1e-6, atol=1e-6)
        record = {"procedure": procedure, "max_abs_prediction_difference": float(
            np.max(np.abs(replay-expected.y_pred.to_numpy()))), "test_cells": len(cells),
            "new_site_count": len(ids), "months": months, "prediction_replay": "rtol/atol1e-6"}
        if procedure == "unmonitored_integrated":
            _, _, neural = predictor.prepare_inputs(inputs)
            with np.load(run/"test_inputs.npz", allow_pickle=False) as original:
                for key in ("raw", "env", "age", "support", "extra"):
                    np.testing.assert_array_equal(neural[key], original[key])
            record["prepared_input_replay"] = "bitwise for raw/env/age/support/extra"
            export = output/"exports"/run.name
            if export.exists():
                loaded = PortableDOCReconstructor.load(export)
                if loaded.metadata["source_completion_sha256"] != sha256_file(run/"complete.json"):
                    raise ValueError("portable export refers to a different fitted source")
            else:
                predictor.save(export)
                loaded = PortableDOCReconstructor.load(export)
            reloaded = loaded.predict(inputs)
            np.testing.assert_allclose(reloaded, prediction, atol=1e-12, rtol=1e-12)
            record["save_load_replay"] = "rtol/atol1e-12 (parallel forest reduction)"
            one = {key: value[:1] if key != "months" else value for key, value in inputs.items()}
            np.testing.assert_allclose(loaded.predict(one), prediction[:1], rtol=1e-6, atol=1e-6)
            record["arbitrary_new_node_count"] = True
        records.append(record)
        print(json.dumps(record), flush=True)
        del predictor
    write_json(output/"verification.json", {"source_run": str(run),
        "source_completion_sha256": sha256_file(run/"complete.json"),
        "verifier_sha256": sha256_file(__file__),
        "portable_module_sha256": sha256_file("scripts/portable_doc_reconstructor_v1.py"),
        "role": "inference equivalence, not new model selection or external validation",
        "water_quality_labels_passed": False, "procedures": records})


if __name__ == "__main__":
    main()
