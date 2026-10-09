"""Replay geographical fits through the named-site current-source facade."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from portable_current_source_doc_v2 import PortableCurrentSourceDOC
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, DATASET
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_current_source_portable_v2")
GEO = Path("experiments/phase4_transfer/doc_current_availability_attention_geographical_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", default=["1013", "1019", "0708", "1030", "1101"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    daily, _, _ = load_daily_pack(DAILY_ROOT, sha256_file(DATASET), tuple(dataset["y"].shape))
    t, rows = dataset["y"].shape[1], []
    for huc4 in args.huc4:
        for seed in args.seeds:
            run = GEO/"runs"/f"huc4_{huc4}_seed{seed}"
            model = PortableCurrentSourceDOC.from_geographical_run(run)
            panel = pd.read_parquet(run/"predictions.parquet")
            expected = panel[panel.model_name.eq("available_real_integrated")]
            cells = expected.cell.to_numpy()
            ids = np.unique(cells//t)
            inputs = {key: np.asarray(dataset[key])[ids] for key in ("site_no", "x", "x_mask", "static", "regime")}
            inputs.update(months=dataset["months"], daily_features=daily[ids])
            actual = model.predict(inputs)[np.searchsorted(ids, cells//t), cells % t]
            np.testing.assert_allclose(actual, expected.y_pred, rtol=1e-6, atol=1e-6)
            row = {"run": run.name, "source_completion_hash": sha256_file(run/"complete.json"),
                "receiving_stations": len(ids), "cells": len(cells),
                "max_abs_replay_error_mg_l": float(np.abs(actual-expected.y_pred).max()),
                "same_fitted_model_no_reselection": True}
            if not rows:
                export = args.root/"technical_export"
                if not export.exists():
                    model.save(export)
                restored = PortableCurrentSourceDOC.load(export)
                if restored.metadata != model.metadata:
                    raise ValueError("technical export belongs to a different geographical fit")
                loaded = restored.predict(inputs)
                original = model.predict(inputs)
                np.testing.assert_allclose(loaded, original, rtol=0, atol=1e-12)
                row["technical_save_load"] = "within1e-12 mg/L; parallel forest reduction"
                row["max_abs_save_load_error_mg_l"] = float(np.abs(loaded-original).max())
            rows.append(row)
            print(row, flush=True)
    write_json(args.root/"verification/geographical_facade_replay.json", {
        "dataset_hash": sha256_file(DATASET), "verifier_sha256": sha256_file(__file__),
        "scope": "portable replay of previously evaluated geographical fits; no new training or testing claim",
        "runs": rows})


if __name__ == "__main__":
    main()
