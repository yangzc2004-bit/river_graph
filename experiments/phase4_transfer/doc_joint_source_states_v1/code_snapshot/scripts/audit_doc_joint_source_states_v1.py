"""Audit source-only pH/conductance labels for DOC representation learning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_unmonitored_residual_v1 import ROOT as PARENT
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_joint_source_states_v1")
DATASETS = {name: Path(f"data/processed/mississippi_graph_{name}_st357.pt")
            for name in ("ph", "spec_conductance")}


def main():
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    data = {name: torch.load(path, weights_only=False, map_location="cpu") for name, path in DATASETS.items()}
    rows, roles = [], {}
    for partition in (142, 143, 144):
        parent = PARENT/"runs"/f"split{partition}_seed42"
        config = json.loads((parent/"config.json").read_text())
        verify_files(parent, "complete.json", config)
        doc = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(config["mask_path"], allow_pickle=False) as masks:
            source = np.unique(masks["train"]//doc["y"].shape[1])
            receiving = np.unique(np.r_[masks["val"], masks["test"]]//doc["y"].shape[1])
        if np.intersect1d(source, receiving).size:
            raise ValueError("this audit requires whole-station source/receiving separation")
        roles[str(partition)] = {"mask_path": config["mask_path"], "mask_hash": config["mask_hash"],
            "source_station_ids": source.tolist(), "receiving_station_values_inspected": False}
        for name, auxiliary in data.items():
            for key in ("site_no", "months"):
                np.testing.assert_array_equal(auxiliary[key], doc[key])
            if auxiliary["y"].shape != doc["y"].shape:
                raise ValueError("auxiliary label grid differs from the DOC task")
            mask = np.asarray(auxiliary["y_mask"])[source].astype(bool)
            values = np.asarray(auxiliary["y"])[source][mask]
            if not np.isfinite(values).all() or not len(values):
                raise ValueError("source auxiliary labels must be finite and available")
            if name == "spec_conductance" and (values < 0).any():
                raise ValueError("log1p conductance requires nonnegative valid labels")
            rows.append({"split_seed": partition, "analyte": name, "source_stations": len(source),
                "source_stations_with_labels": int(mask.any(axis=1).sum()), "source_label_cells": int(mask.sum()),
                "source_min": float(values.min()), "source_max": float(values.max()),
                "source_median": float(np.median(values))})
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"source_label_availability.csv", index=False)
    write_json(root/"data_audit.json", {"analyzer_sha256": sha256_file(__file__),
        "sources": {str(path): sha256_file(path) for path in DATASETS.values()},
        "role_definitions": roles, "station_month_alignment": True,
        "target_station_chemistry_used": False, "new_model_fitting": False,
        "label_statistics_scope": "source training stations only; joint fit statistics from source training stations only",
        "note": "older cross-analyte station selection used receiving chemistry; do not reuse that setup for the water-quality-free K0 task"})
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
