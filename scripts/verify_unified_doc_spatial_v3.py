"""Reload episodic DOC components and reproduce the complete query products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json
from run_unified_doc_spatial_v2 import make_predictions, read_source
from run_unified_doc_spatial_v3 import ROOT

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_station_adapter import EpisodicStationProjector
from river_graph.models.regularized_station_fusion import RegularizedStationFusion
from river_graph.models.support_shape_adapter import SupportShapeAdapter


def verify(root):
    runtime = verify_runtime_snapshot(root)
    records = []
    for split_seed in (142, 143, 144):
        for seed in (42, 43, 44):
            run = root / "runs" / f"split{split_seed}_seed{seed}"
            config = json.loads((run / "config.json").read_text())
            completion = verify_files(run, "complete.json", config)
            sidecar = json.loads((run / "predictions.meta.json").read_text())
            expected = {
                "config": config, "config_hash": completion["config_hash"],
                "runtime_snapshot_hash": runtime, "dataset_sha256": config["dataset_hash"],
                "mask_sha256": config["mask_hash"],
                "run_identity_sha256": run_identity_sha256(completion["config_hash"], config["started_at"], runtime),
                "prediction_sha256": completion["files"]["predictions.parquet"],
            }
            if any(sidecar.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Prediction identity changed: {run}")
            for name, value in sidecar["model_files"].items():
                if completion["files"].get(name) != value:
                    raise ValueError(f"Changed model binding: {run / name}")
            source = Path(config["source_run"])
            for key in ("source", "fusion"):
                if sha256_file(Path(config[f"{key}_run"]) / "complete.json") != config[f"{key}_completion_hash"]:
                    raise ValueError(f"Changed {key} package")
            _, _, dataset, split, full = read_source(source)
            for name in ("gru", "tree"):
                model = EpisodicStationProjector.from_dict(json.loads((run / f"{name}_projector.json").read_text()))
                if model.epochs_run_ > config["epochs"]:
                    raise ValueError("Projector exceeded declared training budget")
            context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
            fusion = RegularizedStationFusion.from_dict(json.loads((run / "fusion.json").read_text()))
            adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in
                        json.loads((run / "adapters.json").read_text()).items()}
            with np.load(run / "representations.npz", allow_pickle=False) as saved:
                shapes = {name: saved[name] for name in saved.files}
            with np.load(run / "source_oof.npz", allow_pickle=False) as saved:
                oof = saved["pred_z"].ravel()
            train_mask = np.zeros(len(oof), dtype=bool)
            train_mask[split["train"]] = True
            if not np.isfinite(oof[train_mask]).all() or not np.isnan(oof[~train_mask]).all():
                raise ValueError("OOF coverage differs from source training cells")
            months = dataset["y"].shape[1]
            truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
            support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
            labels = np.full_like(truth, np.nan)
            labels[support] = truth[support]
            replay = make_predictions(full, {"context": context, "fusion": fusion.predict(context, temporal)},
                                      shapes, adapters, labels, split, months)
            stored = pd.read_parquet(run / "predictions.parquet")
            keys = ["model_name", "k", "cell"]
            replay, stored = (frame.sort_values(keys).reset_index(drop=True) for frame in (replay, stored))
            pd.testing.assert_frame_equal(replay, stored[replay.columns], check_exact=True)
            np.testing.assert_array_equal(stored.y_true, truth[stored.cell.to_numpy()])
            records.append({"split_seed": split_seed, "seed": seed, "rows": len(stored),
                            "reload_predictions": "bitwise_equal", "identity": "ok"})
    write_json(root / "verification.json", {"complete": True, "runs": records,
               "runtime_snapshot_hash": runtime, "verifier_sha256": sha256_file(__file__)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(f"Verified {len(verify(args.root))}/9 products with bitwise-equal reloaded predictions")


if __name__ == "__main__":
    main()
