"""Reload fitted DOC adapters and reproduce every saved outer prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json
from run_unified_doc_spatial_v2 import ROOT, make_predictions, read_source

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.regularized_station_fusion import RegularizedStationFusion
from river_graph.models.support_shape_adapter import SupportShapeAdapter


def verify(root):
    runtime_hash = verify_runtime_snapshot(root)
    results = []
    for split_seed in (142, 143, 144):
        for seed in (42, 43, 44):
            run = root / "runs" / f"split{split_seed}_seed{seed}"
            config = json.loads((run / "config.json").read_text())
            complete = verify_files(run, "complete.json", config)
            sidecar = json.loads((run / "predictions.meta.json").read_text())
            expected = {
                "config": config, "config_hash": complete["config_hash"],
                "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
                "runtime_snapshot_hash": runtime_hash,
                "run_identity_sha256": run_identity_sha256(
                    complete["config_hash"], config["started_at"], runtime_hash),
                "prediction_sha256": complete["files"]["predictions.parquet"],
            }
            if any(sidecar.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Prediction identity differs: {run}")
            for name in ("fusion.json", "adapters.json", "bases.json", "representations.npz"):
                if sidecar["model_files"].get(name) != complete["files"][name]:
                    raise ValueError(f"Adapter binding differs: {run / name}")
            source = Path(config["source_run"])
            if sha256_file(source / "complete.json") != config["source_completion_hash"]:
                raise ValueError(f"Source experts changed: {source}")
            _, _, dataset, split, full = read_source(source)
            context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
            fusion = RegularizedStationFusion.from_dict(json.loads((run / "fusion.json").read_text()))
            adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in
                        json.loads((run / "adapters.json").read_text()).items()}
            with np.load(run / "representations.npz", allow_pickle=False) as saved:
                shapes = {name: saved[name] for name in saved.files}
            months = dataset["y"].shape[1]
            truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
            support, _ = support_query_cells(split, target_role="test", k=5, n_months=months)
            labels = np.full_like(truth, np.nan)
            labels[support] = truth[support]
            replay = make_predictions(full, {"context": context, "fusion": fusion.predict(context, temporal)},
                                      shapes, adapters, labels, split, months)
            stored = pd.read_parquet(run / "predictions.parquet")
            keys = ["model_name", "k", "cell"]
            stored, replay = (frame.sort_values(keys).reset_index(drop=True) for frame in (stored, replay))
            pd.testing.assert_frame_equal(stored[replay.columns], replay, check_exact=True)
            np.testing.assert_array_equal(stored.y_true, truth[stored.cell.to_numpy()])
            results.append({"split_seed": split_seed, "seed": seed, "rows": len(replay),
                            "identity": "ok", "reload_predictions": "bitwise_equal"})
    report = {"complete": True, "runs": results, "runtime_snapshot_hash": runtime_hash,
              "verification_script_sha256": sha256_file(__file__)}
    write_json(root / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = verify(args.root)
    print(f"Verified {len(report['runs'])}/9 runs; all reloaded predictions are bitwise equal")


if __name__ == "__main__":
    main()
