"""Replay learned DOC memory and its held-station query products from saved weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json
from run_unified_doc_spatial_v2 import make_predictions, read_source
from run_unified_doc_spatial_v4 import HEADS, ROOT

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.episodic_temporal_features import extract_temporal_inputs
from river_graph.models.regularized_station_fusion import RegularizedStationFusion
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor


def verify(root, split_seeds=(142, 143, 144), seeds=(42, 43, 44)):
    runtime = verify_runtime_snapshot(root)
    records = []
    for split_seed in split_seeds:
        for seed in seeds:
            run = root / "runs" / f"split{split_seed}_seed{seed}"
            config = json.loads((run / "config.json").read_text())
            torch.set_num_threads(config["torch_threads"])
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
                    raise ValueError(f"Model binding changed: {run / name}")
            for key in ("prior", "source", "fusion"):
                if sha256_file(Path(config[f"{key}_run"]) / "complete.json") != config[f"{key}_completion_hash"]:
                    raise ValueError(f"Changed {key} package")
            source = Path(config["source_run"])
            _, _, dataset, split, full = read_source(source)
            expert = UnifiedDOCReconstructor.load(source, dataset, split)
            features = extract_temporal_inputs(expert, dataset, split)
            full_inputs = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
            memory = EpisodicTemporalAdapter.from_payload(torch.load(run / "memory.pt", weights_only=False))
            if memory.to_dict() != json.loads((run / "memory.json").read_text()):
                raise ValueError("Memory summary differs from checkpoint")
            with np.load(run / "representations.npz", allow_pickle=False) as saved:
                shapes = {name: saved[name] for name in saved.files}
            for name, values in (("gru_tuned_anchor", memory.transform(full_inputs)),
                                 ("gru_frozen_anchor", memory.transform_initial(full_inputs))):
                np.testing.assert_array_equal(values.reshape(-1, 2), shapes[name])
            prior = Path(config["prior_run"])
            prior_config = json.loads((prior / "config.json").read_text())
            verify_files(prior, "complete.json", prior_config)
            with np.load(prior / "representations.npz", allow_pickle=False) as saved:
                for name in HEADS[:3]:
                    np.testing.assert_array_equal(shapes[name], saved[name])
            context, temporal = (full[f"{name}_pred"].to_numpy() for name in ("context", "temporal"))
            fusion = RegularizedStationFusion.from_dict(json.loads((run / "fusion.json").read_text()))
            adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in
                        json.loads((run / "adapters.json").read_text()).items()}
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
                            "recurrent_feature_reload": "bitwise_equal",
                            "prediction_reload": "bitwise_equal", "identity": "ok"})
            print(f"Verified {run.name}", flush=True)
    write_json(root / "verification.json", {"complete": True, "runs": records,
               "runtime_snapshot_hash": runtime, "verifier_sha256": sha256_file(__file__)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[142, 143, 144])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    print(f"Verified {len(verify(args.root, args.split_seeds, args.seeds))} products")


if __name__ == "__main__":
    main()
