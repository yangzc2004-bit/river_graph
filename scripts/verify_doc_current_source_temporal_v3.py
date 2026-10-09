"""Replay saved current-source temporal states and causal input boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_current_source_temporal_v3_r1 import MASKS, ROOT
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)
from river_graph.models.station_adapted_hybrid import StationAdaptedHybrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    records = []
    for mask in MASKS:
        for seed in (42, 43, 44):
            run = args.root/"runs"/f"{mask}_seed{seed}"
            config = json.loads((run/"config.json").read_text())
            if config["runtime_snapshot_hash"] != runtime:
                raise ValueError("fitting runtime differs")
            verify_files(run, "complete.json", config)
            if (run/"reference_reuse.json").exists():
                for name, expected in json.loads((run/"reference_reuse.json").read_text())["files"].items():
                    if sha256_file(name) != expected:
                        raise ValueError("preserved reused reference changed")
            with np.load(run/"inputs.npz", allow_pickle=False) as saved:
                inputs = {key[5:]: saved[key].copy() for key in saved.files if key.startswith("full_")}
            with np.load(run/"components.npz", allow_pickle=False) as saved:
                environment, expected = saved["environment"].copy(), saved["available_native"].copy()
                expected_fusion = saved["available_fusion"].copy()
            model = AvailableSourceAttentionResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
            actual = model.predict(inputs, environment)
            np.testing.assert_array_equal(actual, expected)
            fusion = StationAdaptedHybrid.from_dict(json.loads((run/"fusion.json").read_text()))
            combined = fusion.predict_components(environment.ravel(), actual.ravel())["hybrid"].reshape(environment.shape)
            np.testing.assert_array_equal(combined, expected_fusion)
            boundary = environment.shape[1]//2
            changed = {key: value.copy() for key, value in inputs.items()}
            for key in ("raw", "extra", "donor_hydro_bank"):
                changed[key][:, boundary:] += .1
            # Perturb only real observed future donors. Invalid candidates and
            # the unit zero-value prior must keep their operator contract.
            changed["donor_values"][:, boundary:, :-1] += .1*changed["donor_valid"][:, boundary:, :-1]
            checked = model.predict(changed, environment)
            np.testing.assert_array_equal(checked[:, :boundary], actual[:, :boundary])
            prior = pd.read_parquet(Path(config["parent_run"])/"predictions.parquet")
            frame = pd.read_parquet(run/"predictions.parquet")
            old = frame[~frame.model_name.isin(("available_native", "available_fusion"))][prior.columns]
            keys = ["model_name", "cell"]
            pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),
                prior.sort_values(keys).reset_index(drop=True), check_exact=True)
            records.append({"run": run.name, "full_grid_replay": "bitwise", "fusion_replay": "bitwise",
                "future_inputs": "past predictions unchanged", "old_rows_preserved": len(prior)})
    write_json(args.root/"verification.json", {"runtime_snapshot_hash": runtime, "status": "passed",
        "runs": records, "verifier_sha256": sha256_file(__file__)})
    print("All six current-source temporal refits replayed and verified", flush=True)


if __name__ == "__main__":
    main()
