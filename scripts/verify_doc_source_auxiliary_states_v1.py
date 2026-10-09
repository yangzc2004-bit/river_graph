"""Recompute source-fold label statistics and auxiliary checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from run_doc_source_auxiliary_states_v1 import AUXILIARY_DATASETS, ROOT
from run_unified_doc_spatial import verify_files, write_json

from river_graph.models.source_auxiliary_pretraining import (
    prepare_auxiliary_labels,
    source_auxiliary_targets,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("complete all nine source packages before protocol verification")
    auxiliary = {name: torch.load(path, weights_only=False, map_location="cpu") for name, path in AUXILIARY_DATASETS.items()}
    rows = []
    for path in complete:
        run = path.parent
        config = json.loads((run/"config.json").read_text())
        verify_files(run, "complete.json", config)
        doc = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        months = doc["y"].shape[1]
        with np.load(config["mask_path"], allow_pickle=False) as split:
            source = np.unique(split["train"]//months)
            receiving = np.unique(np.r_[split["val"], split["test"]]//months)
        if np.intersect1d(source, receiving).size:
            raise ValueError("source auxiliary sites must exclude all receiving sites")
        target, valid = source_auxiliary_targets(auxiliary, source, doc["site_no"], doc["months"])
        for name in ("source_shuffle", "source_auxiliary"):
            summary = json.loads((run/f"{name}_auxiliary.json").read_text())
            np.testing.assert_array_equal(summary["source_station_ids"], source)
            held_ids = np.asarray(summary["validation_station_ids"])
            if not len(held_ids) or not np.isin(held_ids, source).all():
                raise ValueError("auxiliary validation sites must belong to source only")
            held = np.isin(source, held_ids)
            _, fit_mask, val_mask, mean, scale = prepare_auxiliary_labels(target, valid, held)
            for actual, expected in ((summary["target_mean"], mean), (summary["target_scale"], scale),
                    (summary["fit_label_counts"], fit_mask.sum((0, 1))),
                    (summary["validation_label_counts"], val_mask.sum((0, 1)))):
                np.testing.assert_array_equal(actual, expected)
            trace = summary["trace"]
            if [row["epoch"] for row in trace] != list(range(31)):
                raise ValueError("both auxiliary arms must complete the fixed30 epochs")
            selected = min(trace, key=lambda row: row["validation_mse"])
            if (selected["epoch"] != summary["best_epoch"]
                    or selected["validation_mse"] != summary["validation_mse"]):
                raise ValueError("auxiliary checkpoint does not match source-fold selection")
            for row in trace:
                np.testing.assert_allclose(row["validation_mse"],
                    .5*(row["validation_ph_mse"]+row["validation_log_ec_mse"]), rtol=0, atol=1e-14)
            weights = torch.load(run/f"{name}_auxiliary.pt", weights_only=False, map_location="cpu")
            if not all(torch.isfinite(v).all() for state in weights.values() for v in state.values()):
                raise ValueError("nonfinite saved auxiliary parameters")
            rows.append({"run": run.name, "arm": name, "source_scalers_recomputed": True,
                "receiving_stations_excluded": True, "auxiliary_checkpoint_selection_verified": True,
                "best_epoch": summary["best_epoch"]})
    write_json(args.root/"verification/auxiliary_protocol.json", rows)
    print(f"Verified source normalization, held-station roles and auxiliary selection for {len(rows)} fits.")


if __name__ == "__main__":
    main()
