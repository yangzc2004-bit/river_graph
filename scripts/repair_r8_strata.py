"""Rebuild only R8 diagnostic strata with the matched river feature graph."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.graph_upgrade_v2 import observation_statistics
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.transfer import DATASETS

ROOT = Path("experiments/phase4_transfer/graph_upgrade_v2/continuation_r8_rf_nomsg_blend")


def main():
    changed = []
    for meta_path in sorted((ROOT / "runs").glob("*/meta.json")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text())
        config = meta["config"]
        data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
        with np.load(config["mask_path"], allow_pickle=False) as archive:
            split = {key: archive[key] for key in archive.files}
        n, t = data["y"].shape
        visible = torch.zeros(n*t, dtype=torch.bool)
        for role in ("train", "val", "context"):
            visible[split.get(role, np.array([], dtype=np.int64))] = True
        stats = observation_statistics(data["y"].float(), visible.reshape(n, t),
                                       data["edge_index"].long())
        frame = pd.read_parquet(run / "test_predictions.parquet")
        cells = frame.cell.to_numpy(dtype=int)
        age = np.rint(np.expm1(stats[1].T.numpy().reshape(-1)[cells] * np.log1p(12)))
        known = stats[2].T.numpy().reshape(-1)[cells].astype(bool)
        frame["observation_age_group"] = np.select(
            [~known, age == 0, age <= 3, age <= 12],
            ["never_observed", "fresh", "recent_1_3", "seasonal_4_12"],
            default="old_over_12",
        )
        support = stats[-2].T.numpy().reshape(-1)[cells]
        frame["upstream_support_group"] = np.where(support > 0, "visible_upstream", "none")
        frame.to_parquet(run / "test_predictions.parquet", index=False)
        meta["artifacts"]["test_predictions.parquet"] = sha256_file(run / "test_predictions.parquet")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        changed.append(str(run))
    (ROOT / "strata_repair.json").write_text(json.dumps({
        "changed_runs": changed,
        "reason": "R8 diagnostics use river feature edge set while spatial messages remain empty; predictions unchanged",
    }, indent=2) + "\n")
    print(f"repaired {len(changed)} runs")


if __name__ == "__main__":
    main()
