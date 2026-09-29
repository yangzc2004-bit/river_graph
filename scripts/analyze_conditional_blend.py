"""Post-hoc validation-only age-gated RF/GNN blend diagnostic.

This does not retrain either model. For each saved run, alpha_RF is selected
separately within sufficiently populated observation-age groups using only the
saved validation predictions, then applied once to the saved test predictions.
It is exploratory evidence for whether a future conditional gate is worth
implementing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.graph_upgrade_v2 import observation_statistics
from river_graph.experiments.transfer import DATASETS

ALPHAS = np.linspace(0, 1, 21)
MIN_VALIDATION_CELLS = 20


def age_groups(dataset, split, cells):
    y = dataset["y"].float()
    n, t = y.shape
    visible = torch.zeros(n * t, dtype=torch.bool)
    for role in ("train", "context"):
        visible[np.asarray(split.get(role, []), dtype=int)] = True
    stats = observation_statistics(y, visible.reshape(n, t), dataset["edge_index"].long())
    age = np.rint(np.expm1(stats[1].T.numpy().reshape(-1) * np.log1p(12)))
    known = stats[2].T.numpy().reshape(-1).astype(bool)
    age = age[np.asarray(cells, dtype=int)]
    known = known[np.asarray(cells, dtype=int)]
    return np.select([~known, age == 0, age <= 3, age <= 12],
                     ["never_observed", "fresh", "recent_1_3", "seasonal_4_12"],
                     default="old_over_12")


def select(y, rf, gnn):
    values = []
    for alpha in ALPHAS:
        values.append((float(np.abs(y - (alpha*rf + (1-alpha)*gnn)).mean()), float(alpha)))
    return min(values, key=lambda p: (p[0], -p[1]))[1]


def run(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        config = meta["config"]
        run = meta_path.parent
        val = pd.read_parquet(run / "val_predictions.parquet")
        test = pd.read_parquet(run / "test_predictions.parquet")
        dataset = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
        with np.load(config["mask_path"], allow_pickle=False) as archive:
            split = {key: archive[key] for key in archive.files}
        val_group = age_groups(dataset, split, val.cell)
        test_group = age_groups(dataset, split, test.cell)
        global_alpha = float(meta["alpha_rf"])
        alpha_by_group = {}
        for group in np.unique(val_group):
            select_group = val_group == group
            alpha_by_group[group] = (
                select(val.y_true.to_numpy()[select_group], val.rf_pred.to_numpy()[select_group],
                       val.gnn_pred.to_numpy()[select_group])
                if select_group.sum() >= MIN_VALIDATION_CELLS else global_alpha
            )
        alpha = np.asarray([alpha_by_group[g] for g in test_group])
        pred = alpha * test.rf_pred.to_numpy() + (1-alpha) * test.gnn_pred.to_numpy()
        row = {"mask": config["mask"], "seed": config["seed"],
               "global_alpha_rf": global_alpha,
               "conditional_alpha_rf": json.dumps(alpha_by_group, sort_keys=True),
               "conditional_mae": float(np.abs(test.y_true.to_numpy()-pred).mean()),
               "global_mae": float(np.abs(test.y_true.to_numpy()-test.blend_pred.to_numpy()).mean()),
               "rf_mae": float(np.abs(test.y_true.to_numpy()-test.rf_pred.to_numpy()).mean()),
               "groups_test": json.dumps({g: int((test_group == g).sum()) for g in np.unique(test_group)}, sort_keys=True)}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(root / "analysis" / "conditional_age_gate_runs.csv", index=False)
    summary = out.groupby("mask", as_index=False).agg(
        seeds=("seed", "count"), rf_mae=("rf_mae", "mean"),
        global_mae=("global_mae", "mean"), conditional_mae=("conditional_mae", "mean"))
    summary["conditional_gain_vs_global_pct"] = 100 * (1-summary.conditional_mae/summary.global_mae)
    summary.to_csv(root / "analysis" / "conditional_age_gate_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(out[["mask", "seed", "global_alpha_rf", "conditional_alpha_rf", "global_mae", "conditional_mae"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
