"""Use observed pH/EC station profiles to select DOC source stations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    load_split,
    predict_query_models,
    station_descriptors,
)
from sklearn.preprocessing import StandardScaler

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
KS = (20, 40, 80, 160)


def analyte_profile(path: str) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    y = data["y"].numpy().astype(float)
    mask = data["y_mask"].numpy().astype(float)
    blocks: list[np.ndarray] = []
    for value in (y,):
        den = np.maximum(mask.sum(1), 1.0)
        mean = (value * mask).sum(1) / den
        sd = np.sqrt(((value - mean[:, None]) ** 2 * mask).sum(1) / den)
        blocks.extend([mean, sd, mask.mean(1)])
        for q in (0.1, 0.5, 0.9):
            blocks.append(np.asarray([
                np.quantile(value[i, mask[i] > 0], q) if mask[i].any() else 0.0
                for i in range(value.shape[0])
            ]))
    return np.stack(blocks, axis=1)


def run(args: argparse.Namespace) -> None:
    doc = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    ph = analyte_profile(DATASETS["ph"])
    ec = analyte_profile(DATASETS["spec_conductance"])
    base = station_descriptors(doc)
    desc = StandardScaler().fit_transform(np.concatenate([base, ph, ec], axis=1)).astype(np.float32)
    outer = load_split(MASK)
    internal, _rows, _ = e3_internal_split(doc["y_mask"].numpy(), outer["test"])
    y = doc["y"].numpy().reshape(-1)
    rows: list[dict] = []
    for seed in args.seeds:
        for split, role in ((internal, "val"), (outer, "test")):
            cells = np.asarray(split[role], dtype=np.int64)
            fit_x = build_rf_features(doc, split, FIT_ROLES, target_transform="log1p", include_network=True)
            eval_x = build_rf_features(doc, split, TEST_ROLES, target_transform="log1p", include_network=True)
            for k in KS:
                pred = predict_query_models(doc, split, np.asarray(split["train"], dtype=np.int64), cells,
                                             fit_x, eval_x, desc, k, seed, args.n_estimators, 4, 4)
                rows.append({"seed": seed, "role": role, "k": k, **metrics(y[cells], pred)})
    out = pd.DataFrame(rows)
    selected = int(out[out.role == "val"].groupby("k").mae.mean().sort_values().index[0])
    test = out[(out.role == "test") & (out.k == selected)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "summary.txt").write_text(f"selected_k={selected}\n{out.to_string(index=False)}\n")
    print(out[out.role == "val"].groupby("k").mae.mean())
    print("selected", selected)
    print(test.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
