"""Value-blind support-schedule pilot for spatial few-shot DOC transfer.

Existing source-pool-40 ExtraTrees predictions are reused when available.  A
small internal station-heldout block selects a schedule and K in {3, 5}; all
outer candidates are then scored in one batch on a common query population.
The residual shrinkage factors (K3=.5, K5=.75) are carried over from the
previous nested support-residual experiment and are not selected on E3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    load_split,
    predict_query_models,
    station_descriptors,
)

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
SEEDS = (42, 43, 44)
KS = (3, 5)
ALPHA = {3: 0.50, 5: 0.75}
OUTER_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
INTERNAL_MASK = Path(
    "experiments/phase4_transfer/kgml_local_transport_v1/"
    "spatial_validation_e3_supportmatched/pilot/masks/"
    "doc__e3_spatial_validation_supportmatched.npz"
)
SOURCE_PRODUCT = Path(
    "experiments/phase4_transfer/spatial_adaptation/"
    "source_selection_v1_5seed/product"
)
SCHEDULES = ("fixed_five", "recent_five", "seasonal_five", "flow_quantile_five")


def schedule_for_station(
    cells: np.ndarray,
    name: str,
    flow: np.ndarray,
) -> np.ndarray:
    """Return five value-blind support cells in chronological order."""
    cells = np.sort(np.asarray(cells, dtype=np.int64))
    if len(cells) < 5:
        raise ValueError("each target station needs five candidate months")
    if name == "fixed_five":
        pos = np.linspace(0, len(cells) - 1, 5, dtype=int)
        # Keep the historical order so K=3 is a nested prefix.
        return cells[pos[[0, 2, 4, 1, 3]]]
    if name == "recent_five":
        return cells[-5:]
    month = cells % 12
    if name == "seasonal_five":
        phases = np.asarray([0, 3, 6, 9, 11])
        remaining = list(range(len(cells)))
        chosen: list[int] = []
        for phase in phases:
            j = min(remaining, key=lambda i: min((int(month[i]) - int(phase)) % 12,
                                                  (int(phase) - int(month[i])) % 12))
            chosen.append(j)
            remaining.remove(j)
        return cells[np.asarray(chosen)]
    if name == "flow_quantile_five":
        values = np.asarray(flow[cells], dtype=float)
        qs = np.quantile(values[np.isfinite(values)], [0.1, 0.3, 0.5, 0.7, 0.9])
        remaining = list(range(len(cells)))
        chosen = []
        for q in qs:
            j = min(remaining, key=lambda i: abs(values[i] - q))
            chosen.append(j)
            remaining.remove(j)
        return cells[np.asarray(chosen)]
    raise ValueError(name)


def build_schedule(cells: np.ndarray, name: str, flow: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(station): schedule_for_station(
            cells[cells // T == station], name, flow,
        )
        for station in np.unique(cells // T)
    }


def source_predictions(
    data: dict,
    split: dict,
    cells: np.ndarray,
    desc: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    return predict_query_models(
        data, split, np.asarray(split["train"], dtype=np.int64), cells,
        fit_x, eval_x, desc, 40, seed, n_estimators, 4, 4,
    )


def make_predictions(
    y: np.ndarray,
    base: np.ndarray,
    cells: np.ndarray,
    schedule: dict[int, np.ndarray],
    k: int,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    by_cell = {int(c): i for i, c in enumerate(cells)}
    support = np.sort(np.concatenate([v[:k] for v in schedule.values()]))
    if np.intersect1d(support, query).size:
        raise ValueError("support/query overlap")
    delta: dict[int, float] = {}
    for station, scells in schedule.items():
        idx = np.asarray([by_cell[int(c)] for c in scells[:k]], dtype=int)
        delta[station] = float(np.mean(
            np.log1p(np.maximum(y[idx], 0.0))
            - np.log1p(np.maximum(base[idx], 0.0)),
        ))
    qidx = np.asarray([by_cell[int(c)] for c in query], dtype=int)
    z = np.log1p(np.maximum(base[qidx], 0.0))
    z += ALPHA[k] * np.asarray([delta.get(int(c) // T, 0.0) for c in query])
    return query, np.maximum(np.expm1(z), 0.0)


def common_query(cells: np.ndarray, schedules: dict[str, dict[int, np.ndarray]]) -> np.ndarray:
    reserve = np.concatenate([
        np.concatenate(list(per_station.values())) for per_station in schedules.values()
    ])
    return np.setdiff1d(cells, np.unique(reserve), assume_unique=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-estimators", type=int, default=120)
    args = p.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    y_all = data["y"].numpy().reshape(-1).astype(float)
    flow_all = data["x"].numpy()[:, :, 1].reshape(-1).astype(float)
    desc = station_descriptors(data)
    splits = {
        "internal_val": load_split(INTERNAL_MASK),
        "outer_test": load_split(OUTER_MASK),
    }
    rows: list[dict] = []
    for role, split in splits.items():
        target_cells = np.sort(np.asarray(split["val" if role == "internal_val" else "test"], dtype=np.int64))
        schedules = {
            name: build_schedule(target_cells, name, flow_all)
            for name in SCHEDULES
        }
        query = common_query(target_cells, schedules)
        for seed in SEEDS:
            if role == "outer_test":
                source = pd.read_parquet(SOURCE_PRODUCT / f"predictions_seed{seed}.parquet")
                source = source.sort_values("cell").reset_index(drop=True)
                cells = source.cell.to_numpy(dtype=np.int64)
                base = source.y_pred.to_numpy(float)
                y = source.y_true.to_numpy(float)
                if not np.array_equal(cells, target_cells):
                    raise ValueError("outer source product does not match E3 test cells")
            else:
                cells = target_cells
                base = source_predictions(data, split, cells, desc, seed, args.n_estimators)
                y = y_all[cells]
            for name, schedule in schedules.items():
                for k in KS:
                    qcells, pred = make_predictions(y, base, cells, schedule, k, query)
                    idx = np.searchsorted(cells, qcells)
                    delta = np.asarray([
                        float(np.log1p(max(y[cells.tolist().index(int(c))], 0.0))
                        - np.log1p(max(base[cells.tolist().index(int(c))], 0.0)))
                        for c in qcells
                    ])
                    rows.extend({
                        "role": role, "seed": seed, "schedule": name, "k": k,
                        "cell": int(cell), "station": int(cell) // T,
                        "y_true": float(truth), "base_pred": float(base[j]),
                        "y_pred": float(value), "support_delta_query": float(d),
                        "query_n": len(qcells),
                    } for cell, truth, value, d, j in zip(
                        qcells, y[idx], pred, delta, idx, strict=True,
                    ))
    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out_dir / "predictions.parquet", index=False)
    score = (frame.groupby(["role", "schedule", "k"], as_index=False)
             .apply(lambda x: pd.Series({
                 **metrics(x.y_true.to_numpy(), x.y_pred.to_numpy()),
                 "query_n": int(x.query_n.iloc[0]),
             }), include_groups=False).reset_index(drop=True))
    score.to_csv(args.out_dir / "scores.csv", index=False)
    internal = score[score.role == "internal_val"].sort_values("mae", kind="stable")
    selected = internal.iloc[0].to_dict()
    pd.DataFrame([selected]).to_csv(args.out_dir / "selected.csv", index=False)
    outer = score[score.role == "outer_test"]
    outer.to_csv(args.out_dir / "outer_all.csv", index=False)
    chosen = outer[(outer.schedule == selected["schedule"]) & (outer.k == selected["k"])]
    chosen.to_csv(args.out_dir / "outer_selected.csv", index=False)
    (args.out_dir / "spec.json").write_text(json.dumps({
        "schedules": list(SCHEDULES), "K": list(KS), "seeds": list(SEEDS),
        "source_pool": 40, "alpha_by_k": ALPHA,
        "selection": "internal support-matched station-heldout validation",
        "outer": "single batch score on a common query excluding the union of five candidate cells",
        "reference": "current fixed five-point K=5 outer MAE approximately 2.02",
    }, indent=2) + "\n")
    (args.out_dir / "verdict.md").write_text(
        "# Support schedule pilot\n\n"
        f"Internal selection chose `{selected['schedule']}` with K={int(selected['k'])} "
        f"(MAE {selected['mae']:.4f}) using source-pool 40 and the preselected "
        "mean-residual shrinkage factors. The complete outer table is in "
        "`outer_all.csv`; `outer_selected.csv` is the one selected row.\n\n"
        "This pilot is a value-blind schedule diagnostic. It uses a common "
        "outer query obtained by excluding the union of each schedule's five "
        "candidate cells. If no schedule beats the existing fixed K=5 result "
        "(about MAE 2.02 on its original paired query), the schedule line is "
        "closed. No labels are used to construct schedules.\n"
    )
    print(internal.to_string(index=False))
    print(outer.to_string(index=False))


if __name__ == "__main__":
    main()
