"""Support-aware regional gate pilot for the fixed E3 spatial transfer task.

The pilot combines the existing spatial KGML prediction with the source-pool
40 ExtraTrees regional expert.  All model selection is done on the nested
internal station holdout; the frozen E3 test is scored once.  No model is
retrained here: the script only consumes existing predictions and computes
support residual corrections from the five opened target labels.
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
OUTER_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
KGML_ROOT = Path(
    "experiments/phase4_transfer/kgml_local_transport_v1/"
    "spatial_validation_e3_supportmatched/pilot/runs"
)
INTERNAL_MASK = KGML_ROOT.parent / "masks" / "doc__e3_spatial_validation_supportmatched.npz"
SOURCE_PRODUCT = Path(
    "experiments/phase4_transfer/spatial_adaptation/"
    "source_selection_v1_5seed/product"
)
SEEDS = (42, 43, 44)
KS = (0, 5)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_THRESHOLDS = (0.0, 0.1, 0.25, 0.5, 1.0)


def support_schedule(cells: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Open first/middle/last/quarter/month positions; keep a paired query."""
    cells = np.sort(np.asarray(cells, dtype=np.int64))
    opened: list[int] = []
    reserved: list[int] = []
    for station in np.unique(cells // T):
        local = cells[cells // T == station]
        if len(local) <= 5:
            raise ValueError("target station has fewer than five cells")
        positions = np.linspace(0, len(local) - 1, 5, dtype=int)
        ordered = local[positions[[0, 2, 4, 1, 3]]]
        reserved.extend(ordered.tolist())
        opened.extend(ordered[:k].tolist())
    support = np.asarray(sorted(set(opened)), dtype=np.int64)
    # Every K uses the same query population.  For K=0 the five reserved
    # months remain hidden; they are simply not opened as support labels.
    query = np.setdiff1d(cells, np.asarray(sorted(set(reserved))), assume_unique=True)
    return support, query


def support_correction(
    y: np.ndarray,
    base: np.ndarray,
    all_cells: np.ndarray,
    support: np.ndarray,
) -> dict[int, float]:
    """Station mean log residual using only opened support labels."""
    if not len(support):
        return {}
    by_cell = {int(c): i for i, c in enumerate(all_cells)}
    residual: dict[int, list[float]] = {}
    for cell in support:
        idx = by_cell[int(cell)]
        station = int(cell) // T
        residual.setdefault(station, []).append(
            float(np.log1p(max(y[idx], 0.0)) - np.log1p(max(base[idx], 0.0))),
        )
    return {s: float(np.mean(v)) for s, v in residual.items()}


def apply_residual(
    base: np.ndarray,
    cells: np.ndarray,
    station_delta: dict[int, float],
    alpha: float = 1.0,
) -> np.ndarray:
    z = np.log1p(np.maximum(np.asarray(base, dtype=float), 0.0))
    for i, cell in enumerate(cells):
        z[i] += alpha * station_delta.get(int(cell) // T, 0.0)
    return np.maximum(np.expm1(z), 0.0)


def load_kgml(path: Path, role: str) -> pd.DataFrame:
    if role == "val":
        frame = pd.read_parquet(path / "val_predictions.parquet")
    elif role == "test":
        frame = pd.read_parquet(path / "test_predictions.parquet")
    else:
        raise ValueError(role)
    return frame.sort_values("cell").reset_index(drop=True)


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


def build_rows(
    data: dict,
    split: dict,
    role: str,
    out_rows: list[dict],
    desc: np.ndarray,
    n_estimators: int,
) -> None:
    for seed in SEEDS:
        if role == "outer_test":
            run = KGML_ROOT / (
                f"residual_context_msgdelta__doc__e3_spatial_validation_supportmatched__seed{seed}"
            )
            kg = load_kgml(run, "test")
            src_path = SOURCE_PRODUCT / f"predictions_seed{seed}.parquet"
            src = pd.read_parquet(src_path).sort_values("cell").reset_index(drop=True)
            all_cells = kg.cell.to_numpy(dtype=np.int64)
            global_pred = kg.y_pred.to_numpy(float)
            regional_pred = src.y_pred.to_numpy(float)
            y = kg.y_true.to_numpy(float)
            graph_delta = kg.graph_delta_raw.to_numpy(float)
            if not np.array_equal(all_cells, src.cell.to_numpy(dtype=np.int64)):
                raise ValueError("KGML and regional outer cells are not aligned")
        else:
            run = KGML_ROOT / (
                f"residual_context_msgdelta__doc__e3_spatial_validation_supportmatched__seed{seed}"
            )
            kg = load_kgml(run, "val")
            all_cells = kg.cell.to_numpy(dtype=np.int64)
            global_pred = kg.y_pred.to_numpy(float)
            y = kg.y_true.to_numpy(float)
            graph_delta = kg.graph_delta_raw.to_numpy(float)
            regional_pred = source_predictions(data, split, all_cells, desc, seed, n_estimators)
        if not np.isfinite(global_pred).all() or not np.isfinite(regional_pred).all():
            raise ValueError("non-finite base prediction")
        by_cell = {int(c): i for i, c in enumerate(all_cells)}
        for k in KS:
            support, query = support_schedule(all_cells, k)
            qidx = np.asarray([by_cell[int(c)] for c in query], dtype=int)
            sdelta = support_correction(y, regional_pred, all_cells, support)
            # Gate signal is the magnitude of the support correction.  It is
            # label-free at query time and uses only the K opened labels.
            delta_q = np.asarray([sdelta.get(int(c) // T, 0.0) for c in query])
            gq = global_pred[qidx]
            rq = regional_pred[qidx]
            g_res = apply_residual(gq, query, sdelta)
            r_res = apply_residual(rq, query, sdelta)
            for name, pred in (
                ("global", gq),
                ("regional", rq),
                ("global_residual", g_res),
                ("regional_residual", r_res),
            ):
                for cell, truth, value in zip(query, y[qidx], pred, strict=True):
                    out_rows.append({
                        "role": role, "seed": seed, "k": k,
                        "candidate": name, "weight": np.nan,
                        "threshold": np.nan, "cell": int(cell),
                        "station": int(cell) // T, "y_true": float(truth),
                        "y_pred": float(value), "graph_delta": float(graph_delta[by_cell[int(cell)]]),
                        "support_correction": float(sdelta.get(int(cell) // T, 0.0)),
                        "support_correction_abs": abs(float(sdelta.get(int(cell) // T, 0.0))),
                    })
            for weight in BLEND_WEIGHTS:
                pred = (1.0 - weight) * g_res + weight * r_res
                for cell, truth, value in zip(query, y[qidx], pred, strict=True):
                    out_rows.append({
                        "role": role, "seed": seed, "k": k,
                        "candidate": "fixed_blend", "weight": weight,
                        "threshold": np.nan, "cell": int(cell),
                        "station": int(cell) // T, "y_true": float(truth),
                        "y_pred": float(value), "graph_delta": float(graph_delta[by_cell[int(cell)]]),
                        "support_correction": float(sdelta.get(int(cell) // T, 0.0)),
                        "support_correction_abs": abs(float(sdelta.get(int(cell) // T, 0.0))),
                    })
            for threshold in GATE_THRESHOLDS:
                use_regional = np.abs(delta_q) >= threshold
                pred = np.where(use_regional, r_res, g_res)
                for cell, truth, value in zip(query, y[qidx], pred, strict=True):
                    out_rows.append({
                        "role": role, "seed": seed, "k": k,
                        "candidate": "support_gate", "weight": np.nan,
                        "threshold": threshold, "cell": int(cell),
                        "station": int(cell) // T, "y_true": float(truth),
                        "y_pred": float(value), "graph_delta": float(graph_delta[by_cell[int(cell)]]),
                        "support_correction": float(sdelta.get(int(cell) // T, 0.0)),
                        "support_correction_abs": abs(float(sdelta.get(int(cell) // T, 0.0))),
                    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-estimators", type=int, default=120)
    args = p.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    # Use the exact support-matched validation mask used to produce the KGML
    # validation predictions.  Replacing it with the generic E3 internal
    # split would let the source expert see labels from the KGML validation
    # stations and would make the gate comparison invalid.
    internal = load_split(INTERNAL_MASK)
    desc = station_descriptors(data)
    rows: list[dict] = []
    build_rows(data, internal, "internal_val", rows, desc, args.n_estimators)
    build_rows(data, outer, "outer_test", rows, desc, args.n_estimators)
    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out_dir / "predictions.parquet", index=False)
    # Nested selection uses pooled internal cell MAE, then the selected
    # candidate is evaluated once on outer E3. K=0 has no support gate.
    val = frame[frame.role == "internal_val"]
    scores = val.groupby(["k", "candidate", "weight", "threshold"], dropna=False).apply(
        lambda x: float(np.mean(np.abs(x.y_true - x.y_pred))), include_groups=False,
    ).rename("mae").reset_index()
    selected = []
    for k in KS:
        take = scores[scores.k == k].sort_values("mae", kind="stable").iloc[0]
        selected.append(take.to_dict())
    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(args.out_dir / "selected.csv", index=False)
    outer = frame[frame.role == "outer_test"]
    selected_rows = []
    for row in selected_df.itertuples(index=False):
        m = (outer.k == row.k) & (outer.candidate == row.candidate)
        if pd.isna(row.weight):
            m &= outer.weight.isna()
        else:
            m &= outer.weight == row.weight
        if pd.isna(row.threshold):
            m &= outer.threshold.isna()
        else:
            m &= outer.threshold == row.threshold
        selected_rows.append({
            "k": row.k, "candidate": row.candidate, "weight": row.weight,
            "threshold": row.threshold, **metrics(outer.loc[m].y_true.to_numpy(), outer.loc[m].y_pred.to_numpy()),
        })
    pd.DataFrame(selected_rows).to_csv(args.out_dir / "outer_selected.csv", index=False)
    # Station-level paired bootstrap: station means first, then resample
    # stations.  This avoids mixing cell and station weighting.
    station = (outer.groupby(["seed", "k", "candidate", "weight", "threshold", "station"], dropna=False)
               .apply(lambda x: float(np.mean(np.abs(x.y_true - x.y_pred))), include_groups=False)
               .rename("mae").reset_index())
    station.to_csv(args.out_dir / "outer_station_mae.csv", index=False)
    boot_rows = []
    rng = np.random.default_rng(731)
    for row in selected_df.itertuples(index=False):
        m = (station.k == row.k) & (station.candidate == row.candidate)
        if pd.isna(row.weight): m &= station.weight.isna()
        else: m &= station.weight == row.weight
        if pd.isna(row.threshold): m &= station.threshold.isna()
        else: m &= station.threshold == row.threshold
        current = station.loc[m].groupby("station").mae.mean()
        zero = station[(station.k == 0) & (station.candidate == "global")].groupby("station").mae.mean()
        joined = pd.concat([current.rename("current"), zero.rename("zero")], axis=1).dropna()
        delta = joined.current.to_numpy() - joined.zero.to_numpy()
        draws = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(3000)])
        boot_rows.append({
            "k": row.k, "candidate": row.candidate, "weight": row.weight,
            "threshold": row.threshold, "delta_mae_vs_global_k0": float(delta.mean()),
            "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
            "n_stations": len(delta),
        })
    pd.DataFrame(boot_rows).to_csv(args.out_dir / "outer_station_bootstrap.csv", index=False)
    spec = {
        "seeds": list(SEEDS), "K": list(KS), "source_pool": 40,
        "kgml_arm": "residual_context_msgdelta",
        "selection": "internal station-heldout only; outer E3 scored once",
        "gate_signal": "absolute station mean log1p support residual",
        "source_predictions": str(SOURCE_PRODUCT),
        "kgml_predictions": str(KGML_ROOT),
        "outer_query": "paired E3 test after reserving five support months per target station",
    }
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(selected_df.to_string(index=False))
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
