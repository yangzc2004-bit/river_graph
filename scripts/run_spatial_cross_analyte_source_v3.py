"""Visibility-policy comparison for cross-analyte source selection.

This version treats auxiliary pH/EC observations as an explicit policy.  The
main rerun is ``k_session``: for a DOC target station, only pH/EC observations
at the K selected DOC support months are visible.  Missing profile dimensions
are excluded from pairwise source distance, never replaced by zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import load_split

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
K_VALUES = (0, 1, 3, 5)
SOURCE_K_VALUES = (40, 80, 160)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_station_descriptors(data: dict) -> np.ndarray:
    """Unscaled label-free descriptors used by the existing regional expert."""
    x = data["x"].numpy().astype(np.float64)
    xm = data["x_mask"].numpy().astype(np.float64)
    n, _t, c = x.shape
    blocks: list[np.ndarray] = []
    for j in range(c):
        a, v = x[:, :, j], xm[:, :, j]
        den = np.maximum(v.sum(1), 1.0)
        mean = (a * v).sum(1) / den
        sd = np.sqrt(((a - mean[:, None]) ** 2 * v).sum(1) / den)
        qs = [np.asarray([
            np.quantile(a[i, v[i] > 0], q) if np.any(v[i] > 0) else 0.0
            for i in range(n)
        ]) for q in (0.10, 0.50, 0.90)]
        blocks.extend([mean, sd, *qs, v.mean(1)])
    blocks.extend([data["static"].numpy()[:, j] for j in range(data["static"].shape[1])])
    blocks.extend([data["regime"].numpy()[:, j] for j in range(data["regime"].shape[1])])
    edge = data["edge_index"].numpy()
    attr = data["edge_attr"].numpy().astype(np.float64)
    blocks.extend([np.bincount(edge[0], minlength=n), np.bincount(edge[1], minlength=n)])
    for j in range(attr.shape[1]):
        blocks.append(np.bincount(edge[1], weights=attr[:, j], minlength=n))
    return np.nan_to_num(np.stack(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def visible_aux_mask(
    aux_mask: np.ndarray,
    target_rows: np.ndarray,
    target_cells: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return pH/EC visibility for the explicit K-session policy."""
    visible = np.asarray(aux_mask, dtype=bool).copy()
    target_rows = np.asarray(target_rows, dtype=int)
    visible[target_rows, :] = False
    schedule, _query = support_schedule(np.asarray(target_cells, dtype=int), T)
    for station, cells in schedule.items():
        months = cells[:k] % T
        if len(months):
            visible[int(station), months] = aux_mask[int(station), months]
    return visible


def profile_statistics(y: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute profile values and availability masks without missing-as-zero."""
    n = y.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    available = np.zeros((n, 6), dtype=bool)
    for i in range(n):
        vals = y[i, visible[i]]
        if not len(vals):
            continue
        den = float(len(vals))
        mean = float(vals.mean())
        out[i] = [mean, float(vals.std()), den / y.shape[1], *np.quantile(vals, [0.1, 0.5, 0.9])]
        available[i, :] = True
    return out, available


def masked_standardize(raw: np.ndarray, available: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    """Standardize on sources only; preserve values for masked-distance use."""
    z = np.zeros_like(raw, dtype=np.float64)
    for j in range(raw.shape[1]):
        take = available[source_rows, j]
        if not take.any():
            z[:, j] = 0.0
            continue
        vals = raw[source_rows[take], j]
        mu, sd = float(vals.mean()), float(vals.std())
        if sd < 1e-8:
            sd = 1.0
        z[:, j] = (raw[:, j] - mu) / sd
    return z.astype(np.float32)


def build_policy_descriptors(
    doc: dict,
    split: dict[str, np.ndarray],
    role: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build target-aware profile descriptors and pairwise availability masks."""
    target_cells = np.asarray(split[role], dtype=int)
    target_rows = np.unique(target_cells // T)
    source_rows = np.unique(np.asarray(split["train"], dtype=int) // T)
    base = base_station_descriptors(doc)
    ph_data = torch.load(DATASETS["ph"], map_location="cpu", weights_only=False)
    ec_data = torch.load(DATASETS["spec_conductance"], map_location="cpu", weights_only=False)
    ph_visible = visible_aux_mask(ph_data["y_mask"].numpy(), target_rows, target_cells, k)
    ec_visible = visible_aux_mask(ec_data["y_mask"].numpy(), target_rows, target_cells, k)
    ph, ph_avail = profile_statistics(ph_data["y"].numpy(), ph_visible)
    ec, ec_avail = profile_statistics(ec_data["y"].numpy(), ec_visible)
    raw = np.concatenate([base, ph, ec], axis=1)
    available = np.concatenate([
        np.ones_like(base, dtype=bool), ph_avail, ec_avail,
    ], axis=1)
    z = masked_standardize(raw, available, source_rows)
    audit = {
        "role": role,
        "k_session": k,
        "target_rows": target_rows.tolist(),
        "source_rows": source_rows.tolist(),
        "ph_visible_cells": int(ph_visible[target_rows].sum()),
        "ec_visible_cells": int(ec_visible[target_rows].sum()),
        "ph_target_profile_dimensions": int(ph_avail[target_rows].sum()),
        "ec_target_profile_dimensions": int(ec_avail[target_rows].sum()),
        "distance_policy": "pairwise available dimensions only; no missing-as-zero",
    }
    return z, available, audit


def masked_neighbours(
    descriptors: np.ndarray,
    available: np.ndarray,
    source: np.ndarray,
    target: int,
    k: int,
) -> np.ndarray:
    both = available[source] & available[int(target)]
    count = both.sum(axis=1)
    if (count == 0).any():
        raise ValueError("target/source descriptor pair has no common dimension")
    diff = np.square(descriptors[source] - descriptors[int(target)])
    distance = (diff * both).sum(axis=1) / count
    return source[np.argsort(distance)[: min(int(k), len(source))]]


def predict_masked(
    data: dict,
    split: dict,
    fit_cells: np.ndarray,
    query_cells: np.ndarray,
    fit_x: np.ndarray,
    query_x: np.ndarray,
    descriptors: np.ndarray,
    available: np.ndarray,
    source_k: int,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor

    source = np.unique(fit_cells // T)
    targets = np.unique(query_cells // T)
    by_station = {int(s): fit_cells[fit_cells // T == s] for s in source}
    yz = np.log1p(data["y"].numpy().reshape(-1))
    pred = np.empty(len(query_cells), dtype=np.float64)
    positions = {int(c): i for i, c in enumerate(query_cells)}
    for station in targets:
        local = np.flatnonzero(query_cells // T == station)
        neighbors = masked_neighbours(descriptors, available, source, int(station), source_k)
        train = np.concatenate([by_station[int(s)] for s in neighbors])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            max_features=1.0,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(fit_x[train], yz[train])
        p = np.expm1(model.predict(query_x[query_cells[local]]))
        for value, cell in zip(p, query_cells[local], strict=True):
            pred[positions[int(cell)]] = value
    return pred


def audit_policies(doc: dict, split: dict, role: str) -> dict:
    rows = np.unique(np.asarray(split[role], dtype=int) // T)
    result = {"role": role, "target_rows": rows.tolist()}
    for name in ("ph", "spec_conductance"):
        data = torch.load(DATASETS[name], map_location="cpu", weights_only=False)
        mask = data["y_mask"].numpy().astype(bool)
        result[name] = {
            "full_aux_target_cells": int(mask[rows].sum()),
            "k_session_target_cells": {
                str(k): int(visible_aux_mask(mask, rows, split[role], k)[rows].sum())
                for k in K_VALUES
            },
        }
    return result


def run(args: argparse.Namespace) -> None:
    doc = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _rows, _ = e3_internal_split(doc["y_mask"].numpy(), outer["test"])
    policy_audit = {
        "internal_val": audit_policies(doc, internal, "val"),
        "outer_test": audit_policies(doc, outer, "test"),
    }
    rows: list[dict] = []
    profile_audits: dict[str, dict] = {}
    for split, role in ((internal, "val"), (outer, "test")):
        fit_x = build_rf_features(doc, split, FIT_ROLES, target_transform="log1p", include_network=True)
        query_x = build_rf_features(doc, split, TEST_ROLES, target_transform="log1p", include_network=True)
        y = doc["y"].numpy().reshape(-1)
        for k in K_VALUES:
            desc, avail, audit = build_policy_descriptors(doc, split, role, k)
            profile_audits[f"{role}_k{k}"] = audit
            for seed in args.seeds:
                for source_k in SOURCE_K_VALUES:
                    cells = np.asarray(split[role], dtype=np.int64)
                    pred = predict_masked(
                        doc, split, np.asarray(split["train"], dtype=np.int64), cells,
                        fit_x, query_x, desc, avail, source_k, seed, args.n_estimators,
                    )
                    rows.append({
                        "seed": seed, "role": role, "profile_k": k,
                        "source_k": source_k, **metrics(y[cells], pred),
                    })
    out = pd.DataFrame(rows)
    val = out[out.role == "val"].groupby(["profile_k", "source_k"], as_index=False).mae.mean()
    chosen = val.sort_values(["mae", "profile_k", "source_k"]).iloc[0]
    selected_profile_k = int(chosen.profile_k)
    selected_source_k = int(chosen.source_k)
    selected_test = out[
        (out.role == "test")
        & (out.profile_k == selected_profile_k)
        & (out.source_k == selected_source_k)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "metrics.csv", index=False)
    val.to_csv(args.out_dir / "validation_grid.csv", index=False)
    selected_test.to_csv(args.out_dir / "selected_outer.csv", index=False)
    (args.out_dir / "policy_audit.json").write_text(json.dumps(policy_audit, indent=2) + "\n")
    (args.out_dir / "profile_audit.json").write_text(json.dumps(profile_audits, indent=2) + "\n")
    spec = {
        "variant": "k_session_mask_aware_cross_analyte_source_v3",
        "policies": {
            "all_analyte_unmonitored": "no target pH/EC auxiliary profile; base descriptor route",
            "doc_unmonitored_aux_observed": "full target pH/EC profile explicitly allowed",
            "k_session": "only target pH/EC labels at K DOC support months; missing dimensions excluded",
        },
        "active_policy": "k_session",
        "source_k_values": list(SOURCE_K_VALUES),
        "profile_k_values": list(K_VALUES),
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "selected_profile_k": selected_profile_k,
        "selected_source_k": selected_source_k,
        "mask": str(MASK),
        "script_sha256": _sha256(Path(__file__)),
        "dataset_hashes": {k: _sha256(Path(v)) for k, v in DATASETS.items()},
    }
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    summary = [
        "# K-session mask-aware cross-analyte source selection",
        "",
        "Only pH/EC observations at the K DOC support months are visible on",
        "target stations. Missing profile dimensions are excluded pairwise from",
        "source distance; they are never encoded as zero.",
        "",
        f"selected_profile_k={selected_profile_k}",
        f"selected_source_k={selected_source_k}",
        "",
        val.to_string(index=False),
        "",
        "Selected outer rows:",
        selected_test.to_string(index=False),
    ]
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(val.to_string(index=False))
    print("selected", selected_profile_k, selected_source_k)
    print(selected_test.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
