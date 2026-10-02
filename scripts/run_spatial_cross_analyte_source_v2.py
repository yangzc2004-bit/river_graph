"""Strict visibility audit and rerun for cross-analyte source selection.

The original exploratory script used full-station pH/EC profiles, including
target-station labels in months hidden by the DOC E3 split.  This version keeps
the old directory untouched and evaluates a strict spatial-zero-shot variant:
all pH/EC labels on the held-out target stations are masked before profiles
are computed, and descriptor scaling is fitted on source stations only.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyte_profile(path: str, *, hidden_rows: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return station profile with labels on hidden target rows excluded."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    y = data["y"].numpy().astype(float)
    mask = data["y_mask"].numpy().astype(bool)
    hidden = np.asarray(hidden_rows if hidden_rows is not None else [], dtype=int)
    if hidden.size:
        if hidden.min() < 0 or hidden.max() >= y.shape[0]:
            raise ValueError("hidden station row outside analyte dataset")
        mask[hidden, :] = False
    blocks: list[np.ndarray] = []
    quantile_counts = []
    for i in range(y.shape[0]):
        vals = y[i, mask[i]]
        quantile_counts.append(int(vals.size))
    for _value in (y,):
        den = np.maximum(mask.sum(1), 1.0)
        mean = (y * mask).sum(1) / den
        sd = np.sqrt(((y - mean[:, None]) ** 2 * mask).sum(1) / den)
        blocks.extend([mean, sd, mask.mean(1)])
        for q in (0.1, 0.5, 0.9):
            blocks.append(np.asarray([
                np.quantile(y[i, mask[i]], q) if mask[i].any() else 0.0
                for i in range(y.shape[0])
            ]))
    return np.stack(blocks, axis=1), {
        "path": str(path),
        "hidden_rows": hidden.tolist(),
        "n_visible_profile_cells": int(mask.sum()),
        "visible_cells_by_station": [int(v) for v in mask.sum(1)],
        "quantile_counts_by_station": quantile_counts,
    }


def strict_descriptors(doc: dict, split: dict[str, np.ndarray], role: str) -> tuple[np.ndarray, dict]:
    """Build profiles with all labels on target stations hidden.

    Hydro/ecology/graph descriptors remain label-free.  Their scaler and the
    pH/EC profile scaler are fitted on source stations only; held-out target
    covariates are transformed but never used to fit the scale.
    """
    target_cells = np.asarray(split[role], dtype=int)
    target_rows = np.unique(target_cells // T)
    ph, ph_audit = analyte_profile(DATASETS["ph"], hidden_rows=target_rows)
    ec, ec_audit = analyte_profile(DATASETS["spec_conductance"], hidden_rows=target_rows)
    base = station_descriptors(doc)
    raw = np.concatenate([base, ph, ec], axis=1)
    source_rows = np.setdiff1d(np.arange(raw.shape[0]), target_rows)
    scaler = StandardScaler().fit(raw[source_rows])
    desc = scaler.transform(raw).astype(np.float32)
    audit = {
        "role": role,
        "target_rows": target_rows.tolist(),
        "source_rows": source_rows.tolist(),
        "profile_scaler_fit_rows": source_rows.tolist(),
        "ph": ph_audit,
        "spec_conductance": ec_audit,
        "descriptor_scaler": "source-station rows only",
    }
    return desc, audit


def audit_original_leakage(doc: dict, split: dict[str, np.ndarray], role: str) -> dict:
    """Quantify labels that the original full-profile function consumed."""
    target_cells = np.asarray(split[role], dtype=int)
    rows = np.unique(target_cells // T)
    result: dict = {"role": role, "target_rows": rows.tolist(), "target_cells": len(target_cells)}
    for name in ("ph", "spec_conductance"):
        data = torch.load(DATASETS[name], map_location="cpu", weights_only=False)
        observed = data["y_mask"].numpy().astype(bool)
        target_profile = observed[rows]
        overlap = observed.ravel()[target_cells]
        result[name] = {
            "target_profile_cells_consumed_by_original": int(target_profile.sum()),
            "target_role_cells_with_profile_label": int(overlap.sum()),
            "target_profile_cells_in_hidden_role_fraction": float(overlap.sum() / max(target_profile.sum(), 1)),
        }
    return result


def run(args: argparse.Namespace) -> None:
    doc = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _rows, _ = e3_internal_split(doc["y_mask"].numpy(), outer["test"])
    audits = {
        "internal_val": audit_original_leakage(doc, internal, "val"),
        "outer_test": audit_original_leakage(doc, outer, "test"),
    }
    descriptors_by_role = {
        "internal_val": strict_descriptors(doc, internal, "val"),
        "outer_test": strict_descriptors(doc, outer, "test"),
    }
    rows: list[dict] = []
    for seed in args.seeds:
        for split, role, desc_key in (
            (internal, "val", "internal_val"),
            (outer, "test", "outer_test"),
        ):
            desc = descriptors_by_role[desc_key][0]
            cells = np.asarray(split[role], dtype=np.int64)
            fit_x = build_rf_features(doc, split, FIT_ROLES, target_transform="log1p", include_network=True)
            eval_x = build_rf_features(doc, split, TEST_ROLES, target_transform="log1p", include_network=True)
            for k in KS:
                pred = predict_query_models(
                    doc,
                    split,
                    np.asarray(split["train"], dtype=np.int64),
                    cells,
                    fit_x,
                    eval_x,
                    desc,
                    k,
                    seed,
                    args.n_estimators,
                    4,
                    4,
                )
                y = doc["y"].numpy().reshape(-1)
                rows.append({"seed": seed, "role": role, "k": k, **metrics(y[cells], pred)})
    out = pd.DataFrame(rows)
    selected = int(out[out.role == "val"].groupby("k").mae.mean().sort_values().index[0])
    test = out[(out.role == "test") & (out.k == selected)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "leakage_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    profile_audits = {k: v[1] for k, v in descriptors_by_role.items()}
    (args.out_dir / "strict_profile_audit.json").write_text(json.dumps(profile_audits, indent=2) + "\n")
    spec = {
        "variant": "strict_target_station_profile_visibility_v2",
        "source_script": "scripts/run_spatial_cross_analyte_source_v2.py",
        "mask": str(MASK),
        "target_profile_visibility": "all pH/EC labels on DOC held-out target rows hidden",
        "descriptor_scaler": "fit on source station rows only",
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "ks": list(KS),
        "selected_k_on_internal_validation": selected,
        "datasets": {k: str(v) for k, v in DATASETS.items()},
        "dataset_hashes": {k: _sha256(Path(v)) for k, v in DATASETS.items()},
    }
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    summary = [
        "# Strict cross-analyte source-selection rerun",
        "",
        "The v1 script consumed full pH/EC station histories. This v2 hides all",
        "pH/EC labels on DOC held-out target stations and fits descriptor scaling",
        "on source stations only.",
        "",
        f"selected_k={selected}",
        "",
        out.to_string(index=False),
        "",
        "Outer test selected rows:",
        test.to_string(index=False),
    ]
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps(audits, indent=2))
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
