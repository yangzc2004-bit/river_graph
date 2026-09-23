"""Phase-3 full-grid uncertainty product (spec §8 steps 2-6).

Per mask and tool (H2X, H2X_nomsg, eco_RF) x 5 seeds: train, predict the full
grid under scenario visibility (§1), write the raw seed ensemble, then
collapse to the product columns (§2), calibrate the central 90% interval on
train/val visible labels only (§3, empirical validation calibration), attach
the §4 covariates and cross-tool model_agreement (§6), and evaluate hidden-
test coverage with station-clustered bootstrap plus the pre-frozen
monotonicity rule (§5).

Usage:
    python scripts/run3b_fullgrid.py                # generate + evaluate
    python scripts/run3b_fullgrid.py --eval-only    # re-score written products
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.baselines.baselines import EcoRandomForest
from river_graph.experiments.phase3_uncertainty import (
    OUT_ROOT,
    SCENARIO_VISIBILITY,
    SEEDS,
    TOOLS,
    calibrate_interval,
    calibration_cells,
    covariates,
    coverage_with_bootstrap,
    family_of,
    monotonicity_check,
    product_hashes,
    scenario_visibility,
    visibility_role,
    write_product,
)
from river_graph.models.gcn import GCNDocModel

DATASET = "data/processed/mississippi_graph_graphfix_st357.pt"
MASKS_DIR = Path("experiments/masks_stcore_v1")
MASKS = [
    "e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44",
    "e2a_strict", "e2b_partial",
    "e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44",
]


def predict_tool(tool: str, seed: int, dataset: dict, split: dict,
                 vis_flat: np.ndarray, fam: str) -> np.ndarray:
    if tool == "eco_RF":
        return EcoRandomForest(n_estimators=200, seed=seed).fit_predict_full(
            dataset, split, predict_visibility=set(SCENARIO_VISIBILITY[fam]))
    model = GCNDocModel(
        architecture="transport_enc", env_groups=None, env_encoder=True,
        edge_set="empty" if tool == "H2X_nomsg" else "river",
        seed=seed, max_epochs=200, patience=20,
    )
    model.fit(dataset, split)
    return model.predict(only_visible=vis_flat)


def seed_preds_path(tool: str, mask: str) -> Path:
    return OUT_ROOT / "seed_preds" / f"P3_{tool}_s42-46__{mask}.parquet"


def product_path(tool: str, mask: str) -> Path:
    return OUT_ROOT / "full_grid" / f"P3_{tool}__{mask}.parquet"


def generate_mask(mask: str, dataset: dict) -> None:
    split = dict(np.load(MASKS_DIR / f"{mask}.npz"))
    fam = family_of(mask)
    vis = scenario_visibility(split, fam)
    n, t = dataset["y"].shape
    sites = [str(s) for s in dataset["site_no"]]
    months = [str(m)[:7] for m in dataset["months"]]
    y_mg = dataset["y"].numpy()
    y_log = np.log1p(y_mg)
    grid_station = np.repeat(sites, t)
    grid_month = np.tile(months, n)
    role = visibility_role(split, n, t)

    seeds_by_tool: dict[str, np.ndarray] = {}
    for tool in TOOLS:
        sp = seed_preds_path(tool, mask)
        if sp.is_file():
            df = pd.read_parquet(sp)
            stacked = np.column_stack(
                [df[f"pred_s{s}"].to_numpy() for s in SEEDS]).T
            seeds_by_tool[tool] = stacked.reshape(len(SEEDS), n, t)
            print(f"[cached seed_preds] {tool} {mask}", flush=True)
            continue
        cols = {}
        stack = []
        for seed in SEEDS:
            pred = predict_tool(tool, seed, dataset, split, vis, fam)
            stack.append(pred)
            cols[f"pred_s{seed}"] = pred.reshape(-1)
        seeds_by_tool[tool] = np.stack(stack)
        sp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "station_id": grid_station, "month": grid_month, **cols,
        }).to_parquet(sp, index=False)
        print(f"[seed_preds] {tool} {mask}", flush=True)

    cov = covariates(dataset, vis)
    vis_stations = np.flatnonzero(
        np.bincount(vis, minlength=n * t).reshape(n, t).sum(axis=1) > 0
    )
    cal = calibration_cells(split, fam)
    calib = {}
    products = {}
    for tool in TOOLS:
        out = calibrate_interval(np.log1p(seeds_by_tool[tool]), y_log, cal)
        calib[tool] = out
        products[tool] = {
            "station_id": grid_station, "month": grid_month,
            "prediction_median": out["prediction_median"].reshape(-1),
            "pi_lower": out["pi_lower"].reshape(-1),
            "pi_upper": out["pi_upper"].reshape(-1),
            "uncertainty": out["uncertainty"].reshape(-1),
            "support_count": np.repeat(cov["support_count"], t),
            "network_distance": np.repeat(cov["network_distance"], t),
            "ecological_novelty": np.repeat(cov["ecological_novelty"], t),
            "visibility_role": role,
        }

    # §6 model agreement: each tool's high-uncertainty flag above its own
    # train-derived median (over cells of visible stations)
    highs = {}
    for tool in TOOLS:
        unc = products[tool]["uncertainty"].reshape(n, t)
        ref = unc[vis_stations].reshape(-1)
        thr = float(np.median(ref))
        highs[tool] = (unc > thr).astype(int)
    agreement = sum(highs.values())
    for tool in TOOLS:
        products[tool]["model_agreement"] = agreement.reshape(-1)

    for tool in TOOLS:
        cfg = {"tool": tool, "seeds": list(SEEDS), "mask": mask,
               "scenario_family": fam,
               "calibration": "empirical validation calibration",
               "qhat": calib[tool]["qhat"],
               "s_floor": calib[tool]["s_floor"],
               "n_cal": calib[tool]["n_cal"]}
        prov = product_hashes(
            DATASET, str(MASKS_DIR / f"{mask}.npz"),
            seed_files=[str(seed_preds_path(tool, mask))],
            config=cfg,
        )
        prov["config"] = cfg
        write_product(product_path(tool, mask), products[tool], prov)
        print(f"[product] {tool} {mask} qhat={calib[tool]['qhat']:.4f}",
              flush=True)
    merged = pd.DataFrame({
        "station_id": grid_station, "month": grid_month,
        "model_agreement": agreement.reshape(-1),
    })
    mpath = OUT_ROOT / "full_grid" / "merged" / f"model_agreement_{mask}.parquet"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(mpath, index=False)


def evaluate(masks: list[str], dataset: dict) -> None:
    n, t = dataset["y"].shape
    sites = [str(s) for s in dataset["site_no"]]
    y_mg = dataset["y"].numpy()
    cov_rows, mono_rows = [], []
    for mask in masks:
        split = dict(np.load(MASKS_DIR / f"{mask}.npz"))
        fam = family_of(mask)
        vis = scenario_visibility(split, fam)
        cov = covariates(dataset, vis)
        vis_stations = np.flatnonzero(
            np.bincount(vis, minlength=n * t).reshape(n, t).sum(axis=1) > 0)
        for tool in TOOLS:
            df = pd.read_parquet(product_path(tool, mask))
            unc = df["uncertainty"].to_numpy().reshape(n, t)
            lo = df["pi_lower"].to_numpy().reshape(n, t)
            hi = df["pi_upper"].to_numpy().reshape(n, t)
            med = df["prediction_median"].to_numpy().reshape(n, t)
            for label, cells in (("test", split["test"]),
                                 ("top5", None), ("top10", None)):
                if cells is None:
                    continue
                res = coverage_with_bootstrap(
                    y_mg, lo, hi, cells, sites, t)
                cov_rows.append({"tool": tool, "mask": mask, "family": fam,
                                 "subset": label, **res})
            tr = np.asarray(split["train"], dtype=np.int64)
            ytr = y_mg.reshape(-1)[tr]
            te = np.asarray(split["test"], dtype=np.int64)
            yte = y_mg.reshape(-1)[te]
            for p, tag in ((0.95, "top5"), (0.90, "top10")):
                thr = float(np.quantile(ytr, p))
                true_hi = yte >= thr
                lo_t, hi_t = lo.reshape(-1)[te], hi.reshape(-1)[te]
                tail_cov = (float(np.mean((yte[true_hi] >= lo_t[true_hi])
                                          & (yte[true_hi] <= hi_t[true_hi])))
                            if true_hi.sum() else np.nan)
                tail_mae = (float(np.mean(np.abs(yte[true_hi]
                                                - med.reshape(-1)[te][true_hi])))
                            if true_hi.sum() else np.nan)
                cov_rows.append({
                    "tool": tool, "mask": mask, "family": fam,
                    "subset": tag, "coverage": tail_cov,
                    "ci95_lo": np.nan, "ci95_hi": np.nan,
                    "n_test": int(true_hi.sum()),
                    "tail_mae": tail_mae,
                    "small_sample_flag": bool(true_hi.sum() < 20),
                })
            for cname, direction in (("network_distance", "up"),
                                     ("ecological_novelty", "up"),
                                     ("support_count", "down")):
                m = monotonicity_check(unc, cov[cname], direction, te,
                                      vis_stations, n, t)
                mono_rows.append({"tool": tool, "mask": mask, "family": fam,
                                  "covariate": cname, **m})
    cov_df = pd.DataFrame(cov_rows)
    mono_df = pd.DataFrame(mono_rows)
    out = OUT_ROOT / "eval"
    out.mkdir(parents=True, exist_ok=True)
    cov_df.to_csv(out / "coverage.csv", index=False)
    mono_df.to_csv(out / "monotonicity.csv", index=False)

    lines = [
        "# Phase-3B evaluation (empirical validation calibration — not conformal)",
        "",
        (f"Generated {datetime.now(timezone.utc).isoformat()} by "
        "`scripts/run3b_fullgrid.py` (one-command recompute with --eval-only)."),
        "",
        "## Coverage on hidden test cells (station-clustered bootstrap)",
        "",
        cov_df[cov_df["subset"] == "test"].to_markdown(index=False,
                                                       floatfmt=".3f"),
        "",
        "## High-DOC tail (3C rules: Q90 primary; n<20 = unstable)",
        "",
        cov_df[cov_df["subset"].isin(["top5", "top10"])].to_markdown(
            index=False, floatfmt=".3f"),
        "",
        "## Monotonicity (pre-frozen rule, train-derived tertiles)",
        "",
        mono_df.to_markdown(index=False, floatfmt=".3f"),
    ]
    (out / "phase3b_report.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print(cov_df[cov_df["subset"] == "test"].to_string(index=False))
    print()
    print(mono_df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", nargs="*", default=MASKS)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dataset = torch.load(DATASET, weights_only=False)
    if not args.eval_only:
        for mask in args.masks:
            print(f"=== {mask} ===", flush=True)
            generate_mask(mask, dataset)
    evaluate(args.masks, dataset)
    prov = product_hashes(DATASET, str(MASKS_DIR / f"{args.masks[0]}.npz"))
    (OUT_ROOT / "run_complete.json").write_text(json.dumps({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "masks": args.masks, "tools": list(TOOLS), "seeds": list(SEEDS),
        "spec_sha256": prov["phase3_spec_sha256"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
