"""3B-R1: val-only recalibration of the stored Phase-3 seed ensembles.

Frozen scope (docs/paper/phase3_uncertainty_spec_v2.md): NO retraining.
Central predictions, seed spread and all three covariates stay bit-identical
to v1; only qhat, pi_lower/pi_upper and uncertainty may change. Calibration
scores use ``val`` residuals ONLY (train and context excluded). v2 products
live in full_grid_v2_valonly/ and bind the v1 product hash, seed-ensemble
hash, v2 spec hash, calibration-cell definition hash and runtime hash.
Empirical validation calibration — never conformal.

Usage:
    python scripts/run3b_r1_valonly.py              # write v2 + report
    python scripts/run3b_r1_valonly.py --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.phase3_uncertainty import (
    SEEDS,
    TOOLS,
    calibrate_interval,
    coverage_with_bootstrap,
    family_of,
    sha256_file,
)

DATASET = "data/processed/mississippi_graph_graphfix_st357.pt"
MASKS_DIR = Path("experiments/masks_stcore_v1")
OUT = Path("experiments/phase3_uncertainty_stcore_v1")
OUT_V2 = OUT / "full_grid_v2_valonly"
SPEC_V2 = Path("docs/paper/phase3_uncertainty_spec_v2.md")
CAL_DEF = {
    "rule": "val_only",
    "keys": ["val"],
    "context_excluded_from_calibration": True,
    "train_excluded_from_calibration": True,
}
M = ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44", "e2a_strict",
     "e2b_partial", "e3_spatial_seed42", "e3_spatial_seed43",
     "e3_spatial_seed44"]


def cal_def_hash() -> str:
    blob = json.dumps(CAL_DEF, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def product_v1(tool: str, mask: str) -> Path:
    return OUT / "full_grid" / f"P3_{tool}__{mask}.parquet"


def product_v2(tool: str, mask: str) -> Path:
    return OUT_V2 / f"P3_{tool}__{mask}__valonly.parquet"


def recompute_center_spread(tool: str, mask: str, n: int, t: int):
    sp = pd.read_parquet(
        OUT / "seed_preds" / f"P3_{tool}_s42-46__{mask}.parquet")
    stack = np.column_stack([sp[f"pred_s{s}"].to_numpy()
                             for s in SEEDS]).T
    seed_log = np.log1p(stack.reshape(len(SEEDS), n, t))
    median_log = np.median(seed_log, axis=0)
    spread = seed_log.std(axis=0)
    s_floor = float(max(np.quantile(spread, 0.10), 1e-3))
    s_eff = np.maximum(spread, s_floor)
    return seed_log, median_log, s_eff, s_floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    ds = torch.load(DATASET, weights_only=False)
    n, t = ds["y"].shape
    y_mg = ds["y"].numpy()
    y_log = np.log1p(y_mg)
    sites = [str(s) for s in ds["site_no"]]
    runtime_sha = __import__(
        "river_graph.experiments.provenance",
        fromlist=["runtime_code_snapshot_sha256"],
    ).runtime_code_snapshot_sha256()

    report_rows, _gate_rows = [], []
    for mask in M:
        split = dict(np.load(MASKS_DIR / f"{mask}.npz"))
        fam = family_of(mask)
        val = np.asarray(split.get("val", []), dtype=np.int64)
        tr = np.asarray(split["train"], dtype=np.int64)
        te = np.asarray(split["test"], dtype=np.int64)
        for tool in TOOLS:
            seed_file = OUT / "seed_preds" / f"P3_{tool}_s42-46__{mask}.parquet"
            v1p = product_v1(tool, mask)
            seed_log, median_log, s_eff, _s_floor = recompute_center_spread(
                tool, mask, n, t)
            v1 = pd.read_parquet(v1p)
            # ---- invariants: center/spread/covariates bit-identical to v1
            med1 = v1["prediction_median"].to_numpy().reshape(n, t)
            assert np.allclose(np.expm1(median_log), med1, rtol=0, atol=1e-9), (
                f"central prediction drifted for {tool} {mask}")
            qhat1 = json.loads(v1["provenance_hashes"].iloc[0])["config"]["qhat"]
            unc1 = v1["uncertainty"].to_numpy().reshape(n, t)
            assert np.allclose(qhat1 * s_eff, unc1, rtol=0, atol=1e-9), (
                f"seed spread drifted for {tool} {mask}")
            p = product_v2(tool, mask)
            if args.report_only and not p.is_file():
                raise SystemExit(f"missing v2 product {p}")
            if not args.report_only:
                # central/spread recomputed above must already match v1
                # (asserted); only qhat/pi/uncertainty change here
                out = calibrate_interval(seed_log, y_log, val)
                assert out["n_cal"] == len(val)
                v2 = v1.copy()
                v2["prediction_median"] = out["prediction_median"].reshape(-1)
                v2["pi_lower"] = out["pi_lower"].reshape(-1)
                v2["pi_upper"] = out["pi_upper"].reshape(-1)
                v2["uncertainty"] = out["uncertainty"].reshape(-1)
                prov = {
                    "v1_product_sha256": sha256_file(v1p),
                    "seed_file_sha256": sha256_file(seed_file),
                    "phase3_spec_v2_sha256": sha256_file(SPEC_V2),
                    "calibration_cells_def_sha256": cal_def_hash(),
                    "runtime_code_snapshot_sha256": runtime_sha,
                    "config": {"tool": tool, "mask": mask, "family": fam,
                               "calibration": "val_only empirical validation "
                                              "calibration (not conformal)",
                               "qhat_v1": qhat1, "qhat_v2": out["qhat"],
                               "s_floor": out["s_floor"], "n_cal": out["n_cal"]},
                }
                v2["provenance_hashes"] = json.dumps(prov, sort_keys=True)
                OUT_V2.mkdir(parents=True, exist_ok=True)
                v2.to_parquet(p, index=False)
                p.with_suffix(".json").write_text(json.dumps({
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "rows": len(v2), "provenance_hashes": prov,
                    "roles": {"train": "fit only (excluded from v2 calibration)",
                              "context": "E2b inference input only "
                                         "(excluded from v2 calibration)",
                              "val": "early stopping + v2 calibration set",
                              "test": "hidden; never read"},
                }, indent=2), encoding="utf-8")
                out["uncertainty"]
            else:
                v2 = pd.read_parquet(p)
                v2["uncertainty"].to_numpy().reshape(n, t)
                out = {"qhat": json.loads(
                    v2["provenance_hashes"].iloc[0])["config"]["qhat_v2"],
                    "n_cal": len(val)}
            lo2 = v2["pi_lower"].to_numpy().reshape(n, t).reshape(-1)
            hi2 = v2["pi_upper"].to_numpy().reshape(n, t).reshape(-1)
            yte = y_mg.reshape(-1)[te]
            hit2 = ((yte >= lo2[te]) & (yte <= hi2[te])).astype(float)
            res = coverage_with_bootstrap(
                y_mg, lo2.reshape(n, t), hi2.reshape(n, t), te, sites, t)
            widths = (hi2[te] - lo2[te])
            lo1 = v1["pi_lower"].to_numpy().reshape(n, t).reshape(-1)
            hi1 = v1["pi_upper"].to_numpy().reshape(n, t).reshape(-1)
            hit1 = ((yte >= lo1[te]) & (yte <= hi1[te])).astype(float)
            width1 = (hi1[te] - lo1[te])
            for p_q, tag in ((0.90, "Q90"), (0.95, "Q95")):
                thr = float(np.quantile(y_mg.reshape(-1)[tr], p_q))
                s = yte >= thr
                row = {"tool": tool, "mask": mask, "family": fam,
                       "subset": tag, "n": int(s.sum()),
                       "coverage_v2": float(hit2[s].mean()) if s.sum() else np.nan,
                       "width_med_v2": float(np.median(widths[s])) if s.sum() else np.nan,
                       "small_sample_flag": bool(s.sum() < 20)}
                report_rows.append(row)
            score = (np.abs(y_log - np.median(seed_log, axis=0)) / s_eff).reshape(-1)
            report_rows.append({
                "tool": tool, "mask": mask, "family": fam, "subset": "test_main",
                "n": len(te),
                "coverage_v2": res["coverage"], "ci95_lo": res["ci95_lo"],
                "ci95_hi": res["ci95_hi"],
                "coverage_v1": float(hit1.mean()),
                "width_med_v2": float(np.median(widths)),
                "width_iqr_v2": float(np.subtract(*np.percentile(widths, [75, 25]))),
                "width_p90_v2": float(np.quantile(widths, 0.9)),
                "width_med_v1": float(np.median(width1)),
                "width_iqr_v1": float(np.subtract(*np.percentile(width1, [75, 25]))),
                "width_p90_v1": float(np.quantile(width1, 0.9)),
                "qhat_v1": qhat1, "qhat_v2": out["qhat"], "n_cal": out["n_cal"],
                "score_q90_train": float(np.quantile(score[tr], 0.9)),
                "score_q90_val": float(np.quantile(score[val], 0.9)),
                "score_q90_test": float(np.quantile(score[te], 0.9)),
                "score_q95_train": float(np.quantile(score[tr], 0.95)),
                "score_q95_val": float(np.quantile(score[val], 0.95)),
                "score_q95_test": float(np.quantile(score[te], 0.95)),
            })

    rep = pd.DataFrame(report_rows)
    outdir = OUT / "eval"
    rep.to_csv(outdir / "r1_valonly_report.csv", index=False)
    main_rows = rep[rep["subset"] == "test_main"]
    gates = []
    for tool in TOOLS:
        sub = main_rows[main_rows.tool == tool]
        c1, c2 = float(sub.coverage_v1.mean()), float(sub.coverage_v2.mean())
        w1, w2 = float(sub.width_med_v1.mean()), float(sub.width_med_v2.mean())
        gates.append({
            "tool": tool, "coverage_v1": round(c1, 3), "coverage_v2": round(c2, 3),
            "gate_080_095_v2": bool(0.80 <= c2 <= 0.95),
            "width_med_v1": round(w1, 2), "width_med_v2": round(w2, 2),
            "width_ratio_v2_v1": round(w2 / w1, 2) if w1 else np.nan,
        })
    gdf = pd.DataFrame(gates)
    n_pass = int(gdf["gate_080_095_v2"].sum())
    lines = [
        "# 3B-R1 val-only report (empirical validation calibration — not conformal)",
        "",
        ("Method revision made after v1 results were seen "
        "(docs/paper/phase3_uncertainty_spec_v2.md). v1 numbers unchanged and "
        "kept side by side."),
        "",
        "## Coverage and width trade-off (per tool, equal-mask means)",
        "",
        gdf.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Tools reaching the coverage gate: **{n_pass}** → "
        + ("calibration-level improvement only; Phase 4/5 stay locked"
           if n_pass >= 2 else "the pause continues"),
        "",
        "## Per tool x family",
        "",
        main_rows.groupby(["tool", "family"])[
            ["coverage_v1", "coverage_v2", "width_med_v1", "width_med_v2"]
        ].mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Standardized residual distributions (Q90 / Q95)",
        "",
        main_rows.groupby("tool")[
            ["score_q90_train", "score_q90_val", "score_q90_test",
             "score_q95_train", "score_q95_val", "score_q95_test"]
        ].mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Tails (3C rules; Q95 n<20 unstable)",
        "",
        rep[rep["subset"].isin(["Q90", "Q95"])].groupby(
            ["tool", "subset"])[["coverage_v2", "width_med_v2", "n"]]
        .mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        ("Monotonicity is NOT recomputed here (spec v2 §4): val-only changes "
        "only the uniform scale and cannot change rankings or directions."),
    ]
    (outdir / "r1_valonly_report.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    print(gdf.to_string(index=False))


if __name__ == "__main__":
    main()
