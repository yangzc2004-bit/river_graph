"""3B-R2: does the v2 uncertainty ranking put wrong station-months first?

Frozen design: docs/paper/phase3_r2_ranking_spec.md. Selection uses ONLY the
v2 uncertainty column (never test DOC). Two reported layers: operational
threshold from train/val-era cells, and top-p ranking diagnostics on hidden
test cells. Primary metric: log1p absolute-error enrichment vs an equal-count
fixed-seed random baseline, with station-clustered bootstrap CIs. Q90 is the
primary tail indicator; Q95 is unstable at n<20. Test DOC is joined only at
final evaluation.

Usage: python scripts/run3b_r2_ranking.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path("experiments/phase3_uncertainty_stcore_v1")
V2 = OUT / "full_grid_v2_valonly"
DATASET = "data/processed/mississippi_graph_graphfix_st357.pt"
MASKS_DIR = Path("experiments/masks_stcore_v1")
TOOLS = ("H2X", "H2X_nomsg", "eco_RF")
M = ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44", "e2a_strict",
     "e2b_partial", "e3_spatial_seed42", "e3_spatial_seed43",
     "e3_spatial_seed44"]


def select_operational(unc: np.ndarray, role: np.ndarray,
                       p_top: float) -> tuple[float, np.ndarray]:
    """Deployment threshold from train/val-era cells ONLY (spec §2)."""
    era = np.isin(role, ["train", "val", "context"])
    thr = float(np.quantile(unc[era], 1.0 - p_top))
    return thr, unc >= thr


def select_rank_diag(unc_te: np.ndarray, p_top: float) -> np.ndarray:
    """Top-p ranking diagnostic on hidden test cells (uncertainty only)."""
    k = max(1, int(np.ceil(p_top * len(unc_te))))
    sel = np.zeros(len(unc_te), dtype=bool)
    sel[np.argsort(-unc_te)[:k]] = True
    return sel
P_MAIN = 0.10
P_SENS = (0.05, 0.20)
RANDOM_REPEATS = 200
RANDOM_SEED = 42
BOOT_DRAWS = 2000


def family_of(mask: str) -> str:
    return ("E1" if mask.startswith("e1") else "E2a" if mask.startswith("e2a")
            else "E2b" if mask.startswith("e2b") else "E3")


def cluster_ci(diff: np.ndarray, stations: np.ndarray,
               draws: int = BOOT_DRAWS) -> tuple[float, float]:
    uniq = np.unique(stations)
    idx = {s: np.flatnonzero(stations == s) for s in uniq}
    rng = np.random.default_rng(0)
    stats = np.empty(draws)
    for i in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in picked])
        stats[i] = float(diff[sel].mean())
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ds = torch.load(DATASET, weights_only=False)
    n, t = ds["y"].shape
    y = ds["y"].numpy()
    y_log = np.log1p(y)
    np.asarray([str(s) for s in ds["site_no"]])
    rows = []
    for mask in M:
        split = dict(np.load(MASKS_DIR / f"{mask}.npz"))
        fam = family_of(mask)
        te = np.asarray(split["test"], dtype=np.int64)
        tr = np.asarray(split["train"], dtype=np.int64)
        ytr = y.reshape(-1)[tr]
        for tool in TOOLS:
            df = pd.read_parquet(
                V2 / f"P3_{tool}__{mask}__valonly.parquet")
            unc = df["uncertainty"].to_numpy().reshape(n, t).reshape(-1)
            med = df["prediction_median"].to_numpy().reshape(n, t).reshape(-1)
            role = df["visibility_role"].to_numpy().reshape(n, t).reshape(-1)
            st_all = df["station_id"].to_numpy().reshape(n, t).reshape(-1)
            # ---- selection (never reads test DOC) ----
            thr90, sel_op_full = select_operational(unc, role, P_MAIN)
            thr_sens = {p: select_operational(unc, role, p)[0]
                        for p in P_SENS}
            unc_te = unc[te]
            sel_diag = select_rank_diag(unc_te, P_MAIN)
            sel_op = sel_op_full[te]
            # ---- evaluation joins test DOC ----
            e_log = np.abs(y_log.reshape(-1)[te] - np.log1p(med[te]))
            e_mg = np.abs(y.reshape(-1)[te] - med[te])
            st_te = st_all[te]
            for layer, sel in (("rank_diag_top10", sel_diag),
                               ("op_threshold_q90", sel_op)):
                if sel.sum() == 0:
                    continue
                rng = np.random.default_rng(RANDOM_SEED)
                base = [e_log[rng.choice(len(te), size=int(sel.sum()),
                                         replace=False)].mean()
                        for _ in range(RANDOM_REPEATS)]
                base_mg = [e_mg[rng.choice(len(te), size=int(sel.sum()),
                                           replace=False)].mean()
                           for _ in range(RANDOM_REPEATS)]
                m_sel = float(e_log[sel].mean())
                m_rand = float(np.mean(base))
                diff = e_log - m_rand  # cluster CI on the selected mean gap
                lo, hi = cluster_ci(np.where(sel, diff, 0.0) * len(te)
                                    / max(sel.sum(), 1), st_te)
                enr = m_sel / m_rand - 1.0 if m_rand else np.nan
                row = {
                    "tool": tool, "mask": mask, "family": fam, "layer": layer,
                    "n_selected": int(sel.sum()),
                    "logerr_selected": m_sel, "logerr_random": m_rand,
                    "logerr_random_mc_std": float(np.std(base)),
                    "enrichment_log1p": enr,
                    "enrichment_ci_lo": lo / m_rand if m_rand else np.nan,
                    "enrichment_ci_hi": hi / m_rand if m_rand else np.nan,
                    "mae_selected_mg": float(e_mg[sel].mean()),
                    "mae_random_mg": float(np.mean(base_mg)),
                    "enrichment_mg": (float(e_mg[sel].mean())
                                      / float(np.mean(base_mg)) - 1.0),
                    "thr90_from_era": thr90,
                }
                for p in P_SENS:
                    row[f"thr_sens_{int(p*100)}"] = thr_sens[p]
                for q, tag in ((0.90, "q90"), (0.95, "q95")):
                    thr = float(np.quantile(ytr, q))
                    hi_mask = y.reshape(-1)[te] >= thr
                    n_hi = int(hi_mask.sum())
                    n_sel = int(sel.sum())
                    recall = (float((sel & hi_mask).sum()) / n_hi
                              if n_hi else np.nan)
                    exp_recall = n_sel / len(te)
                    row[f"highdoc_{tag}_n"] = n_hi
                    row[f"highdoc_{tag}_recall"] = recall
                    row[f"highdoc_{tag}_recall_enr"] = (
                        recall / exp_recall - 1.0 if exp_recall else np.nan)
                    row[f"highdoc_{tag}_unstable"] = bool(n_hi < 20)
                    row[f"tail_{tag}_err_in"] = (
                        float(e_log[sel & hi_mask].mean())
                        if (sel & hi_mask).sum() else np.nan)
                rows.append(row)
            rows.append({
                "tool": tool, "mask": mask, "family": fam,
                "layer": "rank_association",
                "spearman_unc_abslogerr": spearman(unc_te, e_log),
            })
    rep = pd.DataFrame(rows)
    outdir = OUT / "eval"
    rep.to_csv(outdir / "r2_ranking.csv", index=False)

    # ---- judgment at the main operating point (frozen rule) ----
    # Family-level aggregation: a family qualifies when its equal-mask mean
    # enrichment is positive AND a majority of its masks have CI > 0
    # (single-mask families: that mask's CI must exclude 0).
    main_rows = rep[(rep.layer == "rank_diag_top10")]
    verdict_tools = {}
    for tool in TOOLS:
        sub = main_rows[main_rows.tool == tool]
        fam_qualified = 0
        fam_positive = 0
        fam_detail = {}
        for fam, g in sub.groupby("family"):
            enr = float(g["enrichment_log1p"].mean())
            n_ci = int((g["enrichment_ci_lo"] > 0).sum())
            need = 1 if len(g) == 1 else len(g) // 2 + 1
            qualifies = bool(enr > 0 and n_ci >= need)
            fam_positive += int(enr > 0)
            fam_qualified += int(qualifies)
            fam_detail[fam] = {"enrichment_mean": round(enr, 3),
                               "masks_ci_excl_0": f"{n_ci}/{len(g)}",
                               "qualifies": qualifies}
        stable = sub[sub["highdoc_q90_unstable"].eq(False)]
        tail_fams = (stable.groupby("family")["highdoc_q90_recall_enr"]
                     .mean().dropna())
        tail_dir = bool((tail_fams > 0).all()) if len(tail_fams) else None
        ranking_value = bool(fam_qualified >= 2
                             and tail_dir is not False)
        verdict_tools[tool] = {
            "families_direction_positive": fam_positive,
            "families_ci_qualified": fam_qualified,
            "family_detail": fam_detail,
            "highdoc_q90_direction_positive": (tail_dir
                                               if tail_dir is not None
                                               else "insufficient_samples"),
            "ranking_value_at_top10": ranking_value,
        }
    n_valued = sum(1 for v in verdict_tools.values()
                   if v["ranking_value_at_top10"])
    joint_note = ("joint high-risk (>=2 tools) error elevation is in "
                  "r2_ranking.csv per-cell analysis of Phase 4 scope")
    verdict = {
        "verdict": "phase3b_r2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec": "docs/paper/phase3_r2_ranking_spec.md",
        "tools": verdict_tools,
        "tools_with_ranking_value": n_valued,
        "route": ("Phase 4 restricted to >=2-tool joint high-risk regions"
                  if n_valued >= 2 else
                  "R2 failed: close stable-blind-spot and active-sampling "
                  "claims; keep reconstruction, calibration, coverage-width "
                  "trade-off and the negative ranking result"),
        "note": joint_note,
    }
    (outdir / "r2_verdict.json").write_text(json.dumps(verdict, indent=2),
                                            encoding="utf-8")

    lines = [
        "# 3B-R2 ranking report",
        "",
        ("Selection score = v2 uncertainty only (spec frozen before the run). "
        "Test DOC joined at final evaluation only."),
        "",
        "## Main operating point (top 10% ranking diagnostic)",
        "",
        main_rows.groupby(["tool", "family"])[
            ["enrichment_log1p", "enrichment_mg", "n_selected"]
        ].mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Judgment",
        "",
        pd.DataFrame(verdict_tools).T.to_markdown(),
        "",
        f"Tools with ranking value: **{n_valued}** → {verdict['route']}",
        "",
        "## Rank association (Spearman, diagnostic)",
        "",
        rep[rep.layer == "rank_association"].groupby(
            ["tool", "family"])["spearman_unc_abslogerr"].mean()
        .reset_index().to_markdown(index=False, floatfmt=".3f"),
    ]
    (outdir / "r2_ranking_report.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    print(pd.DataFrame(verdict_tools).T.to_string())
    print()
    print(main_rows.groupby(["tool", "family"])[
        ["enrichment_log1p", "enrichment_ci_lo", "enrichment_ci_hi",
         "highdoc_q90_recall_enr", "highdoc_q90_n"]].mean().round(3)
        .to_string())


if __name__ == "__main__":
    main()
