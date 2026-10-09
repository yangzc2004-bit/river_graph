"""3B-R0: one-command recompute of every Phase-3B judgment artifact.

Regenerates from the stored seed ensembles + products (no training):
coverage (with station-clustered bootstrap CIs, incl. tail subsets),
strata coverage (with CI), monotonicity (with structural-identifiability
labels), calibration diagnostics (train/val/test standardized-score
quantiles), and the auto-generated verdict v1_1.

Acceptance-record v1.1 rules encoded here (revision recorded after results
were seen; original verdict stays untouched at eval/phase3b_verdict.md):
- overall coverage = equal-mask mean over the 8 masks of per-mask test
  coverage; gate [0.80, 0.95];
- strata = pooled hidden-test cells per land-cover-dominant stratum,
  n >= 50; gate [0.70, 0.98]; the stratum definition is POST-HOC (chosen
  a priori-natural, NOT optimized) — recorded as such;
- monotonicity keeps the frozen tertile rule; network_distance tertile
  binning is structurally degenerate (a visible station's distance to the
  nearest visible station is 0 by definition) -> those rows are labelled
  ``not_identifiable_by_design`` and never counted as passes;
- tool "passes calibration" iff overall AND all strata pass; Phase 4/5 need
  >= 2 tools passing — passing does not by itself unlock anything;
- interval recalibration cannot change monotonicity (U(x) = qhat * s_eff is
  a single positive scalar per product: rankings, directions, strong-
  reversal calls and model_agreement are invariant to qhat).

Usage: python scripts/eval3b.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.phase3_uncertainty import (
    SEEDS,
    TOOLS,
    calibration_cells,
    covariates,
    family_of,
    monotonicity_check,
    scenario_visibility,
)

DATASET = "data/processed/mississippi_graph_graphfix_st357.pt"
MASKS_DIR = Path("experiments/masks_stcore_v1")
OUT = Path("experiments/phase3_uncertainty_stcore_v1")
M = ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44", "e2a_strict",
     "e2b_partial", "e3_spatial_seed42", "e3_spatial_seed43",
     "e3_spatial_seed44"]
STRATA_MIN_N = 50


def cluster_ci(hit: np.ndarray, stations: np.ndarray, draws: int = 2000,
               seed: int = 0) -> tuple[float, float]:
    uniq = np.unique(stations)
    idx = {s: np.flatnonzero(stations == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    for i in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in picked])
        stats[i] = float(hit[sel].mean())
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def main() -> None:
    ds = torch.load(DATASET, weights_only=False)
    n, t = ds["y"].shape
    y = ds["y"].numpy()
    y_log = np.log1p(y)
    sites = np.asarray([str(s) for s in ds["site_no"]])
    regime = np.asarray(ds["regime"], dtype=float)
    strata = np.array(["forest", "crop_hay", "urban", "wetland"])[
        regime[:, 4:8].argmax(axis=1)]
    outdir = OUT / "eval"
    outdir.mkdir(parents=True, exist_ok=True)

    cov_rows, strata_rows, mono_rows, cal_rows = [], [], [], []
    for mask in M:
        split = dict(np.load(MASKS_DIR / f"{mask}.npz"))
        fam = family_of(mask)
        vis = scenario_visibility(split, fam)
        calibration_cells(split, fam)
        te = np.asarray(split["test"], dtype=np.int64)
        cov = covariates(ds, vis)
        vis_st = np.flatnonzero(
            np.bincount(vis, minlength=n * t).reshape(n, t).sum(axis=1) > 0)
        st_te = sites[te // t]
        yte = y.reshape(-1)[te]
        for tool in TOOLS:
            pdf = pd.read_parquet(OUT / "full_grid" / f"P3_{tool}__{mask}.parquet")
            pdf["prediction_median"].to_numpy().reshape(n, t).reshape(-1)
            lo = pdf["pi_lower"].to_numpy().reshape(n, t).reshape(-1)
            hi = pdf["pi_upper"].to_numpy().reshape(n, t).reshape(-1)
            unc = pdf["uncertainty"].to_numpy().reshape(n, t)
            hit = ((yte >= lo[te]) & (yte <= hi[te])).astype(float)
            clo, chi = cluster_ci(hit, st_te)
            cov_rows.append({
                "tool": tool, "mask": mask, "family": fam, "subset": "test",
                "coverage": float(hit.mean()), "ci95_lo": clo, "ci95_hi": chi,
                "n": len(te), "mean_width_mg_l": float(np.mean(hi[te] - lo[te])),
            })
            tr = np.asarray(split["train"], dtype=np.int64)
            ytr = y.reshape(-1)[tr]
            for p, tag in ((0.95, "top5"), (0.90, "top10")):
                thr = float(np.quantile(ytr, p))
                s = yte >= thr
                hc = hit[s]
                tlo, thi = (cluster_ci(hc, st_te[s]) if s.sum() >= 20
                            else (np.nan, np.nan))
                cov_rows.append({
                    "tool": tool, "mask": mask, "family": fam, "subset": tag,
                    "coverage": float(hc.mean()) if s.sum() else np.nan,
                    "ci95_lo": tlo, "ci95_hi": thi, "n": int(s.sum()),
                    "mean_width_mg_l": float(np.mean(hi[te][s] - lo[te][s]))
                    if s.sum() else np.nan,
                    "small_sample_flag": bool(s.sum() < 20),
                })
            for sname in ("forest", "crop_hay", "urban", "wetland"):
                sel = strata[te // t] == sname
                if sel.sum() < STRATA_MIN_N:
                    continue
                hc = hit[sel]
                slo, shi = cluster_ci(hc, st_te[sel])
                strata_rows.append({
                    "tool": tool, "mask": mask, "family": fam,
                    "stratum": sname, "n": int(sel.sum()),
                    "coverage": float(hc.mean()),
                    "ci95_lo": slo, "ci95_hi": shi,
                })
            for cname, direc in (("network_distance", "up"),
                                 ("ecological_novelty", "up"),
                                 ("support_count", "down")):
                r = monotonicity_check(unc, cov[cname], direc, te, vis_st, n, t)
                if cname == "network_distance":
                    edges = r.get("edges")
                    if (r.get("ok") is None
                            or (edges and edges[0] == 0 and edges[1] == 0)):
                        r = {"ok": None,
                             "reason": "not_identifiable_by_design: visible "
                                       "stations' distance to nearest visible "
                                       "station is 0; tertile binning is "
                                       "structurally degenerate"}
                mono_rows.append({"tool": tool, "mask": mask, "family": fam,
                                  "covariate": cname, **r})
            # calibration diagnostics: standardized-score quantiles per role
            sp = pd.read_parquet(
                OUT / "seed_preds" / f"P3_{tool}_s42-46__{mask}.parquet")
            stack = np.column_stack([sp[f"pred_s{s}"].to_numpy()
                                     for s in SEEDS]).T
            seed_log = np.log1p(stack.reshape(len(SEEDS), n, t))
            median = np.median(seed_log, axis=0)
            s_eff = np.maximum(seed_log.std(axis=0),
                               max(float(np.quantile(seed_log.std(axis=0),
                                                     0.10)), 1e-3))
            score = (np.abs(y_log - median) / s_eff).reshape(-1)
            for role, cells in (("train", tr), ("val", np.asarray(
                    split.get("val", []), dtype=np.int64)), ("test", te)):
                sc = score[cells]
                cal_rows.append({
                    "tool": tool, "mask": mask, "family": fam, "role": role,
                    "n": len(cells),
                    "score_q50": float(np.quantile(sc, 0.5)),
                    "score_q90": float(np.quantile(sc, 0.9)),
                })

    cov_df = pd.DataFrame(cov_rows)
    strata_df = pd.DataFrame(strata_rows)
    mono_df = pd.DataFrame(mono_rows)
    cal_df = pd.DataFrame(cal_rows)
    cov_df.to_csv(outdir / "coverage.csv", index=False)
    strata_df.to_csv(outdir / "strata_coverage.csv", index=False)
    mono_df.to_csv(outdir / "monotonicity.csv", index=False)
    cal_df.to_csv(outdir / "calibration_diagnostics.csv", index=False)

    # ---- gate judgment (acceptance-record v1.1) ----
    gates = {}
    for tool in TOOLS:
        per_mask = cov_df[(cov_df.tool == tool)
                          & (cov_df["subset"] == "test")]["coverage"]
        overall = float(per_mask.mean())
        strat = strata_df[strata_df.tool == tool]
        pooled = strat.groupby("stratum").apply(
            lambda g: float(np.average(g["coverage"], weights=g["n"])),
            include_groups=False,
        )
        strat_ok = bool(len(pooled) and ((pooled >= 0.70)
                                         & (pooled <= 0.98)).all())
        overall_ok = bool(0.80 <= overall <= 0.95)
        gates[tool] = {
            "overall_equal_mask_mean": overall,
            "overall_gate_080_095": overall_ok,
            "strata_pooled": {k: round(v, 3) for k, v in pooled.items()},
            "strata_all_in_070_098": strat_ok,
            "passes_calibration": bool(overall_ok and strat_ok),
        }
    n_pass = sum(1 for g in gates.values() if g["passes_calibration"])
    verdict = {
        "verdict": "phase3b_v1_1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "overall": "equal-mask mean over 8 masks of per-mask test coverage",
            "strata": ("pooled hidden-test cells per land-cover-dominant "
                       "stratum (POST-HOC defined, not optimized), n>=50"),
            "monotonicity": ("frozen tertile rule; network_distance marked "
                             "not_identifiable_by_design (degenerate binning)"),
            "invariance": ("U(x)=qhat*s_eff: interval recalibration cannot "
                           "change monotonicity or model_agreement"),
        },
        "tools": gates,
        "tools_passing_calibration": n_pass,
        "phase4_phase5": ("unlocked" if n_pass >= 2 else
                          "PAUSED (fewer than 2 tools pass calibration)"),
        "original_verdict": "eval/phase3b_verdict.md (v1, kept unchanged)",
    }
    (outdir / "verdict_v1_1.json").write_text(json.dumps(verdict, indent=2),
                                              encoding="utf-8")
    lines = [
        "# Phase-3B verdict v1.1 (auto-generated, 3B-R0)",
        "",
        ("Original v1 verdict kept unchanged at `eval/phase3b_verdict.md`. "
        "Revisions recorded here were made AFTER results were seen and are "
        "labelled as acceptance-record corrections, not as fresh gates."),
        "",
        "## Calibration gates (coverage)",
        "",
        pd.DataFrame(gates).T.to_markdown(floatfmt=".3f"),
        "",
        f"## Phase 4/5 status: **{verdict['phase4_phase5']}**",
        "",
        "## Calibration diagnostics (standardized scores, in-sample vs out)",
        "",
        cal_df.groupby(["tool", "role"])[["score_q50", "score_q90"]]
        .mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interval widths (mean mg/L, test cells)",
        "",
        cov_df[cov_df["subset"] == "test"]
        .groupby("tool")["mean_width_mg_l"].mean().reset_index()
        .to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Corrections to the v1 record",
        "",
        ("- land-cover strata are **post-hoc defined (not optimized)** — the "
        "v1 text calling them a priori is corrected here;"),
        ("- network_distance tertile binning is **structurally degenerate** "
        "(visible stations' distance is 0 by definition): those monotonicity "
        "rows are not-identifiable-by-design, never passes;"),
        ("- provenance: seed files carry retrospective content pins only; see "
        "`eval/provenance_audit.json` for the gap list (no post-hoc "
        "re-stamping as contemporaneous)."),
        "",
        ("Passing these gates does NOT by itself unlock Phase 4/5; the "
        "three-tool rule and the independent ranking-value review "
        "(3B-R2) still apply."),
    ]
    (outdir / "phase3b_verdict_v1_1.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
