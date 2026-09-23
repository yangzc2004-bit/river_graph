"""Freeze and analyze the Phase-2B directional pilot (recomputable, no training).

Reads the P2X_* results and prediction parquets under
experiments/phase2_ablation_stcore_v1/ and writes:

* frozen/benchmark_2b_pilot.csv      — frozen 144-cell metric table (authoritative for 2B)
* frozen/benchmark_2b_summary.csv    — per-arm x scenario-family means
* frozen/freeze_report_2b.md         — freeze report + pilot decision-table verdicts
* analysis/paired_deltas.csv         — paired ΔMAE with station-clustered bootstrap CI
* analysis/highdoc_tables.csv        — top 5%/10% precision, recall, tail MAE and error share
* analysis/phase2b_analysis.md       — scenario-split reading (E2a/E2b never pooled)

2B is a directional pilot: none of these numbers is a paper claim
(docs/paper/phase2_ablation_spec.md §7).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "phase2_ablation_stcore_v1"
PRED = OUT / "predictions"
FROZEN = OUT / "frozen"
ANALYSIS = OUT / "analysis"
BOOT_DRAWS = 2000
ARM_ORDER = ["H2", "H2E", "H2X", "H2X_nomsg", "eco_RF", "eco_MLP"]

NAME_RE = re.compile(
    r"^P2X_(?P<arm>H2X_nomsg|eco_RF|eco_MLP|H2X|H2E|H2)"
    r"(?:_river)?_s(?P<seed>\d+)__(?P<mask>.+)$"
)


def family(mask: str) -> str:
    if mask.startswith("e1"):
        return "E1"
    if mask.startswith("e2a"):
        return "E2a"
    if mask.startswith("e2b"):
        return "E2b"
    return "E3"


def load_cells() -> pd.DataFrame:
    rows = []
    for p in sorted(PRED.glob("*.parquet")):
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        met = meta.get("metrics") or {}
        rows.append({
            "file": p.name, "arm": m.group("arm"), "seed": int(m.group("seed")),
            "mask": m.group("mask"), "family": family(m.group("mask")),
            "mae": met.get("mae"), "r2": met.get("r2"), "rmse": met.get("rmse"),
            "n": met.get("n"),
            "config_hash": meta.get("config_hash"),
            "run_identity_sha256": meta.get("run_identity_sha256"),
            "runtime_code_snapshot_sha256": meta.get("runtime_code_snapshot_sha256"),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(diff: np.ndarray, stations: np.ndarray,
                      draws: int = BOOT_DRAWS, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of mean(diff), resampling stations with all their rows."""
    uniq = np.unique(stations)
    idx_by_station = {s: np.flatnonzero(stations == s) for s in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    for i in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_station[s] for s in picked])
        stats[i] = float(np.mean(diff[sel]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_deltas(cells: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("H2X", "H2E"), ("H2X", "H2X_nomsg"),
        ("H2X", "eco_RF"), ("H2X", "eco_MLP"),
    ]
    out = []
    for fam in ("E1", "E2a", "E2b", "E3"):
        sub = cells[cells["family"] == fam]
        for a, b in pairs:
            diffs, stations, seed_signs = [], [], []
            for seed in sorted(sub["seed"].unique()):
                for mask in sorted(sub["mask"].unique()):
                    pa = PRED / f"{_stem(cells, a, seed, mask)}"
                    pb = PRED / f"{_stem(cells, b, seed, mask)}"
                    if not pa.is_file() or not pb.is_file():
                        continue
                    da = pd.read_parquet(pa)
                    db = pd.read_parquet(pb)
                    da = da[da["split"] == "test"].set_index(["station", "month"])
                    db = db[db["split"] == "test"].set_index(["station", "month"])
                    joined = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
                    if not len(joined):
                        continue
                    d = (np.abs(joined["y_true_a"] - joined["y_pred_a"])
                         - np.abs(joined["y_true_a"] - joined["y_pred_b"])).to_numpy()
                    diffs.append(d)
                    stations.append(np.asarray(joined.index.get_level_values(0)))
                    seed_signs.append(float(np.mean(d)))
            if not diffs:
                continue
            diff = np.concatenate(diffs)
            st = np.concatenate(stations)
            lo, hi = cluster_bootstrap(diff, st)
            mean_a = sub[sub["arm"] == a]["mae"].mean()
            mean_b = sub[sub["arm"] == b]["mae"].mean()
            out.append({
                "family": fam, "arm_a": a, "arm_b": b,
                "mae_a": mean_a, "mae_b": mean_b,
                "rel_reduction_pct": 100.0 * (mean_b - mean_a) / mean_b
                if mean_b else np.nan,
                "paired_d_mae": float(np.mean(diff)),
                "ci95_lo": lo, "ci95_hi": hi,
                "seed_signs_a_better": int(sum(1 for s in seed_signs if s < 0)),
                "seed_signs_total": len(seed_signs),
                "n_rows": len(diff),
            })
    return pd.DataFrame(out)


def _stem(cells: pd.DataFrame, arm: str, seed: int, mask: str) -> str:
    hit = cells[(cells["arm"] == arm) & (cells["seed"] == seed)
                & (cells["mask"] == mask)]
    return hit["file"].iloc[0]


def highdoc_tables(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in cells.iterrows():
        df = pd.read_parquet(PRED / row["file"])
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        ytr = tr["y_true"].to_numpy(float)
        yte = te["y_true"].to_numpy(float)
        ype = te["y_pred"].to_numpy(float)
        sq = (yte - ype) ** 2
        for p in (0.95, 0.90):
            thr = float(np.quantile(ytr, p))
            true_hi = yte >= thr
            pred_hi = ype >= thr
            n_hi = int(true_hi.sum())
            if n_hi:
                recall = float((true_hi & pred_hi).sum() / n_hi)
                tail_mae = float(np.mean(np.abs(yte[true_hi] - ype[true_hi])))
                tail_share = float(sq[true_hi].sum() / sq.sum()) if sq.sum() else np.nan
            else:
                recall = tail_mae = tail_share = np.nan
            precision = (float((true_hi & pred_hi).sum() / pred_hi.sum())
                         if pred_hi.sum() else np.nan)
            out.append({
                "file": row["file"], "arm": row["arm"], "seed": row["seed"],
                "mask": row["mask"], "family": row["family"], "q": p,
                "threshold_mg_l": thr, "n_true_high": n_hi,
                "precision": precision, "recall": recall,
                "tail_mae": tail_mae, "tail_sqerr_share": tail_share,
            })
    return pd.DataFrame(out)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARM_ORDER:
        for fam in ("E1", "E2a", "E2b", "E3"):
            v = cells[(cells["arm"] == arm) & (cells["family"] == fam)]
            if not len(v):
                continue
            rows.append({
                "arm": arm, "family": fam, "n_cells": len(v),
                "mae_mean": v["mae"].mean(), "mae_sd": v["mae"].std(ddof=0),
                "r2_mean": v["r2"].mean(), "n_test_mean": v["n"].mean(),
            })
    return pd.DataFrame(rows)


def freeze_report(cells, summary, deltas, hot) -> str:
    lines = [
        "# Phase-2B freeze report (directional pilot)",
        "",
        (f"Frozen at {datetime.now(timezone.utc).isoformat()} by "
        "`scripts/analyze_phase2b.py` (recomputable). **2B is a pilot: these "
        "numbers are not paper claims** (docs/paper/phase2_ablation_spec.md §7)."),
        "",
        ("Policy bindings verified by `scripts/run2b_executor.py` before every "
        "unit: `configs/phase2_2c_policy.json` (h2x_policy_sha256, "
        "endpoints_sha256, dataset/mask hashes), the runtime snapshot pin, and "
        "run identity (`config_hash` + `run_identity_sha256` + "
        "`runtime_code_snapshot_sha256`) on all 144 sidecars. Audit: 144 "
        "`identity ok`, zero retrained duplicates, zero suspicious names."),
        "",
        ("2B pilot -> frozen reconciliation: the discarded pilot criterion "
        "(15%, two-of-three) is non-operative; operative gates remain "
        "endpoints-v1 (>=10% union over {E2a, E2b, E3} + per-scenario no-harm "
        "margins). E2a and E2b are separate estimands and are never pooled."),
        "",
        "## Mean MAE (mg/L) by arm x scenario family (3 seeds x key masks)",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        ("## Paired comparisons (station-clustered bootstrap, "
        f"{BOOT_DRAWS} draws)"),
        "",
        deltas.to_markdown(index=False, floatfmt=".3f"),
        "",
        ("## High-DOC identification (mean over cells, threshold = train "
        "quantile)"),
        "",
        hot.groupby(["arm", "family", "q"])[["precision", "recall", "tail_mae",
                                             "tail_sqerr_share"]]
        .mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Decision-table reading (pilot only)",
        "",
    ]
    piv = summary.pivot(index="arm", columns="family", values="mae_mean")

    def g(a, f):
        try:
            return piv.loc[a, f]
        except KeyError:
            return float("nan")

    e2x, e2e = g("H2X", "E2a"), g("H2E", "E2a")
    b2x, b2e = g("H2X", "E2b"), g("H2E", "E2b")
    x3, e3 = g("H2X", "E3"), g("H2E", "E3")
    nom = min(g("H2X_nomsg", f) for f in ("E2a", "E2b", "E3"))
    best_ng = min(g("eco_RF", "E2a"), g("eco_MLP", "E2a"),
                  g("eco_RF", "E2b"), g("eco_MLP", "E2b"),
                  g("eco_RF", "E3"), g("eco_MLP", "E3"))
    lines += [
        f"- H2X vs H2E (encoder gain?): E2a {e2x:.3f} vs {e2e:.3f}, "
        f"E2b {b2x:.3f} vs {b2e:.3f}, E3 {x3:.3f} vs {e3:.3f} — "
        + ("encoder adds value beyond raw context (pilot)" if
           min(e2e - e2x, b2e - b2x) > 0 and (e3 - x3) > 0 else
           "encoder gain mixed; no stable extrapolation gain yet"),
        f"- H2X vs no-message (topology necessary?): nomsg best-E2/nomsg-E3 "
        f"min {nom:.3f} vs H2X min "
        f"{min(e2x, b2x, x3):.3f} — "
        + ("edge messages add value" if min(e2x, b2x, x3) < nom else
           "no-message matches or beats H2X: river topology NOT shown "
           "necessary; do not sell topology as the core contribution"),
        (f"- H2X vs best no-graph (primary gate preview): best tabular MAE "
        f"{best_ng:.3f} vs H2X best-scenario values above — the >=10% gate is "
        "decided in 2C on seeds 42-46 with paired bootstrap, not here."),
        ("- All numbers above are pilot screening (3 seeds). They cannot appear "
        "as paper claims and cannot move a primary endpoint."),
    ]
    return "\n".join(lines) + "\n"


def analysis_note(summary, deltas, hot) -> str:
    lines = [
        "# Phase-2B analysis (directional pilot, scenario families never pooled)",
        "",
        ("E2a (strict future, no post-cutoff DOC context) and E2b (running "
        "network) are separate estimands (spec §8.1). E1 is the random-missing "
        "family; E3 is spatial extrapolation. All values: mean over 3 training "
        "seeds x the family's key masks."),
        "",
        "## Paired ΔMAE (negative = arm_a better)",
        "",
        deltas.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## High-DOC top 5%/10% (per family means)",
        "",
        hot.groupby(["arm", "family", "q"])[["precision", "recall", "tail_mae",
                                             "tail_sqerr_share"]]
        .mean().reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Reading",
        "",
        ("- Paired deltas with CI95 crossing 0 are treated as ties in the pilot "
        "reading; seed-sign counts show direction consistency only."),
        ("- Tail: `tail_sqerr_share` is the share of squared error carried by "
        "truly-high cells; `tail_mae` is MAE restricted to them."),
        ("- These are descriptive/predictive statements only (no causal or "
        "mechanistic wording per the paper charter)."),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    if len(cells) != 144:
        raise SystemExit(f"expected 144 result cells, found {len(cells)}")
    summary = summarize(cells)
    deltas = paired_deltas(cells)
    hot = highdoc_tables(cells)

    cells.to_csv(FROZEN / "benchmark_2b_pilot.csv", index=False)
    summary.to_csv(FROZEN / "benchmark_2b_summary.csv", index=False)
    deltas.to_csv(ANALYSIS / "paired_deltas.csv", index=False)
    hot.to_csv(ANALYSIS / "highdoc_tables.csv", index=False)
    (FROZEN / "freeze_report_2b.md").write_text(
        freeze_report(cells, summary, deltas, hot), encoding="utf-8"
    )
    (ANALYSIS / "phase2b_analysis.md").write_text(
        analysis_note(summary, deltas, hot), encoding="utf-8"
    )
    print("frozen ->", FROZEN / "benchmark_2b_pilot.csv")
    print(summary.to_string(index=False))
    print()
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
