"""Freeze and analyze the Phase-2B pilot — corrected edition (2B-R1).

Corrections over the first analysis (the pilot has been seen; corrections are
recorded in frozen/freeze_report_2b.md):

1. Two comparison sets are reported separately and never merged:
   - tabular set (H2X vs eco_RF / eco_MLP): supports only "better than the
     current tabular implementations";
   - message-ablation set (H2X vs H2X_nomsg, matched inputs): the evidence
     about edge messages / topology. The no-message control is not moved out
     of the gate by naming.
2. Main table and bootstrap intervals estimate the SAME quantity: per-mask
   paired means, equal-weighted over masks within a seed, then averaged over
   training seeds. (The previous bootstrap pooled rows and was weighted by
   observation count.)
3. Seed direction counts use training seeds (k of N seeds), not seed x mask
   pairs. Spread is decomposed: mae_sd_seed (training-seed spread of the
   seed-level mean) vs mae_sd_mask (mask spread).
4. A CI covering 0 is reported as "insufficient evidence", never equivalence.
5. High-DOC tables carry denominators (true-high cell counts per mask); small
   samples are reported as "undetected, estimate unstable". Seeds re-predict
   the same cells and do not increase the ecological sample size.
6. Input-visibility shift between the fit stage and the test stage is
   recorded (it is a real covariate shift for tabular arms).

2B is a directional pilot: none of these numbers is a paper claim
(docs/paper/phase2_ablation_spec.md §7).
"""

from __future__ import annotations

import json
import re
from collections import Counter
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
FAMILIES = ("E1", "E2a", "E2b", "E3")
# (label, arm_a, arm_b): tabular comparisons vs matched message ablation
COMPARISONS = [
    ("tabular", "H2X", "eco_RF"),
    ("tabular", "H2X", "eco_MLP"),
    ("message_ablation", "H2X", "H2X_nomsg"),
    ("encoder", "H2X", "H2E"),
]

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
            "config_hash_version": meta.get("config_hash_version", 1),
            "run_identity_sha256": meta.get("run_identity_sha256"),
            "runtime_code_snapshot_sha256": meta.get("runtime_code_snapshot_sha256"),
        })
    return pd.DataFrame(rows)


def seed_level_mae(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, family, seed): equal-weight mean over the family's masks."""
    rows = []
    for (arm, fam, seed), g in cells.groupby(["arm", "family", "seed"]):
        rows.append({"arm": arm, "family": fam, "seed": seed,
                     "mae_seed_mean": g["mae"].mean(),
                     "r2_seed_mean": g["r2"].mean(),
                     "n_masks": len(g)})
    return pd.DataFrame(rows)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    """Main table: equal-weight over masks within seed, then over seeds."""
    sl = seed_level_mae(cells)
    rows = []
    for arm in ARM_ORDER:
        for fam in FAMILIES:
            g = sl[(sl["arm"] == arm) & (sl["family"] == fam)]
            gc = cells[(cells["arm"] == arm) & (cells["family"] == fam)]
            if not len(g):
                continue
            mask_means = gc.groupby("mask")["mae"].mean()
            rows.append({
                "arm": arm, "family": fam,
                "n_seed": g["seed"].nunique(),
                "n_mask": gc["mask"].nunique(),
                "mae_mean": g["mae_seed_mean"].mean(),
                "mae_sd_seed": g["mae_seed_mean"].std(ddof=0),
                "mae_sd_mask": mask_means.std(ddof=0),
                "r2_mean": g["r2_seed_mean"].mean(),
                "n_test_mean": gc["n"].mean(),
            })
    return pd.DataFrame(rows)


def paired_frame(arm_a: str, arm_b: str, seed: int, mask: str) -> pd.DataFrame:
    fa = PRED / _file_of(arm_a, seed, mask)
    fb = PRED / _file_of(arm_b, seed, mask)
    if not fa.is_file() or not fb.is_file():
        return pd.DataFrame(columns=["station", "delta"])
    da = pd.read_parquet(fa)
    db = pd.read_parquet(fb)
    da = da[da["split"] == "test"].set_index(["station", "month"])
    db = db[db["split"] == "test"].set_index(["station", "month"])
    j = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
    if not len(j):
        return pd.DataFrame(columns=["station", "delta"])
    delta = (np.abs(j["y_true_a"] - j["y_pred_a"])
             - np.abs(j["y_true_a"] - j["y_pred_b"])).to_numpy()
    return pd.DataFrame({
        "station": np.asarray(j.index.get_level_values(0)), "delta": delta,
    })


_FILES: dict[tuple[str, int, str], str] = {}


def _file_of(arm: str, seed: int, mask: str) -> str:
    key = (arm, seed, mask)
    if key not in _FILES:
        for p in PRED.glob(f"P2X_{arm}*"):
            m = NAME_RE.match(p.stem)
            if (m and m.group("arm") == arm and int(m.group("seed")) == seed
                    and m.group("mask") == mask):
                _FILES[key] = p.name
                break
        else:
            _FILES[key] = ""
    return _FILES[key]


def estimate_paired(fam_frames: dict[tuple[int, str], pd.DataFrame],
                    seeds: list[int], masks: list[str],
                    stations: np.ndarray, counts: Counter | None) -> float:
    """Equal-mask-weighted mean delta within seed, then mean over seeds."""
    per_seed = []
    for seed in seeds:
        mask_means = []
        for mask in masks:
            df = fam_frames.get((seed, mask))
            if df is None or not len(df):
                continue
            if counts is None:
                mask_means.append(float(df["delta"].mean()))
            else:
                w = df["station"].map(counts).fillna(0).to_numpy()
                if w.sum() == 0:
                    continue
                mask_means.append(float(np.sum(df["delta"].to_numpy() * w) / w.sum()))
        if mask_means:
            per_seed.append(float(np.mean(mask_means)))
    return float(np.mean(per_seed)) if per_seed else float("nan")


def paired_deltas(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fam in FAMILIES:
        masks = sorted(cells[cells["family"] == fam]["mask"].unique())
        seeds = sorted(int(s) for s in cells[cells["family"] == fam]["seed"].unique())
        for set_label, a, b in COMPARISONS:
            frames: dict[tuple[int, str], pd.DataFrame] = {}
            for seed in seeds:
                for mask in masks:
                    df = paired_frame(a, b, seed, mask)
                    if len(df):
                        frames[(seed, mask)] = df
            if not frames:
                continue
            point = estimate_paired(frames, seeds, masks, None, None)
            # seed-level signs (training seeds only, after equal-mask aggregation)
            seed_vals = []
            for seed in seeds:
                v = estimate_paired(frames, [seed], masks, None, None)
                if np.isfinite(v):
                    seed_vals.append(v)
            # station-clustered bootstrap of the SAME estimand
            uni = np.unique(np.concatenate(
                [f["station"].to_numpy() for f in frames.values()]))
            rng = np.random.default_rng(0)
            stats = np.empty(BOOT_DRAWS)
            for i in range(BOOT_DRAWS):
                picked = rng.choice(uni, size=len(uni), replace=True)
                stats[i] = estimate_paired(
                    frames, seeds, masks, uni, Counter(picked.tolist()))
            lo, hi = (float(np.percentile(stats, 2.5)),
                      float(np.percentile(stats, 97.5)))
            sl = seed_level_mae(cells)
            mae_a = sl[(sl["arm"] == a) & (sl["family"] == fam)]["mae_seed_mean"].mean()
            mae_b = sl[(sl["arm"] == b) & (sl["family"] == fam)]["mae_seed_mean"].mean()
            out.append({
                "set": set_label, "family": fam, "arm_a": a, "arm_b": b,
                "mae_a": mae_a, "mae_b": mae_b,
                "rel_reduction_pct": 100.0 * (mae_b - mae_a) / mae_b
                if mae_b else np.nan,
                "paired_d_mae": point, "ci95_lo": lo, "ci95_hi": hi,
                "ci_excludes_0": bool(lo > 0 or hi < 0),
                "evidence": ("A better" if hi < 0 else
                             "B better" if lo > 0 else "insufficient evidence"),
                "seeds_a_better": int(sum(1 for v in seed_vals if v < 0)),
                "n_seeds": len(seed_vals),
            })
    return pd.DataFrame(out)


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
            n_pred = int(pred_hi.sum())
            hits = int((true_hi & pred_hi).sum())
            out.append({
                "file": row["file"], "arm": row["arm"], "seed": row["seed"],
                "mask": row["mask"], "family": row["family"], "q": p,
                "threshold_mg_l": thr,
                "n_true_high": n_hi, "n_pred_high": n_pred, "n_hits": hits,
                "precision": hits / n_pred if n_pred else np.nan,
                "recall": hits / n_hi if n_hi else np.nan,
                "tail_mae": float(np.mean(np.abs(yte[true_hi] - ype[true_hi])))
                if n_hi else np.nan,
                "tail_sqerr_share": float(sq[true_hi].sum() / sq.sum())
                if n_hi and sq.sum() else np.nan,
                "small_sample_flag": bool(n_hi < 20),
            })
    return pd.DataFrame(out)


def input_shift_diagnostics() -> pd.DataFrame:
    """Fit-stage vs test-stage visible-DOC counts (per mask, tabular features)."""
    import torch

    from river_graph.baselines.baselines import ecological_tabular_features

    ds = torch.load(ROOT / "data/processed/mississippi_graph_graphfix_st357.pt",
                    weights_only=False)
    rows = []
    for mask_path in sorted((ROOT / "experiments/masks_stcore_v1").glob("*.npz")):
        mask = mask_path.stem
        split = dict(np.load(mask_path))
        for stage, keep in (("fit", {"train", "context"}),
                            ("test", {"train", "val", "context"})):
            feats, names = ecological_tabular_features(
                ds, split, visibility=keep)
            cnt = feats[:, names.index("month_visdoc_count_excl_self")]
            tr = np.asarray(split["train"], dtype=np.int64)
            te = np.asarray(split["test"], dtype=np.int64)
            rows.append({
                "mask": mask, "family": family(mask), "stage": stage,
                "train_rows_median_visdoc": float(np.median(cnt[tr])),
                "train_rows_mean_visdoc": float(np.mean(cnt[tr])),
                "test_rows_median_visdoc": float(np.median(cnt[te])),
                "test_rows_mean_visdoc": float(np.mean(cnt[te])),
            })
    return pd.DataFrame(rows)


def highdoc_summary(hot: pd.DataFrame) -> pd.DataFrame:
    """Pooled detection with denominators; seeds do not add ecological samples."""
    rows = []
    for (arm, fam, q), g in hot.groupby(["arm", "family", "q"]):
        n_hi = g.groupby("mask")["n_true_high"].first().sum()  # unique cells
        n_hits = g["n_hits"].sum()
        n_pred = g["n_pred_high"].sum()
        rows.append({
            "arm": arm, "family": fam, "q": q,
            "n_true_high_unique_cells": int(n_hi),
            "n_seed_repeats": g["seed"].nunique(),
            "recall_pooled": n_hits / g["n_true_high"].sum()
            if g["n_true_high"].sum() else np.nan,
            "precision_pooled": n_hits / n_pred if n_pred else np.nan,
            "tail_mae_mean": g["tail_mae"].mean(),
            "tail_sqerr_share_mean": g["tail_sqerr_share"].mean(),
            "small_sample_flag": bool(n_hi < 20),
        })
    return pd.DataFrame(rows)


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    if len(cells) != 144:
        raise SystemExit(f"expected 144 result cells, found {len(cells)}")
    summary = summarize(cells)
    deltas = paired_deltas(cells)
    hot = highdoc_tables(cells)
    hot_sum = highdoc_summary(hot)
    shift = input_shift_diagnostics()

    cells.to_csv(FROZEN / "benchmark_2b_pilot.csv", index=False)
    summary.to_csv(FROZEN / "benchmark_2b_summary.csv", index=False)
    seed_level_mae(cells).to_csv(ANALYSIS / "seed_level_mae.csv", index=False)
    deltas.to_csv(ANALYSIS / "paired_deltas.csv", index=False)
    hot.to_csv(ANALYSIS / "highdoc_tables.csv", index=False)
    hot_sum.to_csv(ANALYSIS / "highdoc_summary.csv", index=False)
    shift.to_csv(ANALYSIS / "input_shift_diagnostics.csv", index=False)

    report = [
        "# Phase-2B freeze report (directional pilot) — corrected 2B-R1",
        "",
        (f"Frozen at {datetime.now(timezone.utc).isoformat()} by "
        "`scripts/analyze_phase2b.py` (one-command recompute). **2B is a "
        "pilot: these numbers are not paper claims** (spec §7)."),
        "",
        "## Corrections applied (pilot already seen)",
        "",
        ("- Comparison sets are separated: `tabular` (H2X vs eco_RF/eco_MLP) "
        "supports only \"better than the current tabular implementations\"; "
        "`message_ablation` (H2X vs H2X_nomsg, matched inputs) is the "
        "evidence about edge messages. The no-message control stays in the "
        "gate comparator set — renaming cannot remove it."),
        ("- Main table and bootstrap now estimate the same quantity "
        "(equal-mask within seed, then mean over seeds)."),
        "- Seed direction counts use training seeds (k of N), not seed×mask.",
        ("- CI covering 0 = insufficient evidence, never equivalence. "
        "`mae_sd_seed` is training-seed spread; `mae_sd_mask` is mask spread."),
        ("- High-DOC rows carry denominators; small samples say "
        "\"undetected, estimate unstable\". Seeds re-predict the same cells."),
        "",
        "## Mean MAE (equal-mask within seed, then mean over seeds)",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        (f"## Paired comparisons (same estimand; station-clustered bootstrap, "
        f"{BOOT_DRAWS} draws)"),
        "",
        deltas.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## High-DOC (pooled detection with unique-cell denominators)",
        "",
        hot_sum.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Input-visibility shift (tabular features, fit stage vs test stage)",
        "",
        shift.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## What each comparison set can support",
        "",
        ("- tabular set: \"H2X predicts better than these RF/MLP "
        "implementations\" — it says nothing about topology."),
        ("- message_ablation set: matched-inputs evidence about edge messages. "
        "If no-message matches H2X, edge messages are not shown necessary."),
        ("- encoder set (H2X vs H2E): incremental value of the encoder beyond "
        "raw ecological context."),
    ]
    (FROZEN / "freeze_report_2b.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (ANALYSIS / "phase2b_analysis.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
