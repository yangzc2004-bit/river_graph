"""A2: analyse the existing multi-seed metrics. No training of any kind.

Input is exactly the 15 result JSONs of the H1/H2/H2X multi-seed refresh:
training seed 0 is stored in ``H1f/H2f/H2Xf`` and seeds 1-4 in ``_s1``-``_s4``.
The mapping is hard-coded so a same-named legacy file can never slip in.

The output separates three aggregation levels, because conflating them is the
main way these numbers get misread:

1. ``per_run``      one value per (model, training seed, mask) - 5 x 8 per model
2. ``per_mask``     mean over the 5 training seeds, per mask; the spread here
                    is TRAINING randomness
3. ``per_scenario`` mean over the masks belonging to a scenario; E1/E3 average
                    3 masks with equal weight, E2a/E2b are single masks

``mean R2 over masks`` is NOT the R2 of all cells pooled, and the seed spread
is not a per-cell prediction uncertainty. Both caveats are emitted into the
report so they travel with the numbers.

Usage:
    python scripts/analyze_multiseed.py
    python scripts/analyze_multiseed.py --check-only   # validate, write nothing
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RESULTS = Path("experiments/results")
FROZEN = Path("experiments/frozen_results")
ANALYSIS = Path("experiments/analysis")
OUT_TABLE = FROZEN / "benchmark_multiseed.csv"

# model -> {training seed: result JSON stem}
MULTISEED_FILES: dict[str, dict[int, str]] = {
    "H1": {
        0: "H1f_directed_river",
        1: "H1_directed_river_s1",
        2: "H1_directed_river_s2",
        3: "H1_directed_river_s3",
        4: "H1_directed_river_s4",
    },
    "H2": {
        0: "H2f_transport_river",
        1: "H2_transport_river_s1",
        2: "H2_transport_river_s2",
        3: "H2_transport_river_s3",
        4: "H2_transport_river_s4",
    },
    "H2X": {
        0: "H2Xf_transport_enc_river",
        1: "H2X_transport_enc_river_s1",
        2: "H2X_transport_enc_river_s2",
        3: "H2X_transport_enc_river_s3",
        4: "H2X_transport_enc_river_s4",
    },
}

KEY_MASKS = [
    "e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44",
    "e2a_strict", "e2b_partial",
    "e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44",
]

SCENARIOS = {
    "E1": ["e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44"],
    "E2a": ["e2a_strict"],
    "E2b": ["e2b_partial"],
    "E3": ["e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44"],
}

PER_RUN_METRICS = ("mae", "r2", "rmse", "log_r2", "log_mae", "log_rmse")

# The frozen table's schema; kept byte-compatible so existing readers work.
TABLE_COLUMNS = ["model", "mask", "mae", "mae_std", "r2", "log_r2"]


class ValidationError(RuntimeError):
    """Raised when the multi-seed inputs do not form the expected grid."""


def load_runs() -> pd.DataFrame:
    """Read the 15 JSONs into one row per (model, seed, mask)."""
    rows = []
    for model, seed_files in MULTISEED_FILES.items():
        for seed, stem in sorted(seed_files.items()):
            path = RESULTS / f"{stem}.json"
            if not path.exists():
                raise ValidationError(f"missing result file: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            for mask in KEY_MASKS:
                if mask not in payload:
                    raise ValidationError(
                        f"{stem}.json is missing key mask '{mask}' "
                        f"(has {sorted(payload)})")
                entry = payload[mask]
                row = {"model": model, "seed": seed, "mask": mask,
                       "file": stem}
                for metric in PER_RUN_METRICS:
                    value = entry.get(metric)
                    if value is None:
                        raise ValidationError(
                            f"{stem}.json [{mask}] has no '{metric}' field")
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValidationError(
                            f"{stem}.json [{mask}] '{metric}' is not finite: "
                            f"{value!r}")
                    row[metric] = float(value)
                row["n"] = int(entry["n"])
                rows.append(row)
    df = pd.DataFrame(rows)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Assert the expected model x seed x mask grid, and no duplicates."""
    dupes = df.duplicated(subset=["model", "seed", "mask"]).sum()
    if dupes:
        raise ValidationError(f"{dupes} duplicated (model, seed, mask) rows")
    expected = len(MULTISEED_FILES) * 5 * len(KEY_MASKS)
    if len(df) != expected:
        raise ValidationError(f"expected {expected} runs, found {len(df)}")
    for model, group in df.groupby("model"):
        seeds = sorted(group["seed"].unique())
        if seeds != [0, 1, 2, 3, 4]:
            raise ValidationError(f"{model}: seeds {seeds} != [0..4]")
        for seed, sub in group.groupby("seed"):
            masks = sorted(sub["mask"])
            if masks != sorted(KEY_MASKS):
                raise ValidationError(f"{model} seed {seed}: masks {masks}")
        # a mask has a fixed number of test cells across training seeds
        for mask, sub in group.groupby("mask"):
            if sub["n"].nunique() != 1:
                raise ValidationError(
                    f"{model}/{mask}: test-cell count varies across seeds: "
                    f"{sorted(sub['n'].unique())}")
        # E1's 20/40/60% masks must have strictly increasing test sets
        e1 = (group[group["mask"].str.startswith("e1_")]
              .groupby("mask")["n"].first().sort_index())
        if list(e1) and not e1.is_monotonic_increasing:
            raise ValidationError(
                f"{model}: E1 test-cell counts are not increasing with the "
                f"mask rate: {e1.to_dict()}")


def per_mask_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over the 5 training seeds, one row per (model, mask)."""
    rows = []
    for (model, mask), sub in runs.groupby(["model", "mask"]):
        row = {"model": model, "mask": mask, "n_runs": len(sub),
               "test_cells": int(sub["n"].iloc[0])}
        for metric in PER_RUN_METRICS:
            values = sub[metric].tolist()
            row[f"{metric}_mean"] = st.fmean(values)
            row[f"{metric}_sd_seeds"] = (st.stdev(values)
                                         if len(values) > 1 else float("nan"))
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_table(per_mask: pd.DataFrame) -> pd.DataFrame:
    """Two levels of spread, explicitly named.

    ``*_sd_of_seed_means``: spread of the per-seed scenario means (training
    randomness). ``*_sd_of_mask_means``: spread between the masks inside the
    scenario (mask difficulty), which is NOT a training-uncertainty estimate.
    """
    rows = []
    for model in MULTISEED_FILES:
        for scenario, masks in SCENARIOS.items():
            sub = per_mask[(per_mask["model"] == model) &
                           (per_mask["mask"].isin(masks))]
            if len(sub) != len(masks):
                raise ValidationError(
                    f"{model}/{scenario}: expected {len(masks)} masks, "
                    f"found {len(sub)}")
            row = {"model": model, "scenario": scenario,
                   "n_masks": len(masks), "masks": ",".join(masks),
                   "mean_rule": ("equal weight per mask, then mean over the "
                                 "training seeds" if len(masks) > 1
                                 else "single mask")}
            for metric in ("mae", "r2", "rmse"):
                mask_means = sub[f"{metric}_mean"].tolist()
                row[f"{metric}_mean"] = st.fmean(mask_means)
                row[f"{metric}_sd_of_mask_means"] = (
                    st.stdev(mask_means) if len(mask_means) > 1
                    else float("nan"))
            rows.append(row)

    # seed-level spread per scenario: recompute from per-run values
    out = pd.DataFrame(rows)
    return out


def seed_level_scenario(runs: pd.DataFrame) -> pd.DataFrame:
    """Scenario mean computed inside each seed, then spread across seeds."""
    rows = []
    for model in MULTISEED_FILES:
        for scenario, masks in SCENARIOS.items():
            sub = runs[(runs["model"] == model) & (runs["mask"].isin(masks))]
            per_seed = (sub.groupby("seed")[["mae", "r2", "rmse"]]
                        .mean().sort_index())
            if len(per_seed) != 5:
                raise ValidationError(f"{model}/{scenario}: {len(per_seed)} seeds")
            row = {"model": model, "scenario": scenario}
            for metric in ("mae", "r2", "rmse"):
                values = per_seed[metric].tolist()
                row[f"{metric}_mean"] = st.fmean(values)
                row[f"{metric}_sd_seeds"] = st.stdev(values)
                row[f"{metric}_min_seed"] = min(values)
                row[f"{metric}_max_seed"] = max(values)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_diff(runs: pd.DataFrame, high: str, low: str) -> pd.DataFrame:
    """Same-seed, same-mask paired differences (high - low).

    MAE and R2 are reported separately; a negative MAE delta means ``high``
    has the lower (better) error.
    """
    a = runs[runs["model"] == high].set_index(["seed", "mask"])
    b = runs[runs["model"] == low].set_index(["seed", "mask"])
    common = a.index.intersection(b.index)
    if len(common) != 5 * len(KEY_MASKS):
        raise ValidationError(
            f"{high} vs {low}: only {len(common)} paired cells")
    rows = []
    for key in sorted(common):
        seed, mask = key
        rows.append({
            "comparison": f"{high}-{low}",
            "seed": seed, "mask": mask,
            "d_mae": a.loc[key, "mae"] - b.loc[key, "mae"],
            "d_r2": a.loc[key, "r2"] - b.loc[key, "r2"],
            "d_rmse": a.loc[key, "rmse"] - b.loc[key, "rmse"],
        })
    df = pd.DataFrame(rows)
    # add per-scenario summaries
    summary = []
    for scenario, masks in SCENARIOS.items():
        sub = df[df["mask"].isin(masks)]
        for metric in ("d_mae", "d_r2", "d_rmse"):
            summary.append({
                "comparison": f"{high}-{low}",
                "scope": f"scenario:{scenario}",
                "n_pairs": len(sub),
                "metric": metric,
                "mean": st.fmean(sub[metric]),
                "sd": st.stdev(sub[metric]) if len(sub) > 1 else float("nan"),
                "min": sub[metric].min(), "max": sub[metric].max(),
                "n_favouring_high": int((sub[metric] < 0).sum()
                                        if metric.startswith("d_mae")
                                        or metric == "d_rmse"
                                        else (sub[metric] > 0).sum()),
            })
    return df, pd.DataFrame(summary)


def build_frozen_table(per_mask: pd.DataFrame) -> pd.DataFrame:
    """Rebuild benchmark_multiseed.csv in its original schema."""
    out = per_mask[["model", "mask", "mae_mean", "mae_sd_seeds",
                    "r2_mean", "log_r2_mean"]].copy()
    out.columns = TABLE_COLUMNS
    return out


def compare_with_existing(new: pd.DataFrame, existing: Path) -> dict:
    if not existing.exists():
        return {"exists": False}
    old = pd.read_csv(existing)
    if list(old.columns) != list(new.columns):
        return {"exists": True, "same_columns": False,
                "old_columns": list(old.columns)}
    old = old.sort_values(["model", "mask"]).reset_index(drop=True)
    new = new.sort_values(["model", "mask"]).reset_index(drop=True)
    if len(old) != len(new):
        return {"exists": True, "same_columns": True, "same_rows": False,
                "old_rows": len(old), "new_rows": len(new)}
    diffs = {}
    for col in TABLE_COLUMNS:
        if col in ("model", "mask"):
            continue
        delta = (old[col] - new[col]).abs().max()
        if delta > 1e-9:
            diffs[col] = float(delta)
    return {"exists": True, "same_columns": True, "same_rows": True,
            "max_diffs": diffs, "identical": not diffs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--date", default=datetime.now(timezone.utc)
                    .date().strftime("%Y%m%d"))
    args = ap.parse_args()

    runs = load_runs()
    per_mask = per_mask_table(runs)
    scenario = seed_level_scenario(runs)
    d_2x, s_2x = paired_diff(runs, "H2X", "H2")
    d_21, s_21 = paired_diff(runs, "H2", "H1")
    table = build_frozen_table(per_mask)
    cmp = compare_with_existing(table, OUT_TABLE)

    print(f"runs validated            : {len(runs)} "
          f"({len(MULTISEED_FILES)} models x 5 seeds x {len(KEY_MASKS)} masks)")
    print(f"per-mask rows             : {len(per_mask)}")
    print(f"paired cells H2X-H2, H2-H1: {len(d_2x)}, {len(d_21)}")
    print(f"existing {OUT_TABLE.name:<28}: {cmp}")
    print()
    cols = ["model", "scenario", "mae_mean", "mae_sd_seeds", "r2_mean",
            "r2_sd_seeds"]
    print("scenario means (5 training seeds):")
    print(scenario[cols].round(6).to_string(index=False))

    if args.check_only:
        print("\n--check-only: nothing written")
        return

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(ANALYSIS / f"multiseed_per_run_{args.date}.csv", index=False)
    per_mask.to_csv(ANALYSIS / f"multiseed_per_mask_{args.date}.csv", index=False)
    scenario.to_csv(ANALYSIS / f"multiseed_summary_{args.date}.csv", index=False)
    d_2x.to_csv(ANALYSIS / f"multiseed_paired_H2X_H2_{args.date}.csv", index=False)
    d_21.to_csv(ANALYSIS / f"multiseed_paired_H2_H1_{args.date}.csv", index=False)
    pd.concat([s_2x, s_21]).to_csv(
        ANALYSIS / f"multiseed_paired_summary_{args.date}.csv", index=False)

    if cmp.get("exists") and not cmp.get("identical", False):
        backup = OUT_TABLE.with_name("benchmark_multiseed_pre20260912.csv")
        if not backup.exists():
            shutil.copy2(OUT_TABLE, backup)
            print(f"backup of previous table  : {backup}")
    table.to_csv(OUT_TABLE, index=False)
    print(f"wrote                     : {OUT_TABLE} ({len(table)} rows)")

    for name in sorted(ANALYSIS.glob(f"multiseed_*_{args.date}.csv")):
        print(f"wrote                     : {name}")


if __name__ == "__main__":
    main()
