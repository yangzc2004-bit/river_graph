"""Compare same-month, lagged-river, and no-message DOC attention arms.

The script only reads completed metrics.csv files and writes a new analysis
directory. It keeps the seed-level paired comparison visible: the mean of
seed metrics is the compact summary, while paired differences show whether
the lagged arm gains over the same-month arm and the no-message control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ("mae", "rmse", "r2", "q90_mae")
SEEDS = (42, 43, 44)
ARMS = {
    "same_month_upstream": ("same_month", "residual_attention"),
    "river_lag": ("river_lag", "residual_attention_river"),
    "no_message": ("river_lag", "residual_attention_river_nomsg"),
}


def read_arm(root: Path, arm: str, analyte: str, mask: str) -> pd.DataFrame:
    path = root / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics file is not ready: {path}")
    frame = pd.read_csv(path)
    required = {"arm", "analyte", "mask", "seed", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = frame[
        (frame["arm"] == arm)
        & (frame["analyte"] == analyte)
        & (frame["mask"] == mask)
    ].copy()
    selected["seed"] = selected["seed"].astype(int)
    selected = selected[selected["seed"].isin(SEEDS)].copy()
    if set(selected["seed"]) != set(SEEDS) or len(selected) != len(SEEDS):
        found = sorted(selected["seed"].unique().tolist())
        raise ValueError(f"{path}: {arm} has seeds {found}; expected {list(SEEDS)}")
    if selected["seed"].duplicated().any():
        raise ValueError(f"{path}: duplicate seed rows for {arm}")
    return selected[["seed", *METRICS]].rename(
        columns={metric: f"{arm}__{metric}" for metric in METRICS}
    ).sort_values("seed").reset_index(drop=True)


def paired_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide[["seed"]].copy()
    for metric in METRICS:
        same = wide[f"residual_attention__{metric}"]
        lag = wide[f"residual_attention_river__{metric}"]
        null = wide[f"residual_attention_river_nomsg__{metric}"]
        out[f"lag_minus_same__{metric}"] = lag - same
        out[f"lag_minus_null__{metric}"] = lag - null
        out[f"same_minus_null__{metric}"] = same - null
    same_mae = wide["residual_attention__mae"]
    lag_mae = wide["residual_attention_river__mae"]
    null_mae = wide["residual_attention_river_nomsg__mae"]
    out["lag_gain_vs_same_percent"] = 100 * (same_mae - lag_mae) / same_mae
    out["lag_gain_vs_null_percent"] = 100 * (null_mae - lag_mae) / null_mae
    out["same_gain_vs_null_percent"] = 100 * (null_mae - same_mae) / null_mae
    return out


def summary_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "residual_attention": "same_month_upstream",
        "residual_attention_river": "river_lag",
        "residual_attention_river_nomsg": "no_message",
    }
    for arm, label in labels.items():
        row = {"arm": label, "n_seeds": len(wide)}
        for metric in METRICS:
            values = wide[f"{arm}__{metric}"].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd_seed"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def render_verdict(summary: pd.DataFrame, deltas: pd.DataFrame, *, analyte: str, mask: str) -> str:
    by_arm = summary.set_index("arm")
    lag_same = deltas["lag_gain_vs_same_percent"]
    lag_null = deltas["lag_gain_vs_null_percent"]
    same_null = deltas["same_gain_vs_null_percent"]
    return f"""# Three-arm attention comparison

Analyte: {analyte}  
Mask: {mask}  
Seeds: {", ".join(str(x) for x in SEEDS)}

The comparison uses the same query mask and seed identities for all three
arms. The compact estimate is the unweighted mean of the three seed metrics;
three_arm_paired_deltas.csv retains the per-seed paired differences.

| arm | MAE mean | RMSE mean | R2 mean | Q90 MAE mean |
| --- | ---: | ---: | ---: | ---: |
| same-month upstream | {by_arm.loc["same_month_upstream", "mae_mean"]:.6g} | {by_arm.loc["same_month_upstream", "rmse_mean"]:.6g} | {by_arm.loc["same_month_upstream", "r2_mean"]:.6g} | {by_arm.loc["same_month_upstream", "q90_mae_mean"]:.6g} |
| same-month plus river lag | {by_arm.loc["river_lag", "mae_mean"]:.6g} | {by_arm.loc["river_lag", "rmse_mean"]:.6g} | {by_arm.loc["river_lag", "r2_mean"]:.6g} | {by_arm.loc["river_lag", "q90_mae_mean"]:.6g} |
| no-message control | {by_arm.loc["no_message", "mae_mean"]:.6g} | {by_arm.loc["no_message", "rmse_mean"]:.6g} | {by_arm.loc["no_message", "r2_mean"]:.6g} | {by_arm.loc["no_message", "q90_mae_mean"]:.6g} |

Mean MAE gain of river lag versus same-month upstream: **{lag_same.mean():.3f}%**.

Mean MAE gain of river lag versus no-message: **{lag_null.mean():.3f}%**.

Mean MAE gain of same-month upstream versus no-message: **{same_null.mean():.3f}%**.

Positive gain means lower MAE. These are paired seed summaries; inspect seed
directions, query counts, and the run provenance before assigning a transport
interpretation. Q90 values with fewer than 20 test cells are descriptive only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-month-dir", type=Path, required=True)
    parser.add_argument("--river-lag-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analyte", default="doc")
    parser.add_argument("--mask", default="e2a_strict")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for source, arm in ARMS.values():
        root = args.same_month_dir if source == "same_month" else args.river_lag_dir
        frames.append(read_arm(root, arm, args.analyte, args.mask))
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="seed", how="inner", validate="one_to_one")
    if len(wide) != len(SEEDS):
        raise ValueError("seed alignment failed")
    deltas = paired_deltas(wide)
    summary = summary_table(wide)
    wide.to_csv(args.out_dir / "three_arm_by_seed.csv", index=False)
    deltas.to_csv(args.out_dir / "three_arm_paired_deltas.csv", index=False)
    summary.to_csv(args.out_dir / "three_arm_means.csv", index=False)
    (args.out_dir / "verdict.md").write_text(
        render_verdict(summary, deltas, analyte=args.analyte, mask=args.mask)
    )
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
