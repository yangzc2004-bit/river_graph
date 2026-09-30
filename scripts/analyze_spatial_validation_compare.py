"""Compare spatial residual arms on a spatial validation block and frozen E3 test.

The validation block is for model-selection diagnostics. The original E3 test
stations are reported separately and are never used to choose an arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("rf_context", "residual_context_nomsg", "residual_context_msgdelta",
        "residual_context_both")
SEEDS = (42, 43, 44)


def station_bootstrap(delta: pd.Series, stations: pd.Series, seed: int = 42,
                      reps: int = 5000) -> tuple[float, float, float]:
    grouped = pd.DataFrame({"station": stations, "delta": delta}).groupby("station").delta.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    # The numerator and denominator must use the same resampled stations.
    idx = rng.integers(len(sums), size=(reps, len(sums)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    return float(delta.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def load(root: Path) -> dict[tuple[str, int], dict[str, pd.DataFrame]]:
    out = {}
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["arm"] not in ARMS:
            continue
        run = meta_path.parent
        frames = {role: pd.read_parquet(run / f"{role}_predictions.parquet") for role in ("val", "test")}
        for role, frame in frames.items():
            required = {"cell", "station", "y_true", "final_pred", "upstream_support_group"}
            if not required <= set(frame):
                raise ValueError(f"missing columns in {run}/{role}")
            if frame.cell.duplicated().any() or not np.isfinite(frame[["y_true", "final_pred"]]).all().all():
                raise ValueError(f"invalid predictions in {run}/{role}")
            if not frame.visibility_role.eq(role).all() or frame.visible_input.any():
                raise ValueError(f"visibility mismatch in {run}/{role}")
        if cfg["arm"] == "residual_context_nomsg":
            np.testing.assert_array_equal(frames["test"].graph_delta.to_numpy(), 0)
            np.testing.assert_array_equal(frames["val"].graph_delta.to_numpy(), 0)
            np.testing.assert_allclose(frames["test"].final_pred, frames["test"].context_pred, rtol=0, atol=0)
            np.testing.assert_allclose(frames["val"].final_pred, frames["val"].context_pred, rtol=0, atol=0)
        out[(cfg["arm"], cfg["seed"])] = frames
    if set(out) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("incomplete spatial comparison matrix")
    return out


def seedmean(frames, role: str) -> dict[str, pd.DataFrame]:
    pooled = {}
    for arm in ARMS:
        parts = []
        for seed in SEEDS:
            frame = frames[(arm, seed)][role]
            parts.append(frame[["cell", "station", "y_true", "final_pred", "upstream_support_group"]].assign(
                abs_error=np.abs(frame.y_true - frame.final_pred)))
        merged = pd.concat(parts, ignore_index=True)
        pooled[arm] = merged.groupby(
            ["cell", "station", "y_true", "upstream_support_group"], as_index=False
        ).agg(abs_error=("abs_error", "mean"))
    return pooled


def summarize(pooled, role: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, boot, support = [], [], []
    base = pooled["rf_context"]
    for arm in ARMS:
        frame = pooled[arm]
        rows.append({"role": role, "arm": arm, "n_cells": len(frame),
                     "n_stations": frame.station.nunique(), "mae": float(frame.abs_error.mean())})
    for arm in ("residual_context_msgdelta", "residual_context_both"):
        candidate = pooled[arm]
        merged = base.merge(candidate, on=["cell", "station", "y_true", "upstream_support_group"],
                            suffixes=("_base", "_candidate"), validate="one_to_one")
        delta = merged.abs_error_base - merged.abs_error_candidate
        mean, low, high = station_bootstrap(delta, merged.station, seed=42 if role == "val" else 43)
        boot.append({"role": role, "candidate": arm, "reference": "rf_context",
                     "mae_candidate": float(merged.abs_error_candidate.mean()),
                     "mae_reference": float(merged.abs_error_base.mean()),
                     "gain_mae": mean, "gain_pct": 100 * mean / merged.abs_error_base.mean(),
                     "ci_low": low, "ci_high": high})
    for group in sorted(base.upstream_support_group.unique()):
        for arm in ARMS:
            frame = pooled[arm]
            frame = frame[frame.upstream_support_group.eq(group)]
            support.append({"role": role, "support_group": group, "arm": arm,
                            "n_cells": len(frame), "n_stations": frame.station.nunique(),
                            "mae": float(frame.abs_error.mean())})
    return pd.DataFrame(rows), pd.DataFrame(boot), pd.DataFrame(support)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    frames = load(args.root)
    tables = []
    for role in ("val", "test"):
        summary, bootstrap, support = summarize(seedmean(frames, role), role)
        summary.to_csv(args.root / f"spatial_{role}_summary.csv", index=False)
        bootstrap.to_csv(args.root / f"spatial_{role}_bootstrap.csv", index=False)
        support.to_csv(args.root / f"spatial_{role}_support.csv", index=False)
        tables.extend((summary, bootstrap, support))
    report = ["# Spatial validation comparison", "",
              "The validation block is used for arm selection; the original E3 test is terminal.", ""]
    for role in ("val", "test"):
        report += [f"## {role}", ""]
        for suffix, title in (("summary", "Mean MAE"), ("bootstrap", "Paired station bootstrap"),
                              ("support", "Upstream support strata")):
            report += [f"### {title}", "", pd.read_csv(args.root / f"spatial_{role}_{suffix}.csv").to_string(index=False), ""]
    (args.root / "spatial_validation_comparison.md").write_text("\n".join(report))
    print(pd.concat([tables[0], tables[3]], ignore_index=True).to_string(index=False))
    print(pd.concat([tables[1], tables[4]], ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
