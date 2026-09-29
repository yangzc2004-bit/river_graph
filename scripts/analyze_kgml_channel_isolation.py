"""Compare target-only and hydro-ecology message channels with K2 all-input."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.analyze_kgml_source_isolation import (
    load_runs,
    paired_predictions,
    station_bootstrap,
)


def audit(root: Path, all_root: Path) -> pd.DataFrame:
    all_runs = load_runs(all_root)
    all_cells = {
        mask: paired_predictions(all_runs, mask)[
            ["cell", "station", "month", "y_true", "residual_msgdelta_error"]
        ].rename(columns={"residual_msgdelta_error": "all_error"})
        for mask in sorted(all_runs["mask"].unique())
    }
    rows = []
    for mode in ("target_only", "hydro_ecology"):
        mode_root = root / mode
        runs = load_runs(mode_root)
        for mask in sorted(runs["mask"].unique()):
            paired = paired_predictions(runs, mask)
            reference = all_cells[mask]
            merged = paired.merge(reference, on=["cell", "station", "month", "y_true"], validate="one_to_one")
            for comparison, values in (
                ("mode_vs_null", merged.residual_msgnull_error - merged.residual_msgdelta_error),
                ("mode_vs_all", merged.all_error - merged.residual_msgdelta_error),
            ):
                frame = merged[["station"]].copy()
                frame["gain"] = values
                point, low, high = station_bootstrap(frame.gain, frame.station)
                rows.append(
                    {
                        "mode": mode,
                        "mask": mask,
                        "comparison": comparison,
                        "n_cells": len(merged),
                        "n_stations": merged.station.nunique(),
                        "message_mae": float(merged.residual_msgdelta_error.mean()),
                        "null_mae": float(merged.residual_msgnull_error.mean()),
                        "all_input_mae": float(merged.all_error.mean()),
                        "gain_mae": point,
                        "ci_low": low,
                        "ci_high": high,
                        "gain_pct_vs_null": 100 * point / merged.residual_msgnull_error.mean(),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 8:
        raise ValueError(f"expected eight channel comparisons, found {len(result)}")
    result.to_csv(root / "channel_gain_summary.csv", index=False)
    report = [
        "# KGML channel-isolation verdict",
        "",
        "Positive gain means the candidate message channel has lower error.",
        "`mode_vs_null` compares with the exact zero-message null; `mode_vs_all` compares with K2's all-input message branch.",
        "",
        result.to_string(index=False),
        "",
        "The null comparison identifies information retained by a channel set. The all-input comparison identifies information lost when the message input is restricted.",
        "",
    ]
    (root / "channel_gain_verdict.md").write_text("\n".join(report))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k4_channel_isolation"),
    )
    parser.add_argument(
        "--all-root",
        type=Path,
        default=Path("experiments/phase4_transfer/kgml_local_transport_v1/k2_source_isolation_v3"),
    )
    args = parser.parse_args()
    print(audit(args.root, args.all_root).to_string(index=False))


if __name__ == "__main__":
    main()
