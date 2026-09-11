"""Recompute the frozen benchmark table from stored predictions.

Zero retraining: reads experiments/predictions/*.parquet and rebuilds
experiments/frozen_results/benchmark.csv with the current metric set.

Usage: python scripts/recompute_metrics.py
"""

from pathlib import Path

import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.predictions import PRED_DIR


def main() -> None:
    rows = []
    for f in sorted(PRED_DIR.glob("*.parquet")):
        model, mask = f.stem.split("__")
        df = pd.read_parquet(f)
        test = df[df["split"] == "test"]
        m = metrics(test["y_true"].values, test["y_pred"].values)
        rows.append({"model": model, "mask": mask, **m})
    out = Path("experiments/frozen_results/benchmark.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"recomputed {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
