"""Run baselines B0-B3 over benchmark masks -> results CSV.

    python scripts/run_baselines.py                 # all masks
    python scripts/run_baselines.py --only e1       # subset by name prefix
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from river_graph.baselines.baselines import MLP, Kriging, RandomForest, StationMean
from river_graph.experiments.evaluate import evaluate, load_dataset

RESULTS = Path("experiments/results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="mask-name prefix filter")
    args = ap.parse_args()

    dataset = load_dataset()
    mask_dir = Path("experiments/masks")
    names = sorted(p.stem for p in mask_dir.glob("*.npz"))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    print(f"masks: {names}")

    models = {
        "B0_station_mean": StationMean(),
        "B1_kriging": Kriging(),
        "B2_random_forest": RandomForest(),
        "B3_mlp": MLP(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, model in models.items():
        res = evaluate(model, dataset, names, mask_dir)
        (RESULTS / f"{mname}.json").write_text(json.dumps(res, indent=2))
        for mask_name, m in res.items():
            rows.append({"model": mname, "mask": mask_name, **m})
            print(f"{mname} @ {mask_name}: RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f} R2={m['r2']:.3f} (n={m['n']})")
    pd.DataFrame(rows).to_csv(RESULTS / "baselines.csv", index=False)
    print(f"saved {RESULTS / 'baselines.csv'}")


if __name__ == "__main__":
    main()
