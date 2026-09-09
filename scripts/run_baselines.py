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
    for mname, model in models.items():
        # one mask at a time, merging after each: crash-safe, resumable
        for mask_name in names:
            jpath = RESULTS / f"{mname}.json"
            merged = json.loads(jpath.read_text()) if jpath.exists() else {}
            if mask_name in merged:
                continue
            res = evaluate(model, dataset, [mask_name], mask_dir)
            merged.update(res)
            jpath.write_text(json.dumps(merged, indent=2))
            m = res[mask_name]
            print(f"{mname} @ {mask_name}: RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f} R2={m['r2']:.3f} (n={m['n']})", flush=True)
    # rebuild the flat CSV from all per-model JSONs
    rows = []
    for jpath in sorted(RESULTS.glob("B[0-9]_*.json")):
        for mask_name, m in json.loads(jpath.read_text()).items():
            rows.append({"model": jpath.stem, "mask": mask_name, **m})
    pd.DataFrame(rows).to_csv(RESULTS / "baselines.csv", index=False)
    print(f"saved {RESULTS / 'baselines.csv'}")


if __name__ == "__main__":
    main()
