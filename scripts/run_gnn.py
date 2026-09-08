"""Run the GCN model (topology variants) over benchmark masks.

    python scripts/run_gnn.py --only e1_r20_seed42 --variants river
    python scripts/run_gnn.py                       # all masks x variants
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from river_graph.experiments.evaluate import evaluate, load_dataset
from river_graph.models.gcn import GCNDocModel

RESULTS = Path("experiments/results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="mask-name prefix filter")
    ap.add_argument("--variants", nargs="+", default=["river", "random", "none"])
    args = ap.parse_args()

    dataset = load_dataset()
    mask_dir = Path("experiments/masks")
    names = sorted(p.stem for p in mask_dir.glob("*.npz"))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    print(f"masks: {names}; variants: {args.variants}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in args.variants:
        mname = f"G0_gcn_{variant}"
        model = GCNDocModel(variant=variant)
        res = evaluate(model, dataset, names, mask_dir)
        (RESULTS / f"{mname}.json").write_text(json.dumps(res, indent=2))
        for mask_name, m in res.items():
            rows.append({"model": mname, "mask": mask_name, **m})
            print(f"{mname} @ {mask_name}: RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f} R2={m['r2']:.3f} (n={m['n']})", flush=True)
    out = RESULTS / "gnn.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
