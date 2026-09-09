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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", default="gcn", choices=["gcn", "directed", "transport"],
                    help="encoder: plain GCN (G0), directed relational (H1), "
                         "or transport-gated (H2)")
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v02.pt",
                    help="dataset .pt (v03 adds edge_attr/regime for H2)")
    ap.add_argument("--share-weights", action="store_true",
                    help="H1.5: share relation weights + direction embedding")
    ap.add_argument("--edge-dropout", type=float, default=0.0,
                    help="H1.5: per-edge dropout prob during training")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    mask_dir = Path("experiments/masks")
    names = sorted(p.stem for p in mask_dir.glob("*.npz"))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    print(f"masks: {names}; variants: {args.variants}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        if args.arch == "gcn":
            prefix = "G0_gcn"
        elif args.arch == "transport":
            prefix = "H2_transport"
        elif args.share_weights or args.edge_dropout or args.wd:
            prefix = "H15_directed"
        else:
            prefix = "H1_directed"
        mname = f"{prefix}_{variant}"
        model = GCNDocModel(variant=variant, lr=args.lr, architecture=args.arch,
                            share_weights=args.share_weights,
                            edge_dropout=args.edge_dropout, weight_decay=args.wd)
        # one mask at a time, merging after each: crash-safe long runs
        for mask_name in names:
            jpath = RESULTS / f"{mname}.json"
            merged = json.loads(jpath.read_text()) if jpath.exists() else {}
            if mask_name in merged:
                print(f"{mname} @ {mask_name}: cached, skip")
                continue
            res = evaluate(model, dataset, [mask_name], mask_dir)
            merged.update(res)
            jpath.write_text(json.dumps(merged, indent=2))
            m = res[mask_name]
            print(f"{mname} @ {mask_name}: RMSE={m['rmse']:.3f} "
                  f"MAE={m['mae']:.3f} R2={m['r2']:.3f} (n={m['n']})", flush=True)
    # rebuild the flat CSV from all per-model JSONs (never overwrite blindly)
    rows = []
    for jpath in sorted(RESULTS.glob("G*_*.json")) + sorted(RESULTS.glob("H*_*.json")):
        for mask_name, m in json.loads(jpath.read_text()).items():
            rows.append({"model": jpath.stem, "mask": mask_name, **m})
    out = RESULTS / "gnn.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
