"""Phase 0: freeze the benchmark.

Trains every paper model on every mask ONCE, saves per-cell predictions to
experiments/predictions/*.parquet, and writes the frozen results table to
experiments/frozen_results/benchmark.csv (+ metadata). Downstream analysis
must read the frozen files, never retrain.

Resumable: existing prediction files are skipped.

Usage: python scripts/run_freeze.py [--only-substring]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from river_graph.baselines.baselines import MLP, Kriging, RandomForest, StationMean
from river_graph.experiments.evaluate import load_dataset, load_mask, metrics
from river_graph.experiments.predictions import PRED_DIR, save_predictions
from river_graph.models.gcn import GCNDocModel

FROZEN = Path("experiments/frozen_results")
V02 = "data/processed/mississippi_graph_v02.pt"
V03 = "data/processed/mississippi_graph_v03.pt"
V04 = "data/processed/mississippi_graph_v04.pt"


def model_zoo() -> list[tuple[str, str, object]]:
    """(name, dataset_path, model) in paper order."""
    return [
        ("B0_station_mean", V02, StationMean()),
        ("B1_kriging", V02, Kriging()),
        ("B2_random_forest", V02, RandomForest()),
        ("B3_mlp", V02, MLP()),
        ("G0_gcn_none", V02, GCNDocModel(variant="none")),
        ("G0_gcn_random", V02, GCNDocModel(variant="random")),
        ("G0_gcn_river", V02, GCNDocModel(variant="river")),
        ("H1_directed_river", V02, GCNDocModel(variant="river", architecture="directed")),
        ("H2_transport_river", V03, GCNDocModel(variant="river", architecture="transport")),
        ("H2E_transport_river", V04, GCNDocModel(variant="river", architecture="transport")),
        ("H2X_transport_enc_river", V04, GCNDocModel(variant="river",
                                                     architecture="transport_enc",
                                                     env_encoder=True)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="model-name substring filter")
    args = ap.parse_args()

    FROZEN.mkdir(parents=True, exist_ok=True)
    mask_names = sorted(p.stem for p in Path("experiments/masks").glob("*.npz"))
    datasets = {}

    rows = []
    for mname, dpath, model in model_zoo():
        if args.only and args.only not in mname:
            continue
        if dpath not in datasets:
            datasets[dpath] = load_dataset(dpath)
        ds = datasets[dpath]
        version = Path(dpath).stem
        for mask_name in mask_names:
            out = PRED_DIR / f"{mname}__{mask_name}.parquet"
            if out.exists():
                print(f"skip {mname} @ {mask_name} (frozen)")
                pred_df = pd.read_parquet(out)
                test = pred_df[pred_df["split"] == "test"]
                m = metrics(test["y_true"].values, test["y_pred"].values)
            else:
                split = load_mask(mask_name)
                pred = model.fit_predict(ds, split)
                save_predictions(pred, ds, split, mname, mask_name, version)
                y = ds["y"].numpy()
                m = metrics(y.ravel()[split["test"]], pred.ravel()[split["test"]])
                print(f"frozen {mname} @ {mask_name}: MAE={m['mae']:.3f} "
                      f"R2={m['r2']:.3f} logR2={m['log_r2']:.3f}", flush=True)
            rows.append({"model": mname, "mask": mask_name, **m})

    pd.DataFrame(rows).to_csv(FROZEN / "benchmark.csv", index=False)
    meta = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "masks": mask_names,
        "models": [m[0] for m in model_zoo()],
        "datasets": {"v02": V02, "v03": V03, "v04": V04},
        "note": "Predictions in experiments/predictions/ are the canonical "
                "benchmark artifact. Analysis reads these; no retraining.",
    }
    (FROZEN / "experiment_config.json").write_text(json.dumps(meta, indent=2))
    print(f"frozen: {FROZEN / 'benchmark.csv'} ({len(rows)} model x mask rows)")


if __name__ == "__main__":
    main()
