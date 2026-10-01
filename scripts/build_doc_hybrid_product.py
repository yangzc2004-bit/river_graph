"""Materialize the validation-selected DOC hybrid predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_et_hybrid_affine import _predict_log_affine
from scripts.analyze_et_hybrid_gate import MASKS, merge_experts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.metrics).set_index(["seed", "mask"])
    products: list[pd.DataFrame] = []
    for seed in args.seeds:
        for mask in args.masks:
            key = (seed, mask)
            if key not in detail.index:
                continue
            row = detail.loc[key]
            test = merge_experts(seed, mask, "test")
            if row["selected_variant"] == "log_affine":
                beta = np.array([row["beta0"], row["beta_context"], row["beta_residual"]], dtype=float)
                pred = _predict_log_affine(test, beta)
                route = "validation_selected_log_affine_context_residual"
            else:
                pred = test["context_pred"].to_numpy(dtype=float)
                route = "validation_selected_extra_trees_context"
            out = test[["cell", "y_true", "context_pred"]].copy()
            out["residual_pred"] = test.get("residual_pred", np.nan)
            out["y_pred"] = pred
            out["seed"] = seed
            out["mask"] = mask
            out["model_name"] = "DOC_validation_selected_hybrid"
            out["route"] = route
            out["beta0"] = row.get("beta0", np.nan)
            out["beta_context"] = row.get("beta_context", np.nan)
            out["beta_residual"] = row.get("beta_residual", np.nan)
            path = args.out_dir / f"doc__{mask}__seed{seed}.parquet"
            out.to_parquet(path, index=False)
            products.append(out)
    if not products:
        raise RuntimeError("no complete products found")
    all_products = pd.concat(products, ignore_index=True)
    all_products.to_parquet(args.out_dir / "all_test_predictions.parquet", index=False)
    metrics = all_products.groupby(["seed", "mask"], as_index=False).apply(
        lambda frame: pd.Series({
            "n": len(frame),
            "mae": float(np.abs(frame["y_true"] - frame["y_pred"]).mean()),
            "rmse": float(np.sqrt(np.square(frame["y_true"] - frame["y_pred"]).mean())),
        }), include_groups=False)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "# DOC validation-selected hybrid product\n\n"
        "Temporal products use a log-space affine stack fit on validation "
        "predictions only. Spatial products use the validation-selected "
        "ExtraTrees context expert. Test labels are retained only for the "
        "reported metrics.\n"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
