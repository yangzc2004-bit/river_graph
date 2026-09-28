"""Recompute U4 blend metrics and paired, station-clustered uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import config_hash, sha256_file
from river_graph.experiments.transfer import DATASETS

ROOT = Path("experiments/phase4_transfer/model_upgrade_v1")


def cluster_interval(errors: pd.DataFrame, repeats: int = 2000) -> tuple[float, float]:
    """Average seeds on identical cells, then resample whole stations."""
    cells = errors.groupby(["station", "cell"], sort=True).delta.mean().reset_index()
    groups = cells.groupby("station").delta.agg(["sum", "count"])
    if len(groups) < 2:
        raise ValueError("At least two stations required for cluster bootstrap")
    rng = np.random.default_rng(42)
    draws = rng.integers(len(groups), size=(repeats, len(groups)))
    sums, counts = groups["sum"].to_numpy(), groups["count"].to_numpy()
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def analyze(root: Path) -> pd.DataFrame:
    rows, all_errors, hashes = [], [], {}
    for path in sorted((root / "runs").glob("*/meta.json")):
        meta = json.loads(path.read_text())
        config = meta["config"]
        pred_path = path.parent / "test_predictions.parquet"
        if sha256_file(pred_path) != meta["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {path}")
        if config_hash(config, version=3) != config["config_hash"]:
            raise ValueError(f"V3 hash mismatch: {path}")
        protocol_name = {
            "e2a_strict": (
                "u4_blend_protocol.json" if config["seed"] == 42
                else "u4_blend_confirm_protocol.json"
            ),
            "e3_spatial_seed42": "u5_spatial_blend_protocol.json",
        }[config["mask"]]
        protocol = ROOT / protocol_name
        if sha256_file(protocol) != config["protocol_sha256"]:
            raise ValueError(f"Protocol hash mismatch: {path}")
        ds_path = DATASETS[config["analyte"]]
        if sha256_file(ds_path) != config["dataset_sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
        mask_path = Path("experiments/phase4_transfer/temporal_h2x_v1/target_masks") / (
            f"{config['analyte']}__{config['mask']}.npz")
        if sha256_file(mask_path) != config["mask_sha256"]:
            raise ValueError(f"Mask hash mismatch: {path}")
        ds = torch.load(ds_path, map_location="cpu", weights_only=False)
        frame = pd.read_parquet(pred_path)
        cells = frame.cell.to_numpy(dtype=int)
        with np.load(mask_path) as mask:
            np.testing.assert_array_equal(cells, mask["test"])
        if len(np.unique(cells)) != len(cells):
            raise ValueError("Duplicate test cells")
        np.testing.assert_array_equal(frame.y_true, ds["y"].numpy().ravel()[cells])
        if not np.isfinite(frame[["y_true", "rf_pred", "gnn_pred", "blend_pred"]]).all().all():
            raise ValueError("Incomplete or nonfinite predictions")
        alpha = meta["alpha_rf"]
        grid = meta["validation_grid"]
        selected = min(grid, key=lambda x: (x["val_mae"], -x["alpha_rf"]))
        if selected["alpha_rf"] != alpha:
            raise ValueError("Alpha does not match stored validation scores")
        np.testing.assert_allclose(frame.blend_pred,
                                   alpha * frame.rf_pred + (1 - alpha) * frame.gnn_pred)
        row = {"analyte": config["analyte"], "mask": config["mask"],
               "seed": config["seed"], "alpha_rf": alpha, "n": len(frame)}
        for column, name in (("rf_pred", "temporal_rf"), ("gnn_pred", "h2x_t"),
                             ("blend_pred", "blend")):
            result = metrics(frame.y_true.to_numpy(), frame[column].to_numpy())
            for key in ("mae", "rmse", "r2"):
                np.testing.assert_allclose(result[key], meta["metrics"][name][key])
                row[f"{name}_{key}"] = result[key]
        row["relative_mae_reduction_pct"] = 100 * (
            1 - row["blend_mae"] / row["temporal_rf_mae"])
        rows.append(row)
        all_errors.append(pd.DataFrame({
            "analyte": config["analyte"], "mask": config["mask"],
            "seed": config["seed"], "cell": cells,
            "station": cells // ds["y"].shape[1],
            "delta": abs(frame.y_true - frame.rf_pred) - abs(frame.y_true - frame.blend_pred),
        }))
        hashes[str(path)] = sha256_file(path)
        hashes[str(pred_path)] = sha256_file(pred_path)
    if not rows:
        raise ValueError("No completed runs")
    runs = pd.DataFrame(rows)
    if runs.duplicated(["analyte", "mask", "seed"]).any():
        raise ValueError("Duplicate run identities")
    errors = pd.concat(all_errors, ignore_index=True)
    summary = []
    for (analyte, mask), group in runs.groupby(["analyte", "mask"]):
        subset = errors[(errors.analyte == analyte) & (errors["mask"] == mask)]
        # Replicated seeds must evaluate the same query cells.
        if not subset.groupby("cell").seed.nunique().eq(len(group)).all():
            raise ValueError("Unmatched query across seeds")
        lo, hi = cluster_interval(subset)
        rf, blend = group.temporal_rf_mae.mean(), group.blend_mae.mean()
        summary.append({
            "analyte": analyte, "mask": mask, "completed_seeds": len(group),
            "rf_mae": rf, "gnn_mae": group.h2x_t_mae.mean(), "blend_mae": blend,
            "relative_mae_reduction_pct": 100 * (1 - blend / rf),
            "positive_seeds": int((group.relative_mae_reduction_pct > 0).sum()),
            "paired_rf_minus_blend_mae": rf - blend, "ci_low": lo, "ci_high": hi,
        })
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    complete = len(runs) == 6 and set(runs.seed) == {42, 43, 44}
    audit = {
        "complete": complete, "completed_runs": len(runs), "expected_runs": 6,
        "artifact_hashes": hashes,
        "bootstrap": "2000 station draws; seeds averaged on identical cells; RF MAE minus blend MAE",
        "limitations": [
            "Seed 42 was the feasibility pilot; seeds 43/44 repeat the same query cells, not independent ecological replication.",
            "Bootstrap was specified after seed-42 results; descriptive CI, not a new confirmatory gate.",
            "Saved validation score grid reproduces weight selection, but validation predictions were not exported.",
            "Legacy V3 config hash omits blend-specific parameters and the generic runtime hash omits the U4/RF executor scripts; this is a partial provenance check.",
            "Metadata started_at was recorded at export, not actual fit start.",
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "u4_blend_pilot")
    args = parser.parse_args()
    print(analyze(args.root).to_string(index=False))
