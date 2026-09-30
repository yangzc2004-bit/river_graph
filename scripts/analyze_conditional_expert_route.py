"""Select one expert per missingness family using validation cells only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("rf_context", "residual_nomsg")
SEEDS = (42, 43, 44)
ROOTS = {
    "e1_r20_seed42": Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
    "e2b_partial": Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
    "e2a_strict": Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
    "e3_spatial_seed42": Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
}


def read_runs(root: Path, mask: str) -> dict[tuple[str, int], dict[str, pd.DataFrame]]:
    result = {}
    for meta_path in (root / "runs").glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        cfg = meta["config"]
        if cfg["mask"] != mask or cfg["arm"] not in ARMS or cfg["seed"] not in SEEDS:
            continue
        run = meta_path.parent
        result[(cfg["arm"], cfg["seed"])] = {
            role: pd.read_parquet(run / f"{role}_predictions.parquet") for role in ("val", "test")
        }
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(result) != expected:
        raise ValueError(f"incomplete route matrix for {mask}: {sorted(set(result) ^ expected)}")
    return result


def pooled(frames, role: str) -> dict[str, pd.DataFrame]:
    result = {}
    for arm in ARMS:
        pieces = []
        for seed in SEEDS:
            frame = frames[(arm, seed)][role]
            pieces.append(frame[["cell", "station", "y_true", "final_pred"]].assign(
                abs_error=np.abs(frame.y_true - frame.final_pred)))
        merged = pd.concat(pieces, ignore_index=True)
        result[arm] = merged.groupby(["cell", "station", "y_true"], as_index=False).abs_error.mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_frames = []
    for mask, root in ROOTS.items():
        frames = read_runs(root, mask)
        val, test = pooled(frames, "val"), pooled(frames, "test")
        val_mae = {arm: float(val[arm].abs_error.mean()) for arm in ARMS}
        chosen = min(ARMS, key=val_mae.get)
        row = {"mask": mask, "chosen_arm": chosen,
               "val_rf_context": val_mae["rf_context"],
               "val_residual_nomsg": val_mae["residual_nomsg"]}
        for arm in ARMS:
            row[f"test_{arm}"] = float(test[arm].abs_error.mean())
        row["test_routed"] = row[f"test_{chosen}"]
        rows.append(row)
        selected_frames.append(test[chosen].assign(mask=mask, chosen_arm=chosen))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "route_summary.csv", index=False)
    routed = pd.concat(selected_frames, ignore_index=True)
    pooled_rows = [{"system": "routed", "mae": float(routed.abs_error.mean()), "n": len(routed)}]
    for arm in ARMS:
        pooled_rows.append({"system": arm, "mae": float(pd.concat(
            [pooled(read_runs(ROOTS[m], m), "test")[arm] for m in ROOTS], ignore_index=True
        ).abs_error.mean()), "n": int(sum(len(pooled(read_runs(ROOTS[m], m), "test")[arm]) for m in ROOTS))})
    pd.DataFrame(pooled_rows).to_csv(args.out_dir / "route_pooled.csv", index=False)
    report = ["# Conditional expert route", "",
              "Each family selects an expert from validation cells only.", "",
              summary.to_string(index=False), "", "## Pooled test", "",
              pd.DataFrame(pooled_rows).to_string(index=False), ""]
    (args.out_dir / "route_verdict.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(pd.DataFrame(pooled_rows).to_string(index=False))


if __name__ == "__main__":
    main()
