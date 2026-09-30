"""Calibrate the DOC fusion gate on the support-matched spatial validation block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_fusion_confirmation import fit_gate, selected_alpha, sha256_file
from run_unified_gate_pilot import encode, feature_frame, station_bootstrap_gain

SEEDS = (42, 43, 44, 45, 46)
MASK = "e3_spatial_validation_supportmatched"
BASE = Path("experiments/phase4_transfer/kgml_local_transport_v1")
PILOT = BASE / "spatial_validation_e3_supportmatched" / "pilot"
RF = BASE / "unified_fusion_spatial_validation_rf"
RESIDUAL = BASE / "unified_fusion_spatial_validation_residual"


def run_path(arm: str, seed: int) -> Path:
    root = RESIDUAL if arm == "residual_nomsg" else (PILOT if seed <= 44 else RF)
    matches = []
    for meta_path in (root / "runs").glob("*/meta.json"):
        cfg = json.loads(meta_path.read_text())["config"]
        if cfg["mask"] == MASK and cfg["seed"] == seed and cfg["arm"] == arm:
            matches.append(meta_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one {arm} seed {seed}, found {len(matches)}")
    return matches[0]


def load(role: str) -> dict[str, pd.DataFrame]:
    out = {}
    for arm in ("rf_context", "residual_nomsg"):
        pieces = []
        for seed in SEEDS:
            path = run_path(arm, seed) / f"{role}_predictions.parquet"
            frame = pd.read_parquet(path)
            pieces.append(frame[["cell", "station", "month_index", "y_true", "final_pred",
                                 "age_group", "upstream_support_group", "flow", "flow_visible"]])
        merged = pd.concat(pieces, ignore_index=True)
        merged["abs_error"] = np.abs(merged.y_true - merged.final_pred)
        out[arm] = merged.groupby(
            ["cell", "station", "month_index", "y_true", "age_group",
             "upstream_support_group", "flow", "flow_visible"], as_index=False
        ).agg(final_pred=("final_pred", "mean"), abs_error=("abs_error", "mean"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val, test = load("val"), load("test")
    val_features, _ = feature_frame(val, MASK)
    test_features, _ = feature_frame(test, MASK)
    x_val, x_test = encode(val_features, test_features, "disagreement")
    yv = val["rf_context"].y_true.to_numpy()
    rv = val["rf_context"].final_pred.to_numpy()
    lv = val["residual_nomsg"].final_pred.to_numpy()
    yt = test["rf_context"].y_true.to_numpy()
    rt = test["rf_context"].final_pred.to_numpy()
    lt = test["residual_nomsg"].final_pred.to_numpy()
    alpha0 = selected_alpha(yv, rv, lv)
    model = fit_gate(x_val, rv, lv, yv, alpha0, alpha_floor=0.01)
    with torch.no_grad():
        alpha, fused_z = model(torch.as_tensor(np.log1p(np.clip(rt, 0, None)), dtype=torch.float32),
                               torch.as_tensor(np.log1p(np.clip(lt, 0, None)), dtype=torch.float32),
                               torch.as_tensor(x_test, dtype=torch.float32))
    fused = np.expm1(fused_z.numpy())
    gain, lo, hi = station_bootstrap_gain(
        yt, fused, rt, test["rf_context"].station.to_numpy(), seed=42)
    result = pd.DataFrame([{
        "mask": MASK, "n_val": len(yv), "n_test": len(yt), "alpha_init": alpha0,
        "alpha_test_mean": float(alpha.mean()), "alpha_test_sd": float(alpha.std()),
        "rf_mae": float(np.abs(yt - rt).mean()),
        "residual_mae": float(np.abs(yt - lt).mean()),
        "fusion_mae": float(np.abs(yt - fused).mean()),
        "gain_vs_rf": gain, "ci_lo": lo, "ci_hi": hi,
    }])
    result.to_csv(args.out_dir / "spatial_gate_summary.csv", index=False)
    inputs = []
    for arm in ("rf_context", "residual_nomsg"):
        for seed in SEEDS:
            run = run_path(arm, seed)
            for role in ("val", "test"):
                path = run / f"{role}_predictions.parquet"
                inputs.append({"arm": arm, "seed": seed, "role": role,
                               "path": str(path), "sha256": sha256_file(path)})
    (args.out_dir / "spatial_gate_manifest.json").write_text(json.dumps({
        "mask": MASK, "seeds": list(SEEDS), "feature_set": "disagreement",
        "alpha_floor": 0.01, "inputs": inputs,
        "code_sha256": {p: sha256_file(Path(p)) for p in
                        ("scripts/run_unified_fusion_spatial_check.py",
                         "scripts/run_unified_fusion_confirmation.py",
                         "scripts/run_unified_gate_pilot.py",
                         "src/river_graph/models/unified_fusion.py")},
    }, indent=2) + "\n")
    (args.out_dir / "spatial_gate_verdict.md").write_text(
        "# Support-matched spatial fusion check\n\n"
        "The E3 gate was fitted on the support-matched spatial validation block and evaluated on the same frozen E3 terminal test stations.\n\n"
        + result.to_string(index=False) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
