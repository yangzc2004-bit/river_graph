"""Five-seed DOC confirmation for conditional and global expert fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_gate_pilot import (
    ARMS,
    ROOTS,
    blend,
    encode,
    feature_frame,
    station_bootstrap_gain,
)

from river_graph.models.unified_fusion import UnifiedSpatiotemporalFusion

SEEDS = (42, 43, 44, 45, 46)
BASE = Path("experiments/phase4_transfer/kgml_local_transport_v1")
CONFIRM_RF = BASE / "conditional_expert_confirmation_rf"
CONFIRM_RESIDUAL = BASE / "conditional_expert_confirmation_residual"
COMPLETION_RF = BASE / "unified_fusion_expert_completion_rf"
COMPLETION_RESIDUAL = BASE / "unified_fusion_expert_completion_residual"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_roots(mask: str, arm: str, seed: int) -> list[Path]:
    roots = []
    if seed <= 44:
        roots.append(ROOTS[mask])
    elif arm == "rf_context":
        roots.append(CONFIRM_RF if mask in {"e1_r20_seed42", "e3_spatial_seed42"} else COMPLETION_RF)
    else:
        roots.append(CONFIRM_RESIDUAL if mask in {"e2a_strict", "e2b_partial"} else COMPLETION_RESIDUAL)
    return roots


def find_run(mask: str, arm: str, seed: int) -> Path:
    matches = []
    for root in candidate_roots(mask, arm, seed):
        for meta_path in (root / "runs").glob("*/meta.json"):
            meta = json.loads(meta_path.read_text())
            cfg = meta["config"]
            if cfg["mask"] == mask and cfg["arm"] == arm and cfg["seed"] == seed:
                matches.append(meta_path.parent)
    if len(matches) != 1:
        raise ValueError(f"expected one run for {mask}/{arm}/seed{seed}, found {len(matches)}")
    return matches[0]


def load_mask_five(mask: str, role: str) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    frames = {}
    provenance = []
    for arm in ARMS:
        pieces = []
        for seed in SEEDS:
            run = find_run(mask, arm, seed)
            meta = json.loads((run / "meta.json").read_text())
            path = run / f"{role}_predictions.parquet"
            pieces.append(pd.read_parquet(path)[[
                "cell", "station", "month_index", "y_true", "final_pred",
                "age_group", "upstream_support_group", "flow", "flow_visible",
            ]])
            provenance.append({"mask": mask, "arm": arm, "seed": seed,
                               "role": role, "path": str(path),
                               "sha256": sha256_file(path),
                               "run_meta_sha256": sha256_file(run / "meta.json"),
                               "runtime_snapshot_hash": meta["config"]["runtime_snapshot_hash"]})
        merged = pd.concat(pieces, ignore_index=True)
        merged["abs_error"] = np.abs(merged.y_true - merged.final_pred)
        frames[arm] = merged.groupby(
            ["cell", "station", "month_index", "y_true", "age_group",
             "upstream_support_group", "flow", "flow_visible"], as_index=False
        ).agg(final_pred=("final_pred", "mean"), abs_error=("abs_error", "mean"))
    return frames, provenance


def fit_gate(x: np.ndarray, rf: np.ndarray, local: np.ndarray, y: np.ndarray,
             init_alpha: float, *, alpha_floor: float = 0.01) -> UnifiedSpatiotemporalFusion:
    model = UnifiedSpatiotemporalFusion(x.shape[1], init_alpha=max(init_alpha, alpha_floor),
                                        alpha_floor=alpha_floor, alpha_ceiling=1 - alpha_floor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    features = torch.as_tensor(x, dtype=torch.float32)
    z_rf = torch.as_tensor(np.log1p(np.clip(rf, 0, None)), dtype=torch.float32)
    z_local = torch.as_tensor(np.log1p(np.clip(local, 0, None)), dtype=torch.float32)
    z_y = torch.as_tensor(np.log1p(np.clip(y, 0, None)), dtype=torch.float32)
    for _ in range(500):
        _, fused = model(z_rf, z_local, features)
        loss = torch.mean((fused - z_y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def selected_alpha(y: np.ndarray, rf: np.ndarray, local: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 101)
    losses = []
    for alpha in grid:
        pred = blend(pd.DataFrame({"rf_context": rf, "residual_nomsg": local}),
                     np.full(len(y), alpha))
        losses.append(np.abs(y - pred).mean())
    return float(grid[int(np.argmin(losses))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("observable", "disagreement"), default="disagreement")
    parser.add_argument("--alpha-floor", type=float, default=0.01)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_frames, test_frames, provenance = {}, {}, []
    val_features, test_features = [], []
    val_y, val_rf, val_local = [], [], []
    test_y, test_rf, test_local = [], [], []
    for mask in ROOTS:
        val, val_prov = load_mask_five(mask, "val")
        test, test_prov = load_mask_five(mask, "test")
        provenance.extend(val_prov + test_prov)
        va, _ = feature_frame(val, mask)
        te, _ = feature_frame(test, mask)
        val_frames[mask], test_frames[mask] = val, test
        val_features.append(va)
        test_features.append(te)
        val_y.append(val["rf_context"].y_true.to_numpy())
        val_rf.append(val["rf_context"].final_pred.to_numpy())
        val_local.append(val["residual_nomsg"].final_pred.to_numpy())
        test_y.append(test["rf_context"].y_true.to_numpy())
        test_rf.append(test["rf_context"].final_pred.to_numpy())
        test_local.append(test["residual_nomsg"].final_pred.to_numpy())

    conditional_rows, conditional_effects = [], []
    all_test_y, all_test_rf, all_test_local, all_test_conditional = [], [], [], []
    for mask in ROOTS:
        va, _ = feature_frame(val_frames[mask], mask)
        te, _ = feature_frame(test_frames[mask], mask)
        x_val, x_test = encode(va, te, args.feature_set)
        yv = val_frames[mask]["rf_context"].y_true.to_numpy()
        rv = val_frames[mask]["rf_context"].final_pred.to_numpy()
        lv = val_frames[mask]["residual_nomsg"].final_pred.to_numpy()
        yt = test_frames[mask]["rf_context"].y_true.to_numpy()
        rt = test_frames[mask]["rf_context"].final_pred.to_numpy()
        lt = test_frames[mask]["residual_nomsg"].final_pred.to_numpy()
        alpha0 = selected_alpha(yv, rv, lv)
        model = fit_gate(x_val, rv, lv, yv, alpha0, alpha_floor=args.alpha_floor)
        with torch.no_grad():
            alpha, fused_z = model(torch.as_tensor(np.log1p(np.clip(rt, 0, None)), dtype=torch.float32),
                                   torch.as_tensor(np.log1p(np.clip(lt, 0, None)), dtype=torch.float32),
                                   torch.as_tensor(x_test, dtype=torch.float32))
        fused = np.expm1(fused_z.numpy())
        conditional_rows.append({"mask": mask, "n": len(yt), "alpha_init": alpha0,
                                 "alpha_mean": float(alpha.mean()), "alpha_sd": float(alpha.std()),
                                 "rf_mae": float(np.abs(yt - rt).mean()),
                                 "local_mae": float(np.abs(yt - lt).mean()),
                                 "fusion_mae": float(np.abs(yt - fused).mean())})
        for name, baseline in (("rf_context", rt), ("residual_nomsg", lt)):
            gain, lo, hi = station_bootstrap_gain(
                yt, fused, baseline, test_frames[mask]["rf_context"].station.to_numpy(), seed=42)
            conditional_effects.append({"mask": mask, "baseline": name,
                                        "gain_mae": gain, "ci_lo": lo, "ci_hi": hi})
        all_test_y.append(yt)
        all_test_rf.append(rt)
        all_test_local.append(lt)
        all_test_conditional.append(fused)

    global_val = pd.concat(val_features, ignore_index=True)
    global_test = pd.concat(test_features, ignore_index=True)
    xg_val, xg_test = encode(global_val, global_test, "mask_disagreement")
    global_model = fit_gate(xg_val, np.concatenate(val_rf), np.concatenate(val_local),
                            np.concatenate(val_y), 0.5, alpha_floor=args.alpha_floor)
    with torch.no_grad():
        global_alpha, global_fused_z = global_model(
            torch.as_tensor(np.log1p(np.clip(np.concatenate(test_rf), 0, None)), dtype=torch.float32),
            torch.as_tensor(np.log1p(np.clip(np.concatenate(test_local), 0, None)), dtype=torch.float32),
            torch.as_tensor(xg_test, dtype=torch.float32))
    global_fused = np.expm1(global_fused_z.numpy())
    global_rows = []
    offset = 0
    for mask in ROOTS:
        n = len(test_frames[mask]["rf_context"])
        sl = slice(offset, offset + n)
        yt, rt = np.concatenate(test_y)[sl], np.concatenate(test_rf)[sl]
        lt, gf = np.concatenate(test_local)[sl], global_fused[sl]
        global_rows.append({"mask": mask, "n": n, "alpha_mean": float(global_alpha[sl].mean()),
                            "alpha_sd": float(global_alpha[sl].std()), "rf_mae": float(np.abs(yt - rt).mean()),
                            "local_mae": float(np.abs(yt - lt).mean()),
                            "fusion_mae": float(np.abs(yt - gf).mean())})
        offset += n

    conditional = pd.DataFrame(conditional_rows)
    effects = pd.DataFrame(conditional_effects)
    global_summary = pd.DataFrame(global_rows)
    conditional.to_csv(args.out_dir / "conditional_summary.csv", index=False)
    effects.to_csv(args.out_dir / "conditional_effects.csv", index=False)
    global_summary.to_csv(args.out_dir / "global_summary.csv", index=False)
    y_all = np.concatenate(all_test_y)
    pooled = pd.DataFrame([
        {"system": "conditional_fusion", "mae": float(np.abs(y_all - np.concatenate(all_test_conditional)).mean())},
        {"system": "global_fusion", "mae": float(np.abs(y_all - global_fused).mean())},
        {"system": "rf_context", "mae": float(np.abs(y_all - np.concatenate(all_test_rf)).mean())},
        {"system": "residual_nomsg", "mae": float(np.abs(y_all - np.concatenate(all_test_local)).mean())},
    ])
    pooled.to_csv(args.out_dir / "pooled_summary.csv", index=False)
    manifest = {"seeds": list(SEEDS), "feature_set": args.feature_set,
                "alpha_floor": args.alpha_floor, "arms": list(ARMS),
                "input_predictions": provenance,
                "code_sha256": {p: sha256_file(Path(p)) for p in
                                ("scripts/run_unified_fusion_confirmation.py",
                                 "scripts/run_unified_gate_pilot.py",
                                 "src/river_graph/models/unified_fusion.py")}}
    (args.out_dir / "fusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "fusion_confirmation.md").write_text(
        "# Five-seed DOC unified fusion confirmation\n\n"
        "Both frozen experts use seeds 42--46. The conditional gate is fit separately per missingness family; the global gate is one model across all families. Both gates use validation labels only and are evaluated once on terminal test predictions.\n\n"
        "## Conditional gate\n\n" + conditional.to_string(index=False) +
        "\n\n## Conditional station-clustered effects\n\n" + effects.to_string(index=False) +
        "\n\n## Global gate\n\n" + global_summary.to_string(index=False) +
        "\n\n## Pooled\n\n" + pooled.to_string(index=False) + "\n"
    )
    print(conditional.to_string(index=False))
    print(global_summary.to_string(index=False))
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
