"""Pilot an observable-feature gate over the two completed DOC experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARMS = ("rf_context", "residual_nomsg")
SEEDS = (42, 43, 44)
ROOTS = {
    "e1_r20_seed42": Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
    "e2b_partial": Path("experiments/phase4_transfer/kgml_local_transport_v1/conditional_expert_pilot"),
    "e2a_strict": Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
    "e3_spatial_seed42": Path("experiments/phase4_transfer/kgml_local_transport_v1/k1"),
}


def load_mask(root: Path, mask: str, role: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for arm in ARMS:
        pieces = []
        for seed in SEEDS:
            metas = []
            for meta_path in (root / "runs").glob("*/meta.json"):
                meta = json.loads(meta_path.read_text())
                cfg = meta["config"]
                if cfg["mask"] == mask and cfg["arm"] == arm and cfg["seed"] == seed:
                    metas.append(meta_path)
            if len(metas) != 1:
                raise ValueError(f"expected one {arm} seed {seed} run for {mask}")
            frame = pd.read_parquet(metas[0].parent / f"{role}_predictions.parquet")
            pieces.append(frame[["cell", "station", "month_index", "y_true", "final_pred",
                                 "age_group", "upstream_support_group", "flow", "flow_visible"]])
        merged = pd.concat(pieces, ignore_index=True)
        merged["abs_error"] = np.abs(merged.y_true - merged.final_pred)
        frames[arm] = merged.groupby(
            ["cell", "station", "month_index", "y_true", "age_group",
             "upstream_support_group", "flow", "flow_visible"], as_index=False
        ).agg(final_pred=("final_pred", "mean"), abs_error=("abs_error", "mean"))
    return frames


def feature_frame(frames: dict[str, pd.DataFrame], mask: str) -> tuple[pd.DataFrame, np.ndarray]:
    ref = frames[ARMS[0]]
    if not ref.cell.equals(frames[ARMS[1]].cell):
        raise ValueError("expert query cells differ")
    out = ref[["cell", "month_index", "age_group", "upstream_support_group", "flow", "flow_visible"]].copy()
    out["mask"] = mask
    out["rf_prediction"] = ref.final_pred.to_numpy()
    out["residual_prediction"] = frames["residual_nomsg"].final_pred.to_numpy()
    out["prediction_gap"] = np.abs(
        np.log1p(np.clip(out["rf_prediction"], 0, None))
        - np.log1p(np.clip(out["residual_prediction"], 0, None))
    )
    angle = 2 * np.pi * out.month_index / 12
    out["month_sin"], out["month_cos"] = np.sin(angle), np.cos(angle)
    better = (frames["residual_nomsg"].abs_error < frames["rf_context"].abs_error).astype(int).to_numpy()
    return out, better


def encode(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    if feature_set not in {"mask", "observable", "disagreement", "mask_disagreement"}:
        raise ValueError("feature_set must be mask, observable, disagreement, or mask_disagreement")
    categorical = ["age_group", "upstream_support_group"]
    if feature_set in {"mask", "mask_disagreement"}:
        categorical = ["mask", *categorical]
    numeric = ["month_sin", "month_cos", "flow", "flow_visible"]
    if feature_set in {"disagreement", "mask_disagreement"}:
        numeric = [*numeric, "rf_prediction", "residual_prediction", "prediction_gap"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    a = enc.fit_transform(train[categorical])
    b = enc.transform(test[categorical])
    scaler = StandardScaler()
    return np.column_stack([scaler.fit_transform(train[numeric]), a]), np.column_stack([scaler.transform(test[numeric]), b])


def blend(frame: pd.DataFrame, gate: np.ndarray) -> np.ndarray:
    z_rf = np.log1p(np.clip(frame["rf_context"].to_numpy(), 0, None))
    z_local = np.log1p(np.clip(frame["residual_nomsg"].to_numpy(), 0, None))
    return np.expm1(gate * z_local + (1 - gate) * z_rf)


def station_bootstrap_gain(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    stations: np.ndarray,
    *,
    seed: int,
    draws: int = 1000,
) -> tuple[float, float, float]:
    """Station-clustered CI for positive MAE gain of candidate over baseline."""
    _, inverse = np.unique(stations, return_inverse=True)
    counts = np.bincount(inverse)
    diff = np.abs(y - baseline) - np.abs(y - candidate)
    means = np.bincount(inverse, weights=diff) / counts
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(means), size=(draws, len(means)))
    boot = means[sampled].mean(axis=1)
    return float(means.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def direct_gate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    z_rf_train: np.ndarray,
    z_local_train: np.ndarray,
    z_y_train: np.ndarray,
    z_rf_test: np.ndarray,
    z_local_test: np.ndarray,
) -> np.ndarray:
    """Fit a low-dimensional gate directly against validation prediction loss."""
    torch.manual_seed(42)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    xq = torch.as_tensor(x_test, dtype=torch.float32)
    zr = torch.as_tensor(z_rf_train, dtype=torch.float32)
    zl = torch.as_tensor(z_local_train, dtype=torch.float32)
    zy = torch.as_tensor(z_y_train, dtype=torch.float32)
    weight = torch.zeros((xt.shape[1], 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam((weight, bias), lr=0.05, weight_decay=1e-3)
    for _ in range(500):
        alpha = torch.sigmoid(xt @ weight + bias).squeeze(-1)
        pred = (1 - alpha) * zr + alpha * zl
        loss = torch.mean((pred - zy) ** 2) + 1e-3 * torch.sum(weight ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(xq @ weight + bias).squeeze(-1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("mask", "observable", "disagreement", "mask_disagreement"), default="mask")
    parser.add_argument("--objective", choices=("classifier", "direct", "family_constant"), default="classifier")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_frames, test_frames = {}, {}
    val_x, val_y, val_rf, val_local, test_x, test_rf, test_local = [], [], [], [], [], [], []
    for mask, root in ROOTS.items():
        val = load_mask(root, mask, "val")
        test = load_mask(root, mask, "test")
        va, y = feature_frame(val, mask)
        te, _ = feature_frame(test, mask)
        val_frames[mask], test_frames[mask] = val, test
        val_x.append(va)
        val_y.append(y)
        val_rf.append(val["rf_context"].final_pred.to_numpy())
        val_local.append(val["residual_nomsg"].final_pred.to_numpy())
        test_x.append(te)
        test_rf.append(test["rf_context"].final_pred.to_numpy())
        test_local.append(test["residual_nomsg"].final_pred.to_numpy())
    val_features, val_labels = pd.concat(val_x, ignore_index=True), np.concatenate(val_y)
    test_features = pd.concat(test_x, ignore_index=True)
    selected = {}
    if args.objective == "family_constant":
        test_gate_parts = []
        for mask in ROOTS:
            val_rf_pred = val_frames[mask]["rf_context"].final_pred.to_numpy()
            val_local_pred = val_frames[mask]["residual_nomsg"].final_pred.to_numpy()
            val_y_true = val_frames[mask]["rf_context"].y_true.to_numpy()
            grid = np.linspace(0.0, 1.0, 101)
            losses = []
            for alpha in grid:
                pred = blend(pd.DataFrame({"rf_context": val_rf_pred,
                                           "residual_nomsg": val_local_pred}),
                             np.full(len(val_rf_pred), alpha))
                losses.append(np.abs(val_y_true - pred).mean())
            alpha = float(grid[int(np.argmin(losses))])
            selected[mask] = alpha
            test_gate_parts.append(np.full(len(test_frames[mask]["rf_context"]), alpha))
        test_gate = np.concatenate(test_gate_parts)
    else:
        x_train, x_test = encode(val_features, test_features, args.feature_set)
    if args.objective == "classifier":
        model = make_pipeline(LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42))
        model.fit(x_train, val_labels)
        test_gate = model.predict_proba(x_test)[:, 1]
    elif args.objective == "direct":
        z_rf_train = np.log1p(np.clip(np.concatenate(val_rf), 0, None))
        z_local_train = np.log1p(np.clip(np.concatenate(val_local), 0, None))
        z_y_train = np.log1p(np.clip(np.concatenate([val_frames[m]["rf_context"].y_true.to_numpy() for m in ROOTS]), 0, None))
        z_rf_test = np.log1p(np.clip(np.concatenate(test_rf), 0, None))
        z_local_test = np.log1p(np.clip(np.concatenate(test_local), 0, None))
        test_gate = direct_gate(x_train, x_test, z_rf_train, z_local_train, z_y_train,
                                z_rf_test, z_local_test)
    rows, effects, pooled = [], [], []
    offset = 0
    for mask in ROOTS:
        n = len(test_frames[mask]["rf_context"])
        gate = test_gate[offset:offset + n]
        rf = test_frames[mask]["rf_context"]
        local = test_frames[mask]["residual_nomsg"]
        y = rf.y_true.to_numpy()
        pred = blend(pd.DataFrame({"rf_context": rf.final_pred, "residual_nomsg": local.final_pred}), gate)
        for baseline_name, baseline in (("rf_context", rf.final_pred.to_numpy()),
                                        ("residual_nomsg", local.final_pred.to_numpy())):
            gain, ci_lo, ci_hi = station_bootstrap_gain(
                y, pred, baseline, rf.station.to_numpy(), seed=42 + len(effects)
            )
            effects.append({"feature_set": args.feature_set, "mask": mask,
                            "candidate": "gate", "baseline": baseline_name,
                            "gain_mae": gain, "ci_lo": ci_lo, "ci_hi": ci_hi})
        row = {"feature_set": args.feature_set, "objective": args.objective, "mask": mask, "n": n,
               "family_alpha": selected.get(mask, np.nan),
               "gate_mean_local": float(gate.mean()),
               "rf_context_mae": float(np.abs(y - rf.final_pred).mean()),
               "residual_nomsg_mae": float(np.abs(y - local.final_pred).mean()),
               "gate_mae": float(np.abs(y - pred).mean())}
        rows.append(row)
        pooled.append(pd.DataFrame({"y": y, "pred": pred, "rf": rf.final_pred, "local": local.final_pred}))
        offset += n
    summary = pd.DataFrame(rows)
    effects_df = pd.DataFrame(effects)
    effects_df.insert(1, "objective", args.objective)
    summary.to_csv(args.out_dir / "gate_summary.csv", index=False)
    effects_df.to_csv(args.out_dir / "gate_effects.csv", index=False)
    all_rows = pd.concat(pooled, ignore_index=True)
    pooled_summary = pd.DataFrame([
        {"system": "gate", "mae": float(np.abs(all_rows.y - all_rows.pred).mean())},
        {"system": "rf_context", "mae": float(np.abs(all_rows.y - all_rows.rf).mean())},
        {"system": "residual_nomsg", "mae": float(np.abs(all_rows.y - all_rows.local).mean())},
    ])
    pooled_summary.to_csv(args.out_dir / "gate_pooled.csv", index=False)
    (args.out_dir / "gate_verdict.md").write_text(
        "# Unified observable-feature gate pilot\n\n"
        "The gate was fit on validation errors and evaluated on terminal test predictions.\n\n"
        + f"Feature set: `{args.feature_set}`; objective: `{args.objective}`.\n\n"
        + summary.to_string(index=False) + "\n\n## Pooled test\n\n"
        + pooled_summary.to_string(index=False) + "\n\n## Station-clustered gain\n\n"
        + effects_df.to_string(index=False) + "\n"
    )
    print(summary.to_string(index=False))
    print(pooled_summary.to_string(index=False))


if __name__ == "__main__":
    main()
