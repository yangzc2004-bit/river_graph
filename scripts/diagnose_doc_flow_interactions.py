"""Describe state-dependent flow readout use on source-validation queries only.

No labels are scored and no model is fitted or selected. The quantities are
conditional readout sensitivities with the recurrent state and other derived
features held fixed, not causal effects or physical flow-response coefficients.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_flow_interaction_v1 import ARMS, ROOT
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from verify_doc_tail_residual_v1 import _full_inputs

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.episodic_temporal_adapter import gathered_rolling_states
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)


def distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {prefix + key: None for key in (
            "q05", "q25", "median", "q75", "q95", "positive_fraction", "negative_fraction",
            "zero_fraction", "mean", "mean_abs", "rms")}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite readout diagnostic")
    result = {prefix + key: float(value) for key, value in zip(
        ("q05", "q25", "median", "q75", "q95"), np.quantile(values, [.05, .25, .5, .75, .95]))}
    result.update({prefix + "positive_fraction": float(np.mean(values > 0)),
                   prefix + "negative_fraction": float(np.mean(values < 0)),
                   prefix + "zero_fraction": float(np.mean(values == 0)),
                   prefix + "mean": float(values.mean()),
                   prefix + "mean_abs": float(np.abs(values).mean()),
                   prefix + "rms": float(np.sqrt(np.mean(values**2)))})
    return result


def load_source_without_held_labels(config):
    """Construct the encoder view using source labels and no held DOC values."""
    source = Path(config["source_run"])
    old = json.loads((source / "config.json").read_text())
    if sha256_file(source / "complete.json") != config["source_completion_hash"]:
        raise ValueError("Source checkpoint binding changed")
    verify_files(source, "complete.json", old)
    for kind in ("dataset", "mask"):
        if old[f"{kind}_hash"] != config[f"{kind}_hash"]:
            raise ValueError(f"Different source {kind}")
        if sha256_file(old[f"{kind}_path"]) != old[f"{kind}_hash"]:
            raise ValueError(f"Changed source {kind}")
    dataset = torch.load(old["dataset_path"], map_location="cpu", weights_only=False)
    with np.load(old["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role] for role in ("train", "val", "test", "context")}
    if len(split["context"]):
        raise ValueError("This diagnostic expects train-only DOC input visibility")
    visible_y = torch.zeros_like(dataset["y"])
    visible_y.reshape(-1)[split["train"]] = dataset["y"].reshape(-1)[split["train"]]
    dataset = {**dataset, "y": visible_y}
    # No y_true columns or adapted test-query products are loaded.
    baseline = pd.read_parquet(source / "full_grid.parquet", columns=["cell", "context_pred"])
    np.testing.assert_array_equal(baseline.cell, np.arange(visible_y.numel()))
    return source, dataset, split, baseline.context_pred.to_numpy()


def run_diagnostic(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("Run and execution snapshot differ")
    verify_files(run, "complete.json", config)
    torch.set_num_threads(int(config["torch_threads"]))
    source, dataset, split, context = load_source_without_held_labels(config)
    months = dataset["y"].shape[1]
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    if np.intersect1d(query, split["test"]).size:
        raise ValueError("Validation diagnostic includes target-query cells")
    expert = UnifiedDOCReconstructor.load(source, dataset, split)
    recurrent = _full_inputs(expert)
    flow = build_causal_flow_features(dataset)
    if {key: value for key, value in flow.items() if key != "full"} != config["flow_definition"]:
        raise ValueError("Recomputed flow definition differs")
    inputs = {**recurrent, "extra": flow["full"]}
    del expert, dataset, recurrent
    feature_rows, run_rows, binding = [], [], {"run": run.name,
        "completion_sha256": sha256_file(run / "complete.json"),
        "source_completion_sha256": config["source_completion_hash"], "checkpoint_sha256": {}}
    for arm in ARMS:
        payload_path = run / f"{arm}.pt"
        model = NativeTemporalResidual.from_payload(torch.load(payload_path, map_location="cpu", weights_only=False))
        binding["checkpoint_sha256"][arm] = sha256_file(payload_path)
        if tuple(model.interaction_indices) != (0, 2, 4):
            raise ValueError("Expected the three numerical flow interactions")
        prepared = model._prepare_inputs(inputs)
        state_parts, prediction_parts = [], []
        with torch.inference_mode():
            for start in range(0, len(query), 2048):
                cells = torch.as_tensor(query[start:start + 2048])
                hidden = gathered_rolling_states(
                    model.temporal, model.decay, prepared, cells // months, cells % months,
                    lookback=model.lookback)
                state_parts.append(hidden.double().numpy())
                prediction_parts.append(model._delta_cells(prepared, cells).numpy())
        h, actual_delta = np.concatenate(state_parts), np.concatenate(prediction_parts)
        f = flow["full"].reshape(-1, model.extra_dim)[query].astype(float)
        weight = model.head.weight.detach().double().numpy().reshape(-1)
        bias = float(model.head.bias.detach().double()[0])
        h_weight, f_weight = weight[:model.hidden_size], weight[model.hidden_size:model.hidden_size + model.extra_dim]
        interactions = weight[model.hidden_size + model.extra_dim:].reshape(model.hidden_size, -1)
        state_coefficient = h @ interactions
        numeric = np.asarray(model.interaction_indices)
        total_coefficient = state_coefficient + f_weight[numeric][None, :]
        local_component = h @ h_weight
        additive = f[:, numeric] * f_weight[numeric]
        interaction = f[:, numeric] * state_coefficient
        freshness = np.setdiff1d(np.arange(model.extra_dim), numeric)
        freshness_component = f[:, freshness] @ f_weight[freshness]
        reconstructed = local_component + additive.sum(1) + interaction.sum(1) + freshness_component + bias
        np.testing.assert_allclose(reconstructed, actual_delta, rtol=1e-5, atol=1e-5)
        scale = model.selected_scale_
        raw_final = context[query] + scale * actual_delta
        # Matches the implementation's chosen derivative at the zero boundary.
        floor_inactive = raw_final >= 0
        effective = scale * total_coefficient * floor_inactive[:, None]
        identity = {"run": run.name, "split_seed": split_seed, "seed": seed, "arm": arm,
                    "train_memory": model.train_memory, "best_epoch": model.best_epoch_,
                    "selected_scale": scale, "n_validation_query": len(query),
                    "n_validation_stations": len(np.unique(query // months))}
        for j, index in enumerate(numeric):
            valid = f[:, index + 1] > 0
            for scope, selection in (("all_queries", np.ones(len(query), dtype=bool)),
                                      ("feature_valid", valid)):
                row = {**identity, "feature": flow["feature_names"][index], "feature_index": int(index),
                       "scope": scope, "n_cells": int(selection.sum()), "valid_fraction": float(valid.mean()),
                       "additive_coefficient": float(f_weight[index]),
                       "scaled_additive_coefficient": float(scale * f_weight[index]),
                       **distribution(total_coefficient[selection, j], "coefficient_"),
                       **distribution(effective[selection, j], "effective_coefficient_"),
                       **distribution(scale * additive[selection, j], "additive_contribution_"),
                       **distribution(scale * interaction[selection, j], "interaction_contribution_")}
                feature_rows.append(row)
        realized = np.maximum(raw_final, 0) - context[query]
        row = {**identity, "floor_active_fraction": float(np.mean(~floor_inactive)),
               "decomposition_max_abs_error": float(np.max(np.abs(reconstructed - actual_delta)))}
        for name, values in (("state", scale * local_component),
                             ("numeric_additive", scale * additive.sum(1)),
                             ("numeric_interaction", scale * interaction.sum(1)),
                             ("freshness", scale * freshness_component),
                             ("bias", np.full(len(query), scale * bias)),
                             ("realized_residual", realized)):
            row.update(distribution(values, name + "_"))
        run_rows.append(row)
        del model, prepared, state_parts, prediction_parts
    return feature_rows, run_rows, binding


def write_report(root, features, runs, bindings, runtime):
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    feature_table, run_table = pd.DataFrame(features), pd.DataFrame(runs)
    feature_table.to_csv(output / "interaction_diagnostics.csv", index=False)
    run_table.to_csv(output / "interaction_diagnostics_runs.csv", index=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "diagnostic_script_sha256": sha256_file(__file__), "runtime_snapshot_hash": runtime,
              "scope": "source-validation fixed K0 query cells only; no target/query labels scored",
              "feature_rows": len(features), "run_arm_rows": len(runs), "bindings": bindings,
              "formula": "conditional head coefficient = beta_flow + hidden @ W_interaction",
              "effective_formula": "selected_scale * coefficient * (unclipped_prediction >= 0)",
              "units": "mg/L per unit of a bounded, dimensionless derived flow feature",
              "interpretation": "holding recurrent state and other derived inputs fixed; neither a total discharge derivative nor a causal/physical effect",
              "label_boundary": "all non-training DOC labels zeroed before reconstructing features; no label values enter statistics",
              "decomposition_note": "signed contributions can cancel; mean absolute component magnitudes are not additive attribution percentages"}
    (output / "interaction_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Source-validation flow interaction diagnostics", "",
             ("These diagnostics describe how the fitted readout uses its recurrent state and explicit flow inputs. "
              "They do not fit/select models, inspect target-query labels, or introduce a performance endpoint."), "",
             ("For numerical feature j, the conditional readout coefficient is `beta_j + h @ W_j`. "
             "The effective coefficient also applies the selected residual scale and nonnegative output floor. "
             "Coefficients are in mg/L per unit of the bounded feature, holding the recurrent state and other "
             "derived features fixed. Correlated flow features and flow-dependent recurrent states mean these "
             "are not causal effects, physical transport parameters, or total derivatives with respect to discharge."), "",
             "## Magnitudes across all validation queries", "",
             ("Each value below is the equal mean of the nine run-level means; repeated seeds reuse validation cells. "
              "Signed components can cancel, so their absolute magnitudes are not attribution percentages."), "",
             "| Arm | Numeric additive mean absolute contribution | Numeric interaction mean absolute contribution | Realized correction mean absolute |",
             "|---|---:|---:|---:|"]
    for arm, frame in run_table.groupby("arm", sort=False):
        lines.append(f"| {arm} | {frame.numeric_additive_mean_abs.mean():.4f} | "
                     f"{frame.numeric_interaction_mean_abs.mean():.4f} | {frame.realized_residual_mean_abs.mean():.4f} |")
    lines += ["", "## State-dependent signs on feature-valid query cells", "",
              ("Fractions are equal run means and describe the effective readout coefficient after the selected "
              "scale/floor. `interaction_diagnostics.csv` retains each run’s quantiles, validity fractions and "
              "both valid-only and all-query summaries."), "",
              "| Arm | Feature | Positive fraction | Negative fraction | Zero fraction |",
              "|---|---|---:|---:|---:|"]
    valid = feature_table[feature_table.scope.eq("feature_valid")]
    for (arm, feature), frame in valid.groupby(["arm", "feature"], sort=False):
        lines.append(f"| {arm} | {feature} | {frame.effective_coefficient_positive_fraction.mean():.3f} | "
                     f"{frame.effective_coefficient_negative_fraction.mean():.3f} | "
                     f"{frame.effective_coefficient_zero_fraction.mean():.3f} |")
    lines += ["", ("These summaries describe state-dependent readout use. Whether these adjustments improve "
                   "held-out prediction remains a question for the matched performance comparisons."), ""]
    (output / "interaction_diagnostics.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    requested = [(split, seed) for split in SPLITS for seed in SEEDS]
    missing = [f"split{split}_seed{seed}" for split, seed in requested
               if not (args.root / "runs" / f"split{split}_seed{seed}" / "complete.json").is_file()]
    if missing:
        raise ValueError(f"Complete all nine production packages before diagnostics: {missing}")
    runtime = verify_runtime_snapshot(args.root)
    features, runs, bindings = [], [], []
    for split, seed in requested:
        print(f"Validation readout diagnostic: split{split}_seed{seed}", flush=True)
        f, r, b = run_diagnostic(args.root, split, seed, runtime)
        features.extend(f)
        runs.extend(r)
        bindings.append(b)
        gc.collect()
    write_report(args.root, features, runs, bindings, runtime)
    print(args.root / "analysis" / "interaction_diagnostics.md")


if __name__ == "__main__":
    main()
