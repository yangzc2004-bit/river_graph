"""Audit and analyze the frozen sequential DOC component comparison."""
from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_component_comparison_v1 import (
    DATASET,
    MASKS,
    NODES,
    ROOT,
    STAGES,
    architecture,
    load_model,
    select_choice,
    spec_name,
    specification,
    verify_package,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.river_architecture_comparison import (
    covariate_inputs,
    graph_view,
)


def assert_fields(record, expected):
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError("provenance field disagrees")


def replay_identity(root, region, protocol):
    return {"region": region, "protocol_hash": digest(protocol),
        "regional_complete_sha256": sha256_file(root / "regions" / f"huc4_{region}" / "complete.json"),
        "analyzer_sha256": sha256_file(__file__), "runtime_snapshot": protocol["runtime_snapshot"],
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "torch_threads": torch.get_num_threads()}


def save_replay_receipt(root, region, protocol, difference):
    write_json(root / "analysis/region_replay" / f"huc4_{region}.json", {
        **replay_identity(root, region, protocol), "fits": 33, "validation_and_test_replayed": True,
        "maximum_native_difference": difference})


def preflight_replay(root, protocol, data):
    """One immutable completed fit per region, validation-only smoke audit."""
    rows = []
    for region in protocol["regions"]:
        run = root / "regions" / f"huc4_{region}"
        markers = sorted((run / "fits").glob("*/seed*/fit_complete.json"))
        if not markers:
            continue
        directory = markers[0].parent
        verify_package(directory, "fit_complete.json")
        with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as split:
            source = np.unique(split["train"] // data["y"].shape[1])
            cells = split["val"].copy()
        inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                                  source, lookback=protocol["settings"]["lookback"])
        t = data["y"].shape[1]
        graph = graph_view(data["edge_index"], inputs["order"], np.sort(np.r_[source, np.unique(cells // t)]))
        info = json.loads((directory / "info.json").read_text())
        model = load_model(directory)
        grid = native_prediction(predict(model, inputs, graph, np.arange(t), protocol["settings"]["batch_months"]),
                                 info["target_normalization"])
        prediction = grid[cells % t, np.searchsorted(graph["rows"].numpy(), cells // t)]
        saved = pd.read_parquet(directory / "validation_predictions.parquet")
        np.testing.assert_array_equal(saved.cell, cells)
        np.testing.assert_array_equal(saved.y_true, np.asarray(data["y"]).ravel()[cells])
        np.testing.assert_allclose(prediction, saved.y_pred, rtol=1e-6, atol=1e-6)
        rows.append({"region": region, "fit": str(directory), "validation_cells": len(cells),
                     "maximum_difference": float(np.abs(prediction - saved.y_pred.to_numpy()).max()),
                     "fit_complete_sha256": sha256_file(directory / "fit_complete.json")})
        del model
    write_json(root / "analysis/preflight_replay.json", {"validation_only": True, "records": rows})
    print(json.dumps(rows, indent=2), flush=True)


def audit_region(root, region, protocol, data, replay):
    run = root / "regions" / f"huc4_{region}"
    config = json.loads((run / "config.json").read_text())
    assert_fields(config, {"protocol_hash": digest(protocol), "target_huc4": region,
                          "settings": protocol["settings"],
                          "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
                          "dataset_hash": protocol["dataset_hash"], "mask_hash": protocol["mask_hashes"][region]})
    verify_package(run, "complete.json", config)
    selection_path = run / "selections.json"
    selection = json.loads(selection_path.read_text())
    if selection["test_scores_used"]:
        raise ValueError("test-selected pipeline")
    selection_hash = sha256_file(selection_path)
    meta = json.loads((run / "predictions.meta.json").read_text())
    assert_fields(meta, {"config": config, "config_hash": digest(config),
        "prediction_sha256": sha256_file(run / "predictions.parquet"),
        "selections_sha256": selection_hash, "aliases": selection["aliases"]})
    with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test")}
    t = data["y"].shape[1]
    source = np.unique(split["train"] // t)
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                              source, lookback=protocol["settings"]["lookback"])
    if inputs["normalization"] != json.loads((run / "normalization.json").read_text()):
        raise ValueError("normalization replay differs")
    graphs = {role: graph_view(data["edge_index"], inputs["order"],
              np.sort(np.r_[source, np.unique(split[role] // t)])) for role in ("val", "test")}
    aliases, scores, costs, maximum_difference = selection["aliases"], {}, {}, 0.
    frame = pd.read_parquet(run / "predictions.parquet")
    if frame.duplicated(["model_name", "seed", "cell"]).any() or not np.isfinite(
            frame[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("duplicate or nonfinite query predictions")
    for directory in sorted((run / "fits").glob("*/seed*")):
        key, seed = directory.parent.name, int(directory.name.removeprefix("seed"))
        fit_config = json.loads((directory / "config.json").read_text())
        assert_fields(fit_config, {**config, "seed": seed})
        if spec_name(fit_config["specification"]) != key:
            raise ValueError("fit specification disagrees with path")
        verify_package(directory, "fit_complete.json", fit_config)
        verify_package(directory, "test_complete.json", fit_config)
        if selection["fit_hashes"][f"{key}/seed{seed}"] != sha256_file(directory / "fit_complete.json"):
            raise ValueError("fit changed after selection")
        info = json.loads((directory / "info.json").read_text())
        payload = torch.load(directory / "checkpoint.pt", weights_only=False, map_location="cpu")
        assert_fields(payload, {"specification": fit_config["specification"],
                                "architecture": architecture(protocol["settings"])})
        val = pd.read_parquet(directory / "validation_predictions.parquet")
        test = pd.read_parquet(directory / "test_predictions.parquet")
        for role, part in (("val", val), ("test", test)):
            np.testing.assert_array_equal(part.cell, split[role])
            np.testing.assert_array_equal(part.y_true, np.asarray(data["y"]).ravel()[split[role]])
            if not np.isfinite(part[["y_true", "y_pred"]].to_numpy()).all():
                raise ValueError("incomplete fitted-model coverage")
        val_mae = float(np.abs(val.y_pred.to_numpy(dtype=np.float64) - val.y_true.to_numpy(dtype=np.float64)).mean())
        if abs(val_mae - info["validation_mae"]) > 1e-5:
            raise ValueError("validation selection score disagrees with predictions")
        trace = pd.read_csv(directory / "trace.csv")
        recorded = trace.loc[trace.epoch == info["best_epoch"], "validation_mae"].iloc[0]
        if abs(recorded - info["validation_mae"]) > 1e-7:
            raise ValueError("checkpoint does not match selected validation epoch")
        sidecar = json.loads((directory / "test_predictions.meta.json").read_text())
        assert_fields(sidecar, {"config": fit_config, "config_hash": digest(fit_config),
            "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
            "runtime_snapshot_hash": config["runtime_snapshot_hash"],
            "run_identity_sha256": run_identity_sha256(digest(fit_config), fit_config["started_at"],
                                                       config["runtime_snapshot_hash"]),
            "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
            "prediction_sha256": sha256_file(directory / "test_predictions.parquet"),
            "selections_sha256": selection_hash})
        val_meta = json.loads((directory / "validation_predictions.meta.json").read_text())
        assert_fields(val_meta, {"config_hash": digest(fit_config), "dataset_sha256": config["dataset_hash"],
            "mask_sha256": config["mask_hash"], "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "prediction_sha256": sha256_file(directory / "validation_predictions.parquet")})
        for alias, canonical in aliases.items():
            if canonical != key:
                continue
            part = frame[(frame.model_name == alias) & (frame.seed == seed)]
            np.testing.assert_array_equal(part.cell, test.cell)
            np.testing.assert_array_equal(part.y_true, test.y_true)
            np.testing.assert_array_equal(part.y_pred, test.y_pred)
            np.testing.assert_array_equal(part.station, np.asarray(data["site_no"], str)[test.cell.to_numpy() // t])
            np.testing.assert_array_equal(part.month, np.asarray(data["months"], str)[test.cell.to_numpy() % t])
            for component, value in fit_config["specification"].items():
                if not (part[component] == value).all():
                    raise ValueError("alias component label differs")
        scores[(key, seed)] = info["validation_mae"]
        costs[(key, seed)] = {**info, "inference_seconds": sidecar["inference_seconds"]}
        if replay:
            model = load_model(directory)
            if sum(value.numel() for value in model.parameters()) != info["parameters"]:
                raise ValueError("reported parameter count differs from fitted model")
            for role, part in (("val", val), ("test", test)):
                graph, cells = graphs[role], split[role]
                grid = native_prediction(predict(model, inputs, graph, np.arange(t),
                                                protocol["settings"]["batch_months"]), info["target_normalization"])
                rows = np.searchsorted(graph["rows"].numpy(), cells // t)
                predicted = grid[cells % t, rows]
                stored = part.y_pred.to_numpy()
                np.testing.assert_allclose(predicted, stored, atol=1e-6, rtol=1e-6)
                maximum_difference = max(maximum_difference, float(np.abs(predicted - stored).max()))
            del model
            gc.collect()
    if len(scores) != 33 or len(frame) != len(split["test"]) * 17 * 3:
        raise ValueError("incorrect unique fit or alias coverage")
    choices = []
    for stage, definition in STAGES.items():
        reconstructed = {choice: [scores[(aliases[f"{stage}__{choice}"], seed)]
                                  for seed in protocol["seeds"]] for choice in definition["choices"]}
        recorded = selection["stages"][stage]
        if reconstructed != recorded["validation_scores"]:
            raise ValueError("selection scores differ from immutable fits")
        selected = select_choice(reconstructed, definition["eligible"])
        if selected != recorded["selected"] or aliases[f"selected__{stage}"] != aliases[f"{stage}__{selected}"]:
            raise ValueError("selection rule was not followed")
        for choice, values in reconstructed.items():
            choices.append({"target_huc4": region, "stage": stage, "choice": choice,
                "mean_validation_mae": float(np.mean(values)), "selected": choice == selected,
                "eligible": choice in definition["eligible"], "fit_specification": aliases[f"{stage}__{choice}"]})
    if aliases["selected__reference"] != spec_name(specification()):
        raise ValueError("reference was changed")
    timing = [{"target_huc4": region, "seed": seed, "model_name": alias, "fit_specification": key,
               **costs[(key, seed)]} for alias, key in aliases.items() for seed in protocol["seeds"]]
    print(json.dumps({"audited_region": region, "fits": len(scores), "replay_difference": maximum_difference}), flush=True)
    return frame, timing, choices, maximum_difference


def metrics(part):
    truth, pred = part.y_true.to_numpy(dtype=np.float64), part.y_pred.to_numpy(dtype=np.float64)
    error = pred - truth
    denominator = np.square(truth - truth.mean()).sum()
    station = part.assign(error=np.abs(error)).groupby("station").error.mean()
    tail = part.high_doc.to_numpy()
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator), "bias": float(error.mean()),
        "station_macro_mae": float(station.mean()),
        "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else None,
        "query_rows": len(part), "unique_cells": int(part.cell.nunique()), "stations": int(part.station.nunique())}


def interval(frame, baseline, candidate, unit="target_huc4"):
    cells = frame[frame.model_name.isin([baseline, candidate])].assign(
        error=lambda part: np.abs(part.y_pred.to_numpy(dtype=np.float64) - part.y_true.to_numpy(dtype=np.float64)))
    cells = cells.groupby(["target_huc4", "station", "cell", "model_name"]).error.mean().unstack("model_name").reset_index()
    grouped = cells.groupby(unit).agg(base=(baseline, "sum"), candidate=(candidate, "sum"), n=("cell", "count"))
    rng = np.random.default_rng(261010)
    samples = rng.integers(0, len(grouped), (20000, len(grouped)))
    base = grouped.base.to_numpy()[samples].sum(1)
    other = grouped.candidate.to_numpy()[samples].sum(1)
    gain = 100 * (base - other) / base
    return {"clusters": len(grouped), "cluster_unit": unit,
            "gain_percent_interval": np.quantile(gain, [.025, .975]).tolist()}


def plot_results(out, summary, pairs, basin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), layout="constrained")
    for ax, (stage, definition) in zip(axes, STAGES.items(), strict=True):
        choices = list(definition["choices"])
        values = [summary.loc[f"{stage}__{choice}", "mae"] for choice in choices]
        baseline = {"environment": "mlp", "time": "gru", "fusion": "concat"}[stage]
        colors = ["#b4bdc8" if choice not in definition["eligible"] else
                  "#345e8a" if choice == baseline else "#4d9b92" for choice in choices]
        ax.barh(choices, values, color=colors, height=.63)
        ax.invert_yaxis()
        ax.set_xlabel("DOC MAE (mg/L; lower is better)")
        ax.set_title(f"{stage.capitalize()} candidates")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(values) * 1.2)
        for i, value in enumerate(values):
            ax.text(value + .025, i, f"{value:.3f}", va="center", fontsize=9)
    figure.suptitle("DOC component comparison · 5 held-out regions × 3 seeds\n"
                   "Time/fusion comparisons use preceding validation-selected modules", fontsize=12)
    figure.savefig(out / "component_mae.png", dpi=180)
    figure.savefig(out / "component_mae.pdf")
    plt.close(figure)
    selected = [p for p in pairs if p["candidate"].startswith("selected__")]
    figure, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    for i, pair in enumerate(selected):
        lo, hi = pair["basin_bootstrap"]["gain_percent_interval"]
        gain = pair["gain_percent"]
        ax.plot([lo, hi], [i, i], color="#345e8a", linewidth=2)
        ax.scatter(gain, i, color="#345e8a", s=35, zorder=3)
    ax.set_yticks(range(len(selected)), [p["candidate"].removeprefix("selected__") for p in selected])
    ax.invert_yaxis()
    ax.axvline(0, color="#777", linewidth=1)
    ax.set_xlabel("MAE gain vs MLP/GRU/concat reference (%)\n95% paired basin bootstrap, only 5 basins")
    ax.set_title("Cumulative validation-selected architecture")
    ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(out / "selected_pipeline_gain.png", dpi=180)
    figure.savefig(out / "selected_pipeline_gain.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--region", help="audit/replay one completed region without publishing a partial ranking")
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(protocol) != manifest["protocol_hash"] or sha256_file(DATASET) != protocol["dataset_hash"]:
        raise ValueError("protocol or dataset changed")
    if sha256_file(NODES) != protocol["nodes_hash"]:
        raise ValueError("node inventory changed")
    assert_fields(manifest, {"torch_version": torch.__version__, "numpy_version": np.__version__,
                             "python_version": platform.python_version()})
    for name, expected in protocol["runtime_snapshot"].items():
        if sha256_file(root / "code_snapshot" / name) != expected or sha256_file(name) != expected:
            raise ValueError("frozen execution code changed")
    for region, expected in protocol["mask_hashes"].items():
        if sha256_file(MASKS / f"huc4_{region}.npz") != expected:
            raise ValueError("geographical mask changed")
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    if args.preflight:
        preflight_replay(root, protocol, data)
        return
    if args.region:
        if args.region not in protocol["regions"] or not args.replay:
            raise ValueError("single-region audit requires a scheduled region and --replay")
        _, _, _, difference = audit_region(root, args.region, protocol, data, True)
        save_replay_receipt(root, args.region, protocol, difference)
        return
    frames, timing, choices, differences, receipts = [], [], [], [], {}
    for region in protocol["regions"]:
        cached = root / "analysis/region_replay" / f"huc4_{region}.json"
        receipt = json.loads(cached.read_text()) if cached.exists() else {}
        identity = replay_identity(root, region, protocol)
        reusable = (args.replay and receipt.get("validation_and_test_replayed") is True
                    and receipt.get("fits") == 33
                    and all(receipt.get(key) == value for key, value in identity.items()))
        frame, costs, selections, difference = audit_region(root, region, protocol, data,
                                                          args.replay and not reusable)
        if reusable:
            difference = receipt["maximum_native_difference"]
        elif args.replay:
            save_replay_receipt(root, region, protocol, difference)
        frames.append(frame)
        timing.extend(costs)
        choices.extend(selections)
        differences.append(difference)
        marker = root / "regions" / f"huc4_{region}" / "complete.json"
        receipts[str(marker)] = sha256_file(marker)
    frame = pd.concat(frames, ignore_index=True)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    timing, choices = pd.DataFrame(timing), pd.DataFrame(choices)
    timing.to_csv(out / "timing.csv", index=False)
    choices.to_csv(out / "validation_selection.csv", index=False)
    basin = pd.DataFrame([{"target_huc4": region, "model_name": alias, **metrics(part)}
        for (region, alias), part in frame.groupby(["target_huc4", "model_name"])])
    basin.to_csv(out / "per_basin_metrics.csv", index=False)
    per_run = pd.DataFrame([{"target_huc4": region, "seed": seed, "model_name": alias, **metrics(part)}
        for (region, seed, alias), part in frame.groupby(["target_huc4", "seed", "model_name"])])
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    records = []
    for alias, part in frame.groupby("model_name"):
        cost = timing[timing.model_name == alias]
        records.append({"model_name": alias, **metrics(part),
            "basin_macro_mae": float(basin.loc[basin.model_name == alias, "mae"].mean()),
            "min_parameters": int(cost.parameters.min()), "max_parameters": int(cost.parameters.max()),
            "median_training_seconds": float(cost.training_seconds.median()),
            "median_best_epoch": float(cost.best_epoch.median()), "median_epochs_run": float(cost.epochs_run.median()),
            "max_epoch_selected_fits": int((cost.best_epoch == protocol["settings"]["max_epochs"]).sum()),
            "coverage": 1.})
    summary = pd.DataFrame(records).set_index("model_name")
    reference = summary.loc["selected__reference", "mae"]
    summary["gain_vs_reference_percent"] = 100 * (reference - summary.mae) / reference
    summary.to_csv(out / "summary.csv")
    comparisons = []
    for stage, definition in STAGES.items():
        base = {"environment": "mlp", "time": "gru", "fusion": "concat"}[stage]
        for choice in definition["choices"]:
            if choice != base:
                comparisons.append((f"{stage}__{base}", f"{stage}__{choice}"))
    comparisons.extend(("selected__reference", f"selected__{stage}") for stage in STAGES)
    pairs = []
    wide = basin.pivot(index="target_huc4", columns="model_name", values="mae")
    for baseline, candidate in comparisons:
        base, other = summary.loc[baseline, "mae"], summary.loc[candidate, "mae"]
        pairs.append({"baseline": baseline, "candidate": candidate, "gain_percent": 100 * (base - other) / base,
            "basins_improved": int((wide[candidate] < wide[baseline]).sum()),
            "basin_bootstrap": interval(frame, baseline, candidate),
            "station_bootstrap": interval(frame, baseline, candidate, "station")})
    write_json(out / "paired_comparisons.json", pairs)
    plot_results(out, summary, pairs, basin)
    report = ["# DOC component comparison: experimental results", "",
        "This is a sequential internal development study, not proof of a universally optimal architecture.", "",
        (f"Completed: 165 unique fits, five whole-HUC4 regions, three seeds, {frame.cell.nunique():,} observed query cells "
        f"at {frame.station.nunique()} stations; 100% identical query coverage in every candidate."), "",
        ("MAE is in mg/L. Seed errors are averaged; predictions are not ensembled. Parent modules and final "
        "pipelines were selected solely by source-validation MAE averaged over three seeds per outer region."), "",
        "## Stage comparisons", "",
        summary[["mae", "basin_macro_mae", "rmse", "q90_mae", "min_parameters", "max_parameters"]].to_markdown(), "",
        "Positive gain means lower error. Basin intervals resample the five regions and are exploratory:", ""]
    for pair in pairs:
        lo, hi = pair["basin_bootstrap"]["gain_percent_interval"]
        report.append(f"- {pair['candidate']} vs {pair['baseline']}: {pair['gain_percent']:.2f}% gain; "
                      f"95% basin interval [{lo:.2f}, {hi:.2f}]%; improved in {pair['basins_improved']}/5 regions.")
    report.extend(["", "## Validation choices", "",
        choices[choices.selected][["target_huc4", "stage", "choice", "mean_validation_mae"]].to_markdown(index=False),
        "", "## Optimization budget", "",
        summary[["median_best_epoch", "median_epochs_run", "max_epoch_selected_fits"]].to_markdown(),
        "", ("A checkpoint selected at the 60-epoch ceiling may benefit from a longer budget. "
        "No candidate receives extra epochs after its test outcomes are seen."),
        "", "## Interpretation boundaries", "",
        ("- Environmental and time information ablations are reported even when they beat eligible models. "
        "Excluding them from the full-input selection was declared before fitting."),
        ("- Time comparisons are conditional on each region's validation-selected environmental module; "
        "fusion comparisons additionally condition on its selected time module. This does not explore every interaction."),
        ("- Actual capacity differs. Environmental conditioning and residual fusion share the same additional "
        "600 parameters; conditioning modifies the final time representation rather than recurrent gates."),
        ("- The temporal Transformer includes an extra temporal attention block and dropout; its result compares "
        "this compact recipe with the other declared recipes, not all possible temporal Transformers."),
        ("- Current-only time uses a small MLP, whereas history candidates use other operators. "
         "This is an approximate-capacity recipe comparison, not a same-operator causal isolation of history."),
        ("- The previously inspected five regions share one major river basin. These data cannot demonstrate "
        "global transfer, performance on completely independent rivers or continuous all-reach reconstruction."),
        ("- Inference retains source stations as covariate-only context alongside the target graph. "
         "Target-only graph inference on an independent new river is not tested."),
        "- Candidate comparisons are exploratory and multiple; percentile intervals have no multiplicity correction.",
        ("- Target hydrological covariates are available where observed, with masks and ages. Future prediction, "
        "a no-hydrology setting and a DOC-history assimilation setting are not tested."),
        ("- Availability differs between observed DOC query months and other months inside each station's query span. "
         "A good observed-query score does not directly establish accuracy in actual DOC gaps."),
        "- Concurrent training time reflects early stopping and shared CPU load; it is not a controlled latency comparison.",
        "", "## Audit", "",
        (f"All fit/config/input/output hashes verified. Checkpoint replay: {args.replay}; "
        f"maximum native prediction difference {max(differences):.8g}. "
        "Independent source-grid arithmetic is recorded separately in independent_metrics_audit.json.")])
    availability_path = out / "input_availability.csv"
    if availability_path.exists():
        available = pd.read_csv(availability_path)
        metadata = json.loads((out / "input_availability.meta.json").read_text())
        if metadata["dataset_sha256"] != protocol["dataset_hash"] or metadata["csv_sha256"] != sha256_file(availability_path):
            raise ValueError("input availability diagnostic changed")
        pooled = available[available.target_huc4 == "pooled"]
        report.extend(["", "## Hydrological information availability", "",
            pooled[["population", "cells", "temperature_current_fraction", "discharge_current_fraction",
                    "temperature_mean_observed_months_in_12", "discharge_mean_observed_months_in_12",
                    "no_hydrology_in_12_months_fraction"]].to_markdown(index=False), "",
            ("These diagnostics read covariate masks and query indices, not DOC values. "
            "They do not measure missing-cell accuracy or change any component selections. "
            "Lower input availability in non-query intervals is a concrete limitation of using observed-query "
            "errors to describe full reconstruction performance.")])
    (out / "research_decision.md").write_text("\n".join(report) + "\n")
    write_json(out / "audit.json", {"protocol_hash": digest(protocol), "fits": 165,
        "unique_query_cells": int(frame.cell.nunique()), "query_stations": int(frame.station.nunique()),
        "observed_query_coverage": 1., "checkpoint_replay": args.replay,
        "maximum_replay_difference": max(differences), "validation_selections_reconstructed": True,
        "hashes_verified": True, "analysis_script_sha256": sha256_file(__file__), "source_completions": receipts})
    write_json(root / "batch_complete.json", {"unique_fits": 165, "protocol_hash": digest(protocol),
               "regional_completion_hashes": receipts})
    print(summary[["mae", "gain_vs_reference_percent"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
