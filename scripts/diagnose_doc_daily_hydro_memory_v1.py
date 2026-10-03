"""Describe source-validation learning and selection; never read target products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from river_graph.experiments.unified_spatial_protocol import support_query_cells

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("off", "current_only", "full_history")
TREES = ("tree_current", "tree_history")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class Sources:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, scope=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f"changed diagnostic input: {path}")
        self.files[str(path)] = {"path": str(path), "sha256": value, "read_scope": scope}
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())


def selected_choices(collection, stage, partition, seed):
    rows = []
    for model, state in collection.items():
        if state["selection_role"] != "source_validation":
            raise ValueError("adapter selection must use source_validation")
        for k, selected in state["selection_by_k"].items():
            if stage == "direct":
                matching = [row for row in state["selection_scores"]
                            if row["k"] == int(k) and row["alpha"] == selected["alpha"]
                            and row["ridge_strength"] == selected["ridge_strength"]]
                if len(matching) != 1 or not matching[0]["valid"]:
                    raise ValueError("selected adapter score is not uniquely recorded")
                score, gamma = matching[0]["mae"], 0
            else:
                score, gamma = selected["mae"], selected["gamma"]
            rows.append({"split_seed": partition, "seed": seed, "model": model,
                         "stage": stage, "k": int(k), "alpha": selected["alpha"],
                         "ridge_strength": str(selected["ridge_strength"]),
                         "gamma": gamma, "validation_mae": score})
    return rows


def inspect_run(run, sources, *, require_exact_off):
    completion = sources.read(run / "complete.json")

    def read(name):
        return sources.read(run / name, completion["files"][name])

    config = read("config.json")
    if digest(config) != completion["config_hash"]:
        raise ValueError("configuration does not match completed package")
    partition, seed = config["split_seed"], config["seed"]
    models = {arm: read(f"{arm}.json") for arm in ARMS}
    trees = {arm: read(f"{arm}.json") for arm in TREES}
    adapters, mixers = read("adapters.json"), read("mixers.json")
    timing = read("timing.json")
    prior = Path(config["prior_run"])
    prior_completion = sources.read(prior / "complete.json", config["prior_completion_hash"])
    previous = sources.read(prior / "daily.json", prior_completion["files"]["daily.json"])
    exact = models["off"]["trace"] == previous["trace"]
    prefix = models["off"]["trace"] == previous["trace"][:len(models["off"]["trace"])]
    if require_exact_off and not exact:
        raise ValueError(f"off control differs from preceding daily trace: {run}")
    fit_rows = []
    for arm, model in models.items():
        trace = model["trace"]
        finite = all(np.isfinite(row["validation_mae"]) and
                     (row["training_loss"] is None or np.isfinite(row["training_loss"])) for row in trace)
        if not finite or model["protocol"]["selection_role"] != "source_validation":
            raise ValueError("invalid source-validation trace")
        chosen = trace[model["best_epoch"]]
        if chosen["validation_mae"] != model["validation_metrics"]["validation_mae"]:
            raise ValueError("selected checkpoint does not match the trace")
        fit_rows.append({"split_seed": partition, "seed": seed, "arm": arm,
                         "epochs_run": model["epochs_run"], "best_epoch": model["best_epoch"],
                         "epoch_ceiling": model["config"]["epochs"],
                         "selected_scale": model["selected_scale"],
                         "validation_mae": chosen["validation_mae"],
                         "context_validation_mae": trace[0]["validation_mae"],
                         "cap_limited": (model["epochs_run"] == model["config"]["epochs"]
                                         and trace[-1]["stale_epochs"] < model["config"]["patience"]),
                         "trainable_parameters": model["trainable_parameter_count"],
                         "projection_parameters": model.get("hydro_projection_trainable_parameter_count", 0),
                         "projection_norm": model.get("hydro_projection_parameter_norm", 0),
                         "projection_distance": model.get("hydro_projection_parameter_distance", 0),
                         "temporal_distance": model["temporal_parameter_distance"],
                         "spatial_distance": model["spatial_parameter_distance"],
                         "off_parent_trace_exact": exact if arm == "off" else None,
                         "off_parent_trace_prefix": prefix if arm == "off" else None})

    with np.load(sources.bind(Path(config["mask_path"]), config["mask_hash"]), allow_pickle=False) as archive:
        split = {name: archive[name] for name in archive.files}
    dataset_path = sources.bind(Path(config["dataset_path"]), config["dataset_hash"],
                                "Only station/month dimensions and fixed validation target cells")
    dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
    months = len(dataset["months"])
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    labels = dataset["y"].reshape(-1)[torch.as_tensor(query)]
    truth = np.asarray(labels, dtype=np.float64)
    del dataset, labels
    columns = ["cell", "station", "visibility_role", "context_pred", "ecological_memory",
               "daily_numeric_valid_count", "daily_history_valid_months", "daily_history_possible_months"]
    columns += [f"{arm}_pred" for arm in (*ARMS, *TREES)]
    columns += [f"{arm}_integrated_k0_pred" for arm in ARMS]
    product = sources.bind(run / "full_grid.parquet", completion["files"]["full_grid.parquet"],
                           "Parquet predicate: fixed source-validation query cells only")
    frame = pq.read_table(product, columns=columns, filters=[("cell", "in", query.tolist())]).to_pandas()
    frame = frame.set_index("cell").loc[query]
    if len(frame) != len(query) or not frame.index.is_unique or set(frame.visibility_role) != {"val"}:
        raise ValueError("source-validation full-grid view is misaligned")
    prior_product = sources.bind(prior / "full_grid.parquet", prior_completion["files"]["full_grid.parquet"],
                                 "Parquet predicate: fixed source-validation query cells only")
    prior_frame = pq.read_table(prior_product, columns=["cell", "daily_pred"],
                                filters=[("cell", "in", query.tolist())]).to_pandas().set_index("cell").loc[query]
    same_predictions = np.array_equal(frame.off_pred, prior_frame.daily_pred)
    if require_exact_off and not same_predictions:
        raise ValueError("off control validation prediction differs from parent daily model")
    saved = pd.read_csv(sources.bind(run / "source_validation.csv", completion["files"]["source_validation.csv"]))
    count = frame.daily_numeric_valid_count.to_numpy()
    history = frame.daily_history_valid_months.to_numpy()
    groups = {"all": np.ones(len(query), dtype=bool), "none_valid": count == 0,
              "some_valid": count > 0, "all_valid": count == 3,
              "history_0": history == 0, "history_1_5": (history >= 1) & (history <= 5),
              "history_6_11": (history >= 6) & (history <= 11), "history_12": history == 12}
    metric_rows = []
    for arm in ("context", *ARMS, *TREES):
        for stage in (("direct", "integrated") if arm in ARMS else ("direct",)):
            column = f"{arm}_pred" if stage == "direct" else f"{arm}_integrated_k0_pred"
            gamma = 0 if stage == "direct" else mixers[f"{arm}_integrated_constant"]["gamma_k0"]
            if stage == "integrated":
                base = frame[f"{arm}_pred"].to_numpy()
                expected = base if gamma == 0 else np.maximum(
                    0, frame.context_pred.to_numpy() + (1-gamma)*(base-frame.context_pred.to_numpy())
                    + gamma*frame.ecological_memory.to_numpy())
                np.testing.assert_array_equal(expected, frame[column])
            for group, selected in groups.items():
                recorded = saved[(saved.arm == arm) & (saved.stage == stage) & (saved.group == group)]
                if len(recorded) != 1 or recorded.iloc[0]["n"] != selected.sum():
                    raise ValueError("recorded validation group count differs")
                if not selected.any():
                    continue
                error = frame[column].to_numpy()[selected] - truth[selected]
                mae = float(np.abs(error).mean())
                np.testing.assert_allclose(mae, recorded.iloc[0].mae, rtol=1e-12, atol=1e-12)
                metric_rows.append({"split_seed": partition, "seed": seed, "arm": arm, "stage": stage,
                                    "group": group, "n_query": int(selected.sum()),
                                    "n_stations": frame.station[selected].nunique(), "mae": mae,
                                    "bias_pred_minus_true": float(error.mean()), "gamma_k0": gamma})
    tree_rows = []
    for arm, record in trees.items():
        if record["hyperparameter_search"] or record["replaces_neural_context"]:
            raise ValueError("tree probe protocol changed")
        actual = next(row["mae"] for row in metric_rows
                      if row["arm"] == arm and row["stage"] == "direct" and row["group"] == "all")
        np.testing.assert_allclose(actual, record["validation_mae_native"], rtol=1e-12, atol=1e-12)
        tree_rows.append({"split_seed": partition, "seed": seed, "arm": arm,
                          "n_features": record["n_features"], "validation_mae": actual,
                          "n_estimators": record["forest_parameters"]["n_estimators"],
                          "min_samples_leaf": record["forest_parameters"]["min_samples_leaf"]})
    choice_rows = (selected_choices(adapters, "direct", partition, seed)
                   + selected_choices(mixers, "integrated", partition, seed))
    return fit_rows, metric_rows, tree_rows, choice_rows, {
        "split_seed": partition, "seed": seed, "off_parent_trace_exact": exact,
        "off_parent_trace_prefix": prefix, "off_parent_validation_predictions_exact": same_predictions,
        "elapsed_seconds": timing["elapsed_seconds"]}


def counts(series):
    return "; ".join(f"{value}: {count}" for value, count in series.value_counts().sort_index().items())


def report(root, frames, *, complete):
    fits, metrics, _trees, choices, checks = frames
    curves = choices.groupby(["model", "stage", "k", "split_seed"]).validation_mae.mean().groupby(
        ["model", "stage", "k"]).mean().reset_index()
    summary = metrics.groupby(["arm", "stage", "group", "split_seed"])[["mae", "bias_pred_minus_true"]].mean().groupby(
        ["arm", "stage", "group"]).mean().reset_index()
    lines = ["# Does daily hydrology improve the recurrent state?", "",
             ("All nine formal packages are complete." if complete else "Partial execution diagnostic; formal nine-run evidence is incomplete."),
             ("Only source-validation queries, training traces and saved validation choices are used here. "
              "The validation set also selected checkpoints and calibration coefficients; these are development diagnostics, not independent transfer estimates."), "",
             ("The off control retains daily descriptors at the scalar head. Current-only adds their zero-initialized projection at the final GRU step; full-history adds it at every valid causal step. "
              "The two active modes add the same 512 parameters at hidden size 64. Tree probes use the same source labels and daily-history slots with the preceding forest settings; their predictions do not replace the neural model's fixed context base."), "",
             "## Training", "",
             "| Arm | Validation MAE | Selected epochs | Epochs run | Scale counts | Projection norm | Cap-limited |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for arm in ARMS:
        g = fits[fits.arm == arm]
        lines.append(f"| {arm} | {g.groupby('split_seed').validation_mae.mean().mean():.6f} | "
                     f"{g.best_epoch.min()}–{g.best_epoch.max()} | {g.epochs_run.min()}–{g.epochs_run.max()} | "
                     f"{counts(g.selected_scale)} | {g.projection_norm.min():.5f}–{g.projection_norm.max():.5f} | {int(g.cap_limited.sum())}/{len(g)} |")
    lines += ["", (f"Off-control traces are exactly equal to the preceding daily-head model in {int(checks.off_parent_trace_exact.sum())}/{len(checks)} packages. "
              f"The corresponding fixed validation predictions are equal in {int(checks.off_parent_validation_predictions_exact.sum())}/{len(checks)}. "
              f"All recorded losses are finite. Recorded package durations total {checks.elapsed_seconds.sum():.1f} seconds, including fitting, inference and saved products."), "",
              "## K0 errors by daily information", "",
              "| Stage | Group | Off MAE | Current-only MAE | Full-history MAE | Full−current | Full-history bias |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in ("direct", "integrated"):
        for group in ("all", "none_valid", "some_valid", "history_0", "history_1_5", "history_6_11", "history_12"):
            g = summary[(summary.stage == stage) & (summary.group == group)].set_index("arm")
            if not all(arm in g.index for arm in ARMS):
                continue
            lines.append(f"| {stage} | {group} | {g.loc['off','mae']:.6f} | {g.loc['current_only','mae']:.6f} | "
                         f"{g.loc['full_history','mae']:.6f} | {g.loc['full_history','mae']-g.loc['current_only','mae']:+.6f} | "
                         f"{g.loc['full_history','bias_pred_minus_true']:+.6f} |")
    lines += ["", "Native MAE and signed bias are in mg/L; positive bias means overprediction. Daily validity comes from the frozen data product. History groups count months with at least one valid descriptor in the causal 12-month window, including the current month. Sparse or empty groups are descriptive; their counts are saved by partition. Each displayed result first averages seeds within partition, then gives available partitions equal weight.", "",
              "## Partition directions", "",
              "| Stage | Partition | Current−off MAE | Full−off MAE | Full−current MAE | Full improves current seeds |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in ("direct", "integrated"):
        for partition, g in metrics[(metrics.stage == stage) & (metrics.group == "all") & metrics.arm.isin(ARMS)].groupby("split_seed"):
            p = g.pivot(index="seed", columns="arm", values="mae")
            lines.append(f"| {stage} | {partition} | {(p.current_only-p.off).mean():+.6f} | {(p.full_history-p.off).mean():+.6f} | "
                         f"{(p.full_history-p.current_only).mean():+.6f} | {int((p.full_history<p.current_only).sum())}/{len(p)} |")
    lines += ["", "## Selected validation K curves", "",
              "| Model | K=0 | K=1 | K=3 | K=5 |", "| --- | ---: | ---: | ---: | ---: |"]
    for model in [f"{arm}_gru_tuned_anchor" for arm in ("context", *ARMS, *TREES)] + [f"{arm}_integrated_gru_tuned_anchor" for arm in ARMS]:
        g = curves[curves.model == model].set_index("k")
        lines.append("| " + model + " | " + " | ".join(f"{g.loc[k,'validation_mae']:.6f}" for k in (0, 1, 3, 5)) + " |")
    lines += ["", ("Both constant and temporal support-adapter selections are preserved in the CSV; the table shows the existing temporal basis. Tree-history versus tree-current asks whether these same historical covariates help an explicit-feature model. "
              "A neural history gain alone does not establish a new river-transport mechanism."), "",
              "## Integrated K5 choices", "",
              "| Mode | Gamma counts | Alpha counts | Ridge counts |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        g = choices[(choices.model == f"{arm}_integrated_gru_tuned_anchor") & (choices.k == 5)]
        lines.append(f"| {arm} | {counts(g.gamma)} | {counts(g.alpha)} | {counts(g.ridge_strength)} |")
    lines += ["", "## What this recurrent state represents", "",
              ("The existing DOC-age/support decay acts on the complete previous hidden state, including hydrologic information. "
               "At a station with no visible DOC, the original age feature increases with elapsed calendar months rather than with the age of discharge measurements. "
               "The experiment therefore tests daily hydrology within the existing observation-conditioned recurrence; it does not create a separately timed hydrologic store. "
               "A nonzero projection norm establishes that the branch was trained, not that its weights are physical transport coefficients."), "",
              "No feature, route, epoch budget or model is changed by this diagnostic. Spatial transfer and high-DOC tradeoffs are evaluated separately on the retained target outputs."]
    output = root / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in zip(("training", "groups", "trees", "choices", "checks"), frames):
        table.to_csv(output / f"source_validation_{name}.csv", index=False)
    summary.to_csv(output / "source_validation_group_summary.csv", index=False)
    curves.to_csv(output / "source_validation_selected_curves.csv", index=False)
    (output / "source_validation.md").write_text("\n".join(lines) + "\n")
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Smoke/incomplete source-only report, clearly labeled")
    args = parser.parse_args()
    expected = [args.root / "runs" / f"split{p}_seed{s}" for p in (142, 143, 144) for s in (42, 43, 44)]
    runs = [run for run in expected if (run / "complete.json").exists()]
    complete = len(runs) == 9
    if not runs or (not complete and not args.allow_partial):
        raise ValueError("all nine formal packages required unless --allow-partial is explicit")
    sources = Sources()
    sources.bind(Path(__file__))
    sources.bind(Path("src/river_graph/experiments/graph_upgrade_v2.py"),
                 scope="Original observation-age definition used in the architecture interpretation")
    collected = [[], [], [], [], []]
    for run in runs:
        result = inspect_run(run, sources, require_exact_off=complete)
        for i, rows in enumerate(result):
            collected[i].extend(rows if isinstance(rows, list) else [rows])
    frames = tuple(pd.DataFrame(rows) for rows in collected)
    curves = report(args.root, frames, complete=complete)
    (args.root / "diagnostics" / "source_validation_sources.json").write_text(json.dumps({
        "complete": complete, "scope": "fixed source-validation data and fitting summaries only; no target comparison products",
        "sources": list(sources.files.values())}, indent=2) + "\n")
    print(curves[(curves.k.isin([0, 5])) & curves.model.str.endswith("gru_tuned_anchor")].to_string(index=False))
    print(f"Completed source-only diagnostic: {len(runs)} packages; cap-limited fits={int(frames[0].cap_limited.sum())}")


if __name__ == "__main__":
    main()
