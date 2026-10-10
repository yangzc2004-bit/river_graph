"""Rebuild the paper evidence from saved predictions, without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("experiments/phase4_transfer")
TIME = BASE / "doc_current_source_temporal_v3_r1"
SPACE = BASE / "doc_current_availability_attention_geographical_v1"
EXTERNAL = BASE / "doc_current_source_external_v2"
OUTPUT = Path("docs/paper/doc_spatiotemporal_consolidation_v1")
TABLES = Path("docs/paper/latex/tables/spatiotemporal_v4")
TASKS = {
    "e2a_strict": "Unobserved periods",
    "e2b_partial": "Observation-assisted periods",
    "geographical": "Unmonitored regions",
    "external": "External application",
}
MAIN_MODELS = {
    "e2a_strict": ("station_hidden_trees", "upgraded_fusion", "available_fusion"),
    "e2b_partial": ("station_hidden_trees", "upgraded_fusion", "available_fusion"),
    "geographical": ("station_hidden_trees", "unmonitored_integrated", "available_real_integrated"),
    "external": ("station_hidden_trees", "unmonitored_integrated", "available_real_integrated"),
}
METRICS = ("mae", "rmse", "r2", "bias", "log_mae", "station_equal_mae", "q90_mae")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_metrics(frame, threshold):
    """Cell errors in one fitted task; stations remain the sampling units."""
    if frame.empty or frame.duplicated("cell").any():
        raise ValueError("one prediction per nonempty task cell is required")
    truth, prediction = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    if not np.isfinite(np.r_[truth, prediction]).all() or (np.r_[truth, prediction] < 0).any():
        raise ValueError("DOC truth and prediction must be finite and nonnegative")
    error = prediction - truth
    denominator = np.square(truth - truth.mean()).sum()
    tail = truth >= threshold
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else np.nan,
        "bias": float(error.mean()),
        "log_mae": float(np.abs(np.log1p(prediction) - np.log1p(truth)).mean()),
        "station_equal_mae": float(frame.assign(error=np.abs(error)).groupby("station").error.mean().mean()),
        "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
        "q90_n": int(tail.sum()), "n_cells": len(frame), "n_stations": frame.station.nunique(),
    }


def assert_aligned(panel):
    """Require the same station/month/truth population for every model and seed."""
    if panel.duplicated(["split_seed", "seed", "model_name", "cell"]).any():
        raise ValueError("duplicate model/seed/query identities")
    for _, split in panel.groupby("split_seed"):
        reference = None
        for _, group in split.groupby(["seed", "model_name"]):
            identity = group.sort_values("cell")[["cell", "station", "month", "y_true"]].reset_index(drop=True)
            if reference is None:
                reference = identity
            else:
                pd.testing.assert_frame_equal(reference, identity, check_dtype=False, check_exact=True)


def aggregate_metrics(runs):
    """Mean seed metrics inside a region, then equal-region metrics."""
    regions = runs.groupby(["split_seed", "model_name"], as_index=False)[list(METRICS)].mean()
    return regions, regions.groupby("model_name", as_index=False)[list(METRICS)].mean()


def compare_summary(actual, saved, keys):
    saved = saved.rename(columns={"signed_bias": "bias"})
    merged = actual.merge(saved, on=keys, suffixes=("_rebuilt", "_saved"), validate="one_to_one")
    if len(merged) != len(actual):
        raise ValueError("saved summary omits a rebuilt result")
    for metric in METRICS:
        np.testing.assert_allclose(merged[metric + "_rebuilt"], merged[metric + "_saved"],
                                   rtol=1e-10, atol=1e-10, equal_nan=True)


def load_internal(root, *, temporal, inputs):
    expected = ({f"{mask}_seed{seed}" for mask in ("e2a_strict", "e2b_partial") for seed in (42, 43, 44)}
                if temporal else {f"huc4_{region}_seed{seed}" for region in ("1013", "1019", "0708", "1030", "1101")
                                  for seed in (42, 43, 44, 45, 46)})
    completions = sorted((root / "runs").glob("*/complete.json"))
    if {p.parent.name for p in completions} != expected:
        raise ValueError("the fixed paper task set is incomplete")
    frames, records = [], []
    for completion in completions:
        run = completion.parent
        config = json.loads((run / "config.json").read_text())
        frame = pd.read_parquet(run / "predictions.parquet")
        if not frame.visibility_role.eq("test").all() or not frame.k.eq(0).all():
            raise ValueError("primary internal rows must be K0 test rows")
        if set(frame.model_name) != set(config["models"]):
            raise ValueError("saved prediction model set changed")
        frame["split_seed"] = 1 if temporal and config["mask"] == "e2a_strict" else (
            2 if temporal else int(config["split_seed"]))
        frame["task_id"] = config["mask"] if temporal else "geographical"
        for name, group in frame.groupby("model_name"):
            records.append({"task_id": frame.task_id.iloc[0], "split_seed": int(frame.split_seed.iloc[0]),
                            "seed": config["seed"], "model_name": name,
                            **compute_metrics(group, config["q90_threshold_train"])})
        frames.append(frame)
        for path in (completion, run / "config.json", run / "predictions.parquet", run / "predictions.meta.json"):
            inputs[str(path)] = file_hash(path)
    panel, runs = pd.concat(frames, ignore_index=True), pd.DataFrame(records)
    assert_aligned(panel)
    summaries, regions = [], []
    for task, rows in runs.groupby("task_id"):
        region, summary = aggregate_metrics(rows)
        region["task_id"], summary["task_id"] = task, task
        regions.append(region)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    saved_path = root / "analysis" / ("summary.csv" if temporal else "primary_summary.csv")
    saved = pd.read_csv(saved_path).rename(columns={"mask": "task_id"})
    compare_summary(summary, saved, ["task_id", "model_name"] if temporal else ["model_name"])
    inputs[str(saved_path)] = file_hash(saved_path)
    return panel, runs, pd.concat(regions, ignore_index=True), summary


def load_external(inputs):
    path = EXTERNAL / "predictions.parquet"
    meta = json.loads((EXTERNAL / "predictions.meta.json").read_text())
    if file_hash(path) != meta["prediction_sha256"]:
        raise ValueError("external prediction content no longer matches its sidecar")
    full = pd.read_parquet(path)
    primary = full[full.population.eq("all_observed_k0") & full.k.eq(0)].copy()
    primary["split_seed"], primary["task_id"] = 2040104, "external"
    assert_aligned(primary)
    saved = pd.read_csv(EXTERNAL / "analysis/metrics.csv")
    rows = []
    for name, group in primary.groupby("model_name"):
        threshold = saved[saved.population.eq("all_observed_k0") & saved.model_name.eq(name)].q90_threshold_source.iloc[0]
        rows.append({"task_id": "external", "model_name": name, **compute_metrics(group, threshold)})
    summary = pd.DataFrame(rows)
    compare_summary(summary, saved[saved.population.eq("all_observed_k0")], ["model_name"])
    for source_file in (path, EXTERNAL / "predictions.meta.json", EXTERNAL / "analysis/metrics.csv"):
        inputs[str(source_file)] = file_hash(source_file)
    return full, primary, summary


def selected_effect(panel, task, candidate, reference, saved, draws):
    from analyze_doc_source_retrieval_v1 import paired
    from analyze_unified_doc_spatial import joint_station_bootstrap

    result = joint_station_bootstrap(paired(panel, candidate, reference), draws=draws)
    for key in ("reference_mae", "candidate_mae", "relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
        np.testing.assert_allclose(result[key], saved[key], rtol=1e-10, atol=1e-10)
    return {"task_id": task, "candidate": candidate, "reference": reference, **result}


def latex_table(frame, columns, formats):
    lines = []
    for row in frame.to_dict("records"):
        values = [formats.get(name, lambda x: str(x))(row[name]) for name in columns]
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines) + "\n" + r"\bottomrule" + "\n"


def supplementary_tables(summary, diagnostics, effects):
    """Write compact paper tables, retaining code IDs only in reproduction maps."""
    def display(name):
        aliases = {"station_hidden_trees": "Strong environmental trees", "current_model": "Earlier full model",
            "matched_daily_trees": "Ordinary daily trees", "context_trees": "Context trees",
            "unmonitored_integrated": "Previous complete method", "unmonitored_residual": "Previous native residual",
            "available_real_integrated": "Current complete method", "available_real_native": "Current native residual",
            "available_fusion": "Current complete method", "available_native": "Current native residual",
            "upgraded_fusion": "Previous complete method", "upgraded_native": "Previous native residual",
            "current_fusion": "Earlier complete fusion", "current_native": "Earlier native residual",
            "original_hybrid": "Original hybrid", "innovation_real_trees": "Aggregate-source trees"}
        if name in aliases:
            return aliases[name]
        family, control, version = name.rsplit("_", 2)
        families = {"attention": "Source anomaly attention", "relative": "Relative source anomaly",
                    "level": "Full source innovation", "innovation": "Source innovation",
                    "available": "Expanded current access"}
        controls = {"real": "actual", "fixed": "uniform", "historical": "historical", "seasonal": "seasonal"}
        return f"{families.get(family, family)} / {controls.get(control, control)} / {version}"

    numeric = {name: lambda value: f"{value:.3f}" for name in ("mae", "rmse", "r2", "bias", "q90_mae", "station_equal_mae")}
    for task in TASKS:
        frame = summary[summary.task_id.eq(task)].copy()
        frame["display"] = frame.model_name.map(display)
        (TABLES / f"supp_{task}.tex").write_text(latex_table(frame,
            ["display", "mae", "rmse", "r2", "bias"], numeric))
    external = diagnostics[diagnostics.population.eq("all_observed_k0") & diagnostics.model_name.isin(MAIN_MODELS["external"])].copy()
    external["display"] = external.model_name.map(display)
    (TABLES / "supp_external_diagnostics.tex").write_text(latex_table(external,
        ["display", "station_equal_mae", "q90_mae", "coverage", "width_median"],
        {**numeric, "coverage": lambda x: f"{100*x:.2f}", "width_median": lambda x: f"{x:.3f}"}))
    contrasts = effects.copy()
    contrasts["scenario"] = contrasts.task_id.map(TASKS)
    contrasts["comparison"] = contrasts.apply(lambda row: f"{display(row.candidate)} vs {display(row.reference)}", axis=1)
    (TABLES / "supp_contrasts.tex").write_text(latex_table(contrasts,
        ["scenario", "comparison", "relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"],
        {name: lambda x: f"{x:.3f}" for name in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws != 5000:
        raise ValueError("the paper preserves the existing 5,000-draw comparisons")
    output, inputs = args.output, {}
    old_paper = Path("docs/paper/latex/unmonitored_doc_draft_v3.tex")
    inputs[str(old_paper)] = file_hash(old_paper)
    time_panel, time_runs, time_regions, time_summary = load_internal(TIME, temporal=True, inputs=inputs)
    space_panel, space_runs, space_regions, space_summary = load_internal(SPACE, temporal=False, inputs=inputs)
    external_full, external_panel, external_summary = load_external(inputs)
    summary = pd.concat([time_summary, space_summary, external_summary], ignore_index=True)
    rows, effects = [], []
    for task, models in MAIN_MODELS.items():
        panel = time_panel[time_panel.task_id.eq(task)] if task.startswith("e2") else (
            space_panel if task == "geographical" else external_panel)
        labels = dict(zip(models, ("Strong environmental trees", "Previous complete method", "Current complete method"), strict=True))
        identities = panel[["cell", "station"]].drop_duplicates()
        for name in models:
            record = summary[summary.task_id.eq(task) & summary.model_name.eq(name)].iloc[0].to_dict()
            rows.append({**record, "scenario": TASKS[task], "model_label": labels[name],
                         "n_cells_unique": len(identities), "n_stations_unique": identities.station.nunique(),
                         "members": 3 if task.startswith("e2") else 5,
                         "estimand": "mean seed errors" if task.startswith("e2") else (
                             "equal-region mean of seed errors" if task == "geographical" else "error of five-member mean prediction")})
        effect_path = (TIME if task.startswith("e2") else SPACE if task == "geographical" else EXTERNAL) / "analysis/paired_effects.csv"
        saved_effects = pd.read_csv(effect_path)
        if task.startswith("e2"):
            saved_effects = saved_effects[saved_effects["mask"].eq(task)]
        else:
            saved_effects = saved_effects[saved_effects.population.eq("all_observed_k0") & saved_effects.k.eq(0) & saved_effects.zone.eq("overall")]
        comparisons = [(models[-1], models[0]), (models[-1], models[1])]
        if task.startswith("e2"):
            comparisons.append(("available_native", "station_hidden_trees"))
        if task == "geographical":
            comparisons += [(models[-1], "available_seasonal_integrated"), (models[-1], "level_real_integrated"),
                            (models[-1], "relative_real_integrated"), ("available_real_native", "available_seasonal_native")]
        for candidate, reference in comparisons:
            old = saved_effects[saved_effects.candidate.eq(candidate) & saved_effects.reference.eq(reference)]
            if len(old) != 1:
                raise ValueError("missing unique saved primary contrast")
            effects.append(selected_effect(panel, task, candidate, reference, old.iloc[0], args.bootstrap_draws))
        inputs[str(effect_path)] = file_hash(effect_path)
    main_results = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("main_results", main_results), ("all_model_metrics", summary), ("paired_effects", pd.DataFrame(effects)),
                        ("internal_run_metrics", pd.concat([time_runs, space_runs], ignore_index=True)),
                        ("internal_region_metrics", pd.concat([time_regions, space_regions], ignore_index=True))):
        frame.to_csv(output / f"{name}.csv", index=False)
    curves = pd.read_csv(SPACE / "analysis/curves_summary.csv")
    space_curves = curves[curves.model_name.isin(MAIN_MODELS["geographical"])].copy()
    space_curves["task_id"] = "geographical"
    ext_saved = pd.read_csv(EXTERNAL / "analysis/metrics.csv")
    ext_curves = ext_saved[ext_saved.population.eq("fixed_query_curve") & ext_saved.model_name.isin(MAIN_MODELS["external"])].copy()
    ext_curves["task_id"] = "external"
    pd.concat([space_curves, ext_curves], ignore_index=True).to_csv(output / "support_curves.csv", index=False)
    ext_saved.to_csv(output / "external_diagnostics.csv", index=False)
    for study, filename in ((SPACE, "geographical_strata"), (EXTERNAL, "external_strata")):
        path = study / "analysis" / ("strata_summary.csv" if study == SPACE else "strata.csv")
        pd.read_csv(path).to_csv(output / f"{filename}.csv", index=False)
        inputs[str(path)] = file_hash(path)
    # Validate fixed queries across K and keep primary K0 distinct from curve K0.
    query = external_full[external_full.population.eq("fixed_query_curve")]
    for _, group in query.groupby("model_name"):
        ids = [set(part.cell) for _, part in group.groupby("k")]
        if len(ids) != 4 or any(cells != ids[0] for cells in ids[1:]):
            raise ValueError("external support curve changes its query cells")
    for completion in sorted((SPACE / "runs").glob("*/complete.json")):
        path = completion.parent / "support_curves.parquet"
        curve = pd.read_parquet(path)
        for _, group in curve.groupby("model_name"):
            ids = [set(part.cell) for _, part in group.groupby("k")]
            if len(ids) != 4 or any(cells != ids[0] for cells in ids[1:]):
                raise ValueError("geographical support curve changes its query cells")
        inputs[str(path)] = file_hash(path)
    inputs[str(SPACE / "analysis/curves_summary.csv")] = file_hash(SPACE / "analysis/curves_summary.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    numeric = {name: lambda value: f"{value:.3f}" for name in ("mae", "rmse", "r2", "bias")}
    main_display = main_results.copy()
    main_display.loc[main_display.scenario.duplicated(), "scenario"] = ""
    (TABLES / "main_results.tex").write_text(latex_table(main_display,
        ["scenario", "model_label", "mae", "rmse", "r2", "bias"], numeric))
    for task, prefix in (("geographical", "space"), ("e2a_strict", "time_a"), ("e2b_partial", "time_b")):
        current = MAIN_MODELS[task][-1]
        row = summary[summary.task_id.eq(task) & summary.model_name.eq(current)].iloc[0]
        (TABLES / f"{prefix}_current.tex").write_text(f"{row.mae:.3f}")
    supplementary_tables(summary, ext_saved, pd.DataFrame(effects))
    archive = [
        ("Core", "Temporal reconstruction", str(TIME), "Two tasks, three seeds; fitted fusion and native predictions distinguished"),
        ("Core", "Geographical reconstruction", str(SPACE), "Five regions, five seeds; no receiving chemistry"),
        ("Core", "External application", str(EXTERNAL), "Fixed ensemble; basin seen for previous release"),
        ("Supplement", "Dynamic upstream message", str(BASE / "doc_hydro_river_expansion_v1"), "Small source-development gain; not adopted in release"),
        ("Supplement", "Confluence and storage", str(BASE / "doc_confluence_storage_v1"), "No additional material gain"),
        ("Supplement", "Sampling date and flow phase", str(BASE / "doc_sampling_river_v1"), "No additional material gain"),
        ("Archive", "Real river form typology", str(BASE / "doc_river_planform_typology_v1"), "Three broad types; continuous morphology"),
        ("Archive", "Whole-form DOC synthesis", str(BASE / "doc_river_wholeform_evidence_v1"), "Branch organization information; no stable footprint ranking"),
        ("Archive", "Independent river-form observations", str(BASE / "doc_river_neon_form_validation_v1"), "Primary replication unresolved; conditional nonoverlap signal"),
        ("Archive", "Prior uncertainty ranking", "docs/paper/phase3_r2_ranking_spec.md", "Keep prior failure branch; no active-monitoring claim"),
        ("Archive", "Cross-analyte and chemical adaptation", str(BASE / "doc_nested_confirmation_v1"), "Separate completed exploration; not a contribution of this manuscript"),
    ]
    pd.DataFrame(archive, columns=["placement", "topic", "source", "decision"]).to_csv(output / "evidence_ledger.csv", index=False)
    for _, _, source, _ in archive:
        path = Path(source)
        if path.is_dir():
            path = path / "research_decision.md"
        if path.is_file():
            inputs[str(path)] = file_hash(path)
    counts = {"temporal_fits": 6, "geographical_fits": 25, "external_members": 5,
              "main_result_rows": len(main_results), "recomputed_contrasts": len(effects)}
    # Inputs are checked again after analysis; old predictions and v3 stay intact.
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise ValueError(f"analysis input changed during consolidation: {path}")
    manifest = {"purpose": "Paper consolidation only; no fitting or model selection", "counts": counts,
                "bootstrap_draws": 5000, "bootstrap_seed": 42, "inputs": inputs,
                "generator_sha256": file_hash(__file__), "verification": "point metrics and primary intervals match saved analyses",
                "task_weighting": {"time": "three seed errors, separately per task",
                    "space": "five seeds within region, five regions equal", "external": "fixed five-member prediction mean"},
                "model_changes": False, "new_training": False}
    (output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(main_results[["scenario", "model_label", "mae", "rmse", "r2", "bias"]].to_string(index=False))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
