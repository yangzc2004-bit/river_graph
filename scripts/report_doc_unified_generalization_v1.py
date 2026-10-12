"""Evidence-bound diagnosis, transfer comparisons and selected-model limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unified_generalization_v1")
REPORT = Path("docs/doc_unified_generalization_results_20261012.md")
LABELS = {"reference": "Matched original model", "episode_cells": "Isolated basin graphs",
    "episode_balanced": "Isolated + basin-balanced", "balanced_no_coords": "Balanced + neutral coordinates",
    "balanced_hydro_dropout": "Balanced + hydro missingness augmentation",
    "balanced_gated": "Balanced + reliability-gated river messages", "combined": "All proposed changes combined"}


def paired_interval(recipe, population, metric, draws=3000):
    panels = {}
    for domain in ("val05_test10", "val10_test07", "val07_test05"):
        for seed in (42, 43, 44):
            parts = [pd.read_parquet(ROOT/"fits"/f"{domain}_{r}_s{seed}"/f"{population}_predictions.parquet")
                for r in (recipe, "reference")]
            if not np.array_equal(parts[0].cell, parts[1].cell) or not np.array_equal(parts[0].y_true, parts[1].y_true):
                raise ValueError("unmatched paired queries")
            errors = []
            for part in parts:
                y, p = part.y_true.to_numpy(dtype=float), part.y_pred.to_numpy(dtype=float)
                errors.append(np.abs(p-y) if metric == "mae" else np.abs(np.log1p(p)-np.log1p(y)))
            panels[(domain, seed)] = pd.DataFrame({"station": parts[0].station,
                "delta": errors[0]-errors[1], "count": 1}).groupby("station").agg({"delta": "sum", "count": "sum"})
    stations = sorted(set().union(*(set(frame.index) for frame in panels.values())))
    difference = np.stack([p.reindex(stations, fill_value=0).delta.to_numpy() for p in panels.values()])
    counts = np.stack([p.reindex(stations, fill_value=0)["count"].to_numpy() for p in panels.values()])
    rng = np.random.default_rng(126511)
    sampled = []
    for _ in range(draws):
        ids = rng.integers(len(stations), size=len(stations))
        denominator = counts[:, ids].sum(1)
        if np.all(denominator > 0):
            sampled.append(float((difference[:, ids].sum(1)/denominator).mean()))
    return {"recipe": recipe, "population": population, "metric": metric,
        "delta": float((difference.sum(1)/counts.sum(1)).mean()), "ci95": np.quantile(sampled, [.025, .975]).tolist(),
        "station_blocks": len(stations), "draws": len(sampled),
        "scope": "exploratory station blocks in fixed river datasets; shared source fits and hydrological correlation remain; not global river-population inference"}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    if not audit["passed"]:
        raise ValueError("audit not passed")
    for name, expected in audit["files"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError("audited data changed")
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    detail = pd.read_csv(ROOT/"analysis/by_fit.csv")
    selection = json.loads((ROOT/"selected_model.json").read_text())
    selected = selection["selected_recipe"]
    diagnosis = pd.read_csv(ROOT/"diagnosis/summary.csv")
    stats = detail.groupby(["population", "recipe"])[["mae", "log1p_mae", "r2"]].agg(["min", "max", "std"])
    stats.to_csv(ROOT/"analysis/optimization_variability.csv")
    pairs = detail.merge(detail[detail.recipe == "reference"][["domain", "seed", "population", "mae", "log1p_mae"]],
        on=["domain", "seed", "population"], suffixes=("", "_ref"), validate="many_to_one")
    pairs["native_win"] = pairs.mae < pairs.mae_ref
    pairs["log_win"] = pairs.log1p_mae < pairs.log1p_mae_ref
    pairs["joint_win"] = pairs.native_win & pairs.log_win
    pairs["positive_r2"] = pairs.r2 > 0
    wins = pairs.groupby(["population", "recipe"])[["native_win", "log_win", "joint_win", "positive_r2"]].sum().reset_index()
    wins.to_csv(ROOT/"analysis/panel_consistency.csv", index=False)
    intervals = [paired_interval(selected, population, metric) for population in ("source", "development", "confirmation")
        for metric in ("mae", "log1p_mae")]
    (ROOT/"analysis/paired_intervals.json").write_text(json.dumps(intervals, indent=2)+"\n")
    source = summary[(summary.population == "source") & (summary.recipe == selected)].iloc[0]
    development = summary[(summary.population == "development") & (summary.recipe == selected)].iloc[0]
    confirmation = summary[(summary.population == "confirmation") & (summary.recipe == selected)].iloc[0]
    primary = detail[detail.id == selection["primary_checkpoint_id"]]
    decision = {"selected_recipe": selected, "model": "environmental MLP + causal GRU + reliability-gated upstream/downstream/global Transformer",
        "training": "isolated source HUC2 graphs + equal total basin loss weight", "source_validation_only": True,
        "primary_checkpoint": selection["primary_checkpoint_id"], "parameters": 12571,
        "source_mae_improvement_percent": float(source.mae_improvement_percent),
        "development_mae_improvement_percent": float(development.mae_improvement_percent),
        "confirmation_mae_improvement_percent": float(confirmation.mae_improvement_percent),
        "confirmation_log_mae_improvement_percent": float(confirmation.log1p_mae_improvement_percent),
        "development_r2": float(development.r2), "confirmation_r2": float(confirmation.r2),
        "global_generalization_established": False, "release_readiness": "research candidate; not a validated universal or stable single-checkpoint model",
        "primary_checkpoint_metrics": json.loads(primary.to_json(orient="records", double_precision=15))}
    (ROOT/"research_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False)+"\n")
    lines = ["# Unified DOC model: cross-river diagnosis and optimization", "",
        "Completed 12 October 2026. Transfer errors improve, but absolute skill and single-checkpoint stability remain insufficient for a universal release.", "",
        "## Outcome", "",
        f"Source-only held-region validation selected **{LABELS[selected]}** before either receiving case was scored. The environmental MLP, 12-month GRU and three upstream/downstream/global attention heads remain; six learned cross-message gate scalars are added (12,571 total parameters). The receiving river is evaluated alone and its DOC never fits parameters, normalization, calibration, epochs or recipe choice.", "",
        f"Against the matched newly trained reference, selected native/log MAE improve by {confirmation.mae_improvement_percent:.2f}%/{confirmation.log1p_mae_improvement_percent:.2f}% on independent Hackensack-Passaic. Mean R2 changes from -0.5669 to {confirmation.r2:.4f}; it is barely above zero. Delaware remains a development case with R2={development.r2:.4f}. This is an improvement, not resolution of universal generalization.", "",
        "## Why the previous transfer degraded", "",
        "Frozen-weight sensitivities isolate model reliance; they do not prove causal carbon mechanisms. Changing graph composition with the same receiving inputs and weights changes MAE substantially:", "",
        "| Inference intervention on already inspected Delaware | MAE | log-MAE | R2 |", "|---|---:|---:|---:|"]
    for case in ("reference", "source_context", "coords_neutral", "no_cross", "no_global", "no_directional", "hydro_missing", "eco_neutral"):
        row = diagnosis[(diagnosis.domain == "development_external") & (diagnosis.case == case)].iloc[0]
        lines.append(f"| {case} | {row.mae:.4f} | {row.log1p_mae:.4f} | {row.r2:.4f} |")
    lines += ["", "Three supported risks: (1) source validation previously included source exogenous-node context, while new receiving inference did not; restoring that context reduces MAE from 1.3505 to 0.9603, but this context-dependent mode is not deployed; (2) all receiving longitudes exceed source ranges, forest median changes from approximately 12% to 97%, and approximately 86% of receiving precipitation exceeds source ranges; (3) DOC-query flow availability falls from approximately 79% to 34%, and full-grid flow availability from 51% to 8%. Removing all cross-node information worsens the frozen model, and removing ecological context is particularly harmful. The jointly learned representation requires retraining; inference ablations are not a deployment optimization.", "",
        "Context, location and hydro availability explain model sensitivity and motivate these controlled training recipes; their interactions and underlying biogeochemical causes are not fully isolated. The controlled optimization reference and prior random-station release use different source cohorts/validation policies, so the historical MAE of 1.3505 versus the new 0.9426 cannot be attributed entirely to the gate.", "",
        "## Experiment design", "",
        "63 fresh fits: 7 recipes x 3 source HUC2 regional partitions x 3 seeds. Three rotated validation/test HUC2 pairs are 05/10, 10/07 and 07/05; the other four HUC2 units supply source training. These source units are regional groups within the source drainage, not three independent external rivers. All 357 source sites and observed labels remain in the corpus. No station-prefix-derived HUCs are used: mixed HUC8/HUC12 metadata and dropped leading zeros were correctly parsed from huc_cd. Role masks are disjoint at whole-unit level. Source validation/test inference uses the receiving unit alone for every recipe.", "",
        "Matched reference uses full source graph and cell-weighted loss. Isolated graphs change only source graph composition; balanced episodes add inverse-source-domain-label-count weights, conserving total epoch loss weight. Coordinate-neutral variants zero only normalized latitude/longitude. Sensor augmentation drops whole histories with probabilities 0.1 for temperature and 0.5 for discharge during training, correctly updating masks/ages/padding. Gated variants mix attended and self values with six learned sigmoid strengths, donor hydro availability and a 0.01 mean-gate penalty. Combined applies all changes. Every recipe uses the same core seeded tensors, widths, 12-month windows, source-only statistics, AdamW, shuffled month panels, maximum 60/minimum 15 epochs, patience 10 and one summed gradient update per panel.", "",
        "All 63 source selections and the source-validation-only choice were sealed, then all 126 complete label-free receiving fields were sealed, then independent confirmation DOC was opened. The new [Hackensack-Passaic HUC02030103 case](https://pubs.usgs.gov/of/1998/0282/report.pdf) has 100 stations, 90 station edges, 533 months and 3066 observed DOC queries. It overlaps neither source nor development stations, and source COMIDs are disjoint. All 100 stations pass source-compatible stream-order eligibility; no prediction-dependent filtering occurs. Initial basin-wide WQP temperature retrieval timed out; bounded station-specific requests completed the frozen cohort, with transport failures never treated as empty observations.", "",
        "## Results", "",
        "All values are arithmetic means of nine fit-level metrics; no prediction ensemble. All compared query coverages are 100%. Native MAE/RMSE use mg/L, log-MAE is log1p DOC. Bold is the best numerical value for the four error/skill metrics within each population, not a test-driven architecture choice.", ""]
    for population in ("source", "development", "confirmation"):
        part = summary[summary.population == population]
        lines += [f"### {population}", "", "| Recipe | MAE | log-MAE | RMSE | R2 | Bias |", "|---|---:|---:|---:|---:|---:|"]
        for recipe, label in LABELS.items():
            row = part[part.recipe == recipe].iloc[0]
            numbers = []
            for key in ("mae", "log1p_mae", "rmse", "r2"):
                best = part[key].max() if key == "r2" else part[key].min()
                value = f"{row[key]:.4f}"
                numbers.append(f"**{value}**" if row[key] == best else value)
            lines.append("| "+label+(" (SOURCE SELECTED)" if recipe == selected else "")+" | "+" | ".join(numbers)+f" | {row.bias:+.4f} |")
        lines.append("")
    lines += ["## Consistency, intervals and primary checkpoint", "",
        "| Population | Native wins/9 | Log wins/9 | Joint wins/9 | Positive R2/9 |", "|---|---:|---:|---:|---:|"]
    for row in wins[wins.recipe == selected].itertuples():
        lines.append(f"| {row.population} | {row.native_win} | {row.log_win} | {row.joint_win} | {row.positive_r2} |")
    lines += ["", "Delta means selected recipe error minus matched reference. Intervals use 3000 paired station-block draws within fixed cases; shared source fits, regional/hydrological correlation and multiple comparisons limit inference.", "",
        "| Population | Endpoint | Delta | Exploratory 95% interval |", "|---|---|---:|---|"]
    for interval in intervals:
        lo, hi = interval["ci95"]
        lines.append(f"| {interval['population']} | {interval['metric']} | {interval['delta']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ["", f"Primary checkpoint `{selection['primary_checkpoint_id']}` was fixed before test results. Its values are NOT the nine-fit averages:", "",
        "| Population | MAE | log-MAE | RMSE | R2 |", "|---|---:|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.population} | {row.mae:.4f} | {row.log1p_mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |")
    lines += ["", "The prechosen single checkpoint has negative R2 on both receiving rivers. A slightly positive average confirmation R2 does not make this checkpoint reliable or constitute a prediction ensemble. Retain the gated balanced-episode architecture as the current research candidate; do not advertise a frozen universal model. Development-native MAE is numerically lowest for combined, but this does not replace the source-selected recipe. Removing coordinates does not consistently improve independent confirmation, and adding every change does not beat the selected gated arm.", "",
        "## Verification and remaining gap", "",
        f"Independent audit replayed all 63 source/receiving checkpoints and 126 complete receiving grids, checked {audit['receipts_verified']} fit/field/test receipts and {audit['source_role_checks']} source label roles. Prediction replay error={audit['prediction_replay_max_abs']:.3g}; maximum independently recomputed metric difference={audit['metric_recompute_max_abs']:.3g}. Source normalization, disjoint roles, model capacities, data/mask/code hashes and blind selection/field gates were verified. Full regression suite: 1638 passed, 3 skipped for unavailable local artifacts; Ruff passed.", "",
        "Three hard source-unit test regions also have near-zero/negative mean R2, so this protocol reveals a broader out-of-region weakness. Two external cases do not establish global transfer. Sampling on observed DOC cells, where auxiliary hydro information is richer, does not establish accuracy on naturally unobserved cells; monthly station fields do not validate arbitrary continuous locations/times. The next scientific gap is environment-domain coverage (especially humid forested small catchments) and single-model stability, with a new external case needed for further confirmatory claims.", "",
        "Inspectable evidence: diagnosis/protocol.json and its prediction packages, diagnosis/environment_shift.csv, diagnosis/hydro_shift.csv, protocol.json, manifest.json, source_domains.csv, masks/, selected_model.json, all_selections.json, all_fields.json, analysis/summary.csv, analysis/by_fit.csv, analysis/paired_intervals.json, analysis/panel_consistency.csv, analysis/optimization_variability.csv, analysis/independent_audit.json and research_decision.json.", ""]
    REPORT.write_text("\n".join(lines))
    receipt = {"report_sha256": sha256_file(REPORT), "report_generator_sha256": sha256_file(__file__),
        "decision_sha256": sha256_file(ROOT/"research_decision.json"), "audit_sha256": sha256_file(ROOT/"analysis/independent_audit.json"),
        "files": {name: sha256_file(ROOT/"analysis"/name) for name in ("paired_intervals.json", "panel_consistency.csv", "optimization_variability.csv")}}
    (ROOT/"analysis/report_complete.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"report": str(REPORT), "selected_recipe": selected, "audit_passed": audit["passed"]}))


if __name__ == "__main__":
    main()
