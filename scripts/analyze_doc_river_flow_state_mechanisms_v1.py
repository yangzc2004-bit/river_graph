"""Explain observed confluence DOC fluctuation ratios without equating them to removal."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_flow_state_mechanisms import (
    no_processing_feasibility,
    shapley_mixing_change,
    state_contrast,
    state_summary,
)
from river_graph.analysis.river_joint_campaigns import (
    assign_flow_states,
    project_campaigns,
    resample_years,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_flow_state_mechanisms_v1")
PARENT = Path("experiments/phase4_transfer/doc_river_joint_campaigns_v1")
CODE = (Path("scripts/analyze_doc_river_flow_state_mechanisms_v1.py"),
        Path("src/river_graph/analysis/river_flow_state_mechanisms.py"),
        Path("src/river_graph/analysis/river_joint_campaigns.py"))
INPUTS = (PARENT/"analysis/adjusted_campaigns.parquet", PARENT/"analysis/flow_references.csv",
          PARENT/"analysis/configuration_comparison.csv", ROOT/"study_plan.md")
STATE_METRICS = ("sd_a", "sd_b", "rho", "w", "mixture_sd", "outlet_sd", "departure_sd",
    "outlet_mixture_sd_ratio", "mixing_potential_pct", "population_weight_mixing_potential_pct",
    "dynamic_fixed_mixture_sd_ratio", "normalized_departure_variance",
    "normalized_departure_covariance_term", "median_upstream_flow_share")
CONTRAST_METRICS = ("log_ratio_change", "log_outlet_sd_contribution", "log_mixture_sd_contribution",
    "outlet_sd_change_pct", "mixture_sd_change_pct", "branch_correlation_change",
    "mean_flow_fraction_change", "dynamic_fixed_ratio_log_change",
    "departure_covariance_term_change", "departure_variance_term_change",
    "correlation_pp", "amplitude_balance_pp", "mean_flow_share_pp", "mixing_potential_change_pp")


def point_tables(frame):
    states, contrasts, substitutions = [], [], []
    for (population, site), group in frame.groupby(["population", "receiver"], sort=False):
        metrics = {}
        for state, sub in group.groupby("flow_state", sort=False):
            if len(sub) < 5:
                continue
            metrics[state] = state_summary(sub)
            states.append({"population": population, "receiver": site,
                           "flow_state": state, **metrics[state]})
        if not {"low", "high"}.issubset(metrics):
            continue
        key = {"population": population, "receiver": site}
        contrasts.append(key | {"n_low": metrics["low"]["n_campaigns"],
            "n_high": metrics["high"]["n_campaigns"],
            "n_years": min(metrics["low"]["n_years"], metrics["high"]["n_years"]),
            **state_contrast(metrics["low"], metrics["high"])})
        _, rows = shapley_mixing_change(metrics["low"], metrics["high"])
        substitutions.extend(key | row for row in rows)
    return pd.DataFrame(states), pd.DataFrame(contrasts), pd.DataFrame(substitutions)


def bootstrap(frame, references, *, draws, seed, population):
    rng = np.random.default_rng(seed)
    years = np.sort(frame.date_local.dt.year.unique())
    references = references.set_index("receiver").to_dict("index")
    state_rows, contrast_rows, skipped = [], [], 0
    for draw in range(draws):
        sampled = resample_years(frame, rng.choice(years, len(years), replace=True))
        for site, sub in sampled.groupby("receiver", sort=False):
            try:
                projected = assign_flow_states(project_campaigns(sub), references[site])
            except ValueError:
                skipped += 1
                continue
            metrics = {}
            for state, group in projected.groupby("flow_state", sort=False):
                if len(group) < 5:
                    continue
                try:
                    metrics[state] = state_summary(group)
                except ValueError:
                    skipped += 1
                    continue
                state_rows.append({"draw": draw, "receiver": site, "flow_state": state,
                                   **{k: metrics[state][k] for k in STATE_METRICS}})
            if {"low", "high"}.issubset(metrics):
                contrast_rows.append({"draw": draw, "receiver": site,
                                      **state_contrast(metrics["low"], metrics["high"])})
        if (draw+1) % 1000 == 0:
            print(f"{population}: {draw+1}/{draws} whole-year draws", flush=True)
    return pd.DataFrame(state_rows), pd.DataFrame(contrast_rows), skipped


def intervals(points, samples, keys, metrics, requested):
    result = points.copy()
    for metric in metrics:
        result[metric+"_lo"] = np.nan
        result[metric+"_hi"] = np.nan
        result[metric+"_n_valid"] = 0
    for key, selected in samples.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        mask = np.ones(len(result), dtype=bool)
        for name, value in zip(keys, key, strict=True):
            mask &= result[name].eq(value).to_numpy()
        for metric in metrics:
            values = selected.loc[np.isfinite(selected[metric]), metric]
            result.loc[mask, metric+"_n_valid"] = len(values)
            if len(values) >= .9*requested and result.loc[mask, "n_years"].ge(3).all():
                result.loc[mask, [metric+"_lo", metric+"_hi"]] = np.quantile(values, [.025, .975])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap draws must be positive")
    frame = pd.read_parquet(INPUTS[0])
    references = pd.read_csv(INPUTS[1])
    points, contrasts, substitutions = point_tables(frame)
    state_outputs, contrast_outputs, diagnostics = [], [], []
    for i, (population, group) in enumerate(frame.groupby("population", sort=True)):
        selected_refs = references.loc[references.population.eq(population)]
        states, changes, skipped = bootstrap(group, selected_refs, draws=args.bootstrap_draws,
                                            seed=42+i, population=population)
        state_outputs.append(intervals(points.loc[points.population.eq(population)], states,
            ["receiver", "flow_state"], STATE_METRICS, args.bootstrap_draws))
        contrast_outputs.append(intervals(contrasts.loc[contrasts.population.eq(population)], changes,
            ["receiver"], CONTRAST_METRICS, args.bootstrap_draws))
        diagnostics.append({"population": population, "seed": 42+i,
            "n_calendar_dates": int(group.date_local.nunique()), "n_years": int(group.date_local.dt.year.nunique()),
            "bootstrap_draws": args.bootstrap_draws, "degenerate_estimates_skipped": skipped})
    ledger = no_processing_feasibility(frame)
    budget = ledger.groupby(["population", "receiver", "flow_state", "budget_exclusion"]).agg(
        n_campaigns=("receiver", "size"), n_nonnegative=("nonnegative_unmonitored_solution", "sum")).reset_index()
    outputs = ROOT/"analysis"
    outputs.mkdir(parents=True, exist_ok=True)
    tables = {"state_summary": pd.concat(state_outputs, ignore_index=True),
              "high_minus_low": pd.concat(contrast_outputs, ignore_index=True),
              "analytic_substitutions": substitutions, "water_budget_counts": budget,
              "bootstrap_diagnostics": pd.DataFrame(diagnostics)}
    for name, table in tables.items():
        table.to_csv(outputs/f"{name}.csv", index=False)
    ledger.to_parquet(outputs/"water_budget_ledger.parquet", index=False)
    summary = {"n_joint_dates": int(frame.loc[frame.population.eq("joint_calendar"), "date_local"].nunique()),
        "n_all_configuration_dates": int(frame.population.eq("all_calendar").sum()),
        "bootstrap_draws": args.bootstrap_draws,
        "joint_years": sorted(int(y) for y in frame.loc[frame.population.eq("joint_calendar"), "date_local"].dt.year.unique()),
        "scope": "Exploratory signal decomposition at four nested Krycklan confluences",
        "physical_channel_removal_identified": False, "new_prediction_training": False,
        "daily_flow_is_complete_instantaneous_load": False,
        "fraction_interpretation": "Algebraic no-net-processing feasibility; unmonitored DOC is not measured"}
    (outputs/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    metadata = {"input_hashes": {str(p): sha256_file(p) for p in INPUTS},
        "code_hashes": {str(p): sha256_file(p) for p in CODE},
        "output_hashes": {str(p): sha256_file(p) for p in sorted(outputs.iterdir())},
        "bootstrap_cluster": "complete calendar year, jointly across configurations; projection refit",
        "interval": "95% percentile; >=90% valid draws and >=3 original state years"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
