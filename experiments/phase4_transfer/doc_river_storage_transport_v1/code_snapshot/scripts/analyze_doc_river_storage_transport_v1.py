"""Controlled DOC branch/storage responses on the 59 measured river corridors."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from river_graph.analysis.river_storage_transport import (
    controlled_responses,
    structural_contrasts,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_storage_transport_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1")
FOOTPRINT = PREVIOUS/"analysis/footprints.csv"
REPRESENTATIVES = PREVIOUS/"analysis/representatives.csv"
DTYPES = {c: str for c in ("pair_id", "source_a", "source_b", "target", "huc4")}
PULSE_SDS = (.075, .15, .30)
FRACTIONS = (0., .25, .5, 1.)
DT = .00125
METRICS = ("branch_peak_reduction_pct", "storage_peak_reduction_pct", "combined_peak_reduction_pct",
    "storage_duration_change_pct", "storage_sd_change_pct", "storage_share_of_added_variance",
    "branch_pulse_peak_difference", "storage_pulse_peak_difference", "interaction_pulse_peak_difference")
CODE = tuple(Path(p) for p in (
    "src/river_graph/analysis/river_storage_transport.py",
    "scripts/analyze_doc_river_storage_transport_v1.py",
    "scripts/plot_doc_river_storage_transport_v1.py",
    "scripts/verify_doc_river_storage_transport_v1.py",
    "tests/test_river_storage_transport.py"))


def simulate(footprints, examples, *, dt=DT, keep_curves=True):
    tables, curves = [], []
    selected = set(examples.pair_id)
    for row in footprints.itertuples():
        for sigma in PULSE_SDS:
            keep = keep_curves and sigma == .15 and row.pair_id in selected
            table, trace = controlled_responses(row.branch_a_km, row.branch_b_km, row.common_km,
                row.weight_a, sigma=sigma, fractions=FRACTIONS, dt=dt, keep_curves=keep)
            table["pair_id"] = row.pair_id
            tables.append(table)
            if keep:
                trace["pair_id"] = row.pair_id
                curves.append(trace)
    scenarios = pd.concat(tables, ignore_index=True).merge(footprints, on="pair_id", validate="many_to_one")
    contrasts = structural_contrasts(scenarios).merge(footprints, on="pair_id", validate="many_to_one")
    return scenarios, contrasts, pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def summaries(contrasts):
    keys = ["input_sd", "storage_fraction", "target", "huc4", "component", "cluster"]
    receivers = contrasts.groupby(keys, as_index=False).agg(
        **{m: (m, "mean") for m in METRICS}, n_connections=("pair_id", "nunique"))
    rows, classes = [], []
    for (sigma, fraction), f in receivers.groupby(["input_sd", "storage_fraction"]):
        pair = contrasts[contrasts.input_sd.eq(sigma) & contrasts.storage_fraction.eq(fraction)]
        for metric in METRICS:
            rows.append({"input_sd": sigma, "storage_fraction": fraction, "metric": metric,
                "receiver_equal_mean": f[metric].mean(), "receiver_median": f[metric].median(),
                "receiver_q25": f[metric].quantile(.25), "receiver_q75": f[metric].quantile(.75),
                "pair_equal_mean": pair[metric].mean(), "pair_median": pair[metric].median(),
                "n_receivers": f.target.nunique(), "n_connections": len(pair), "n_systems": f.component.nunique()})
        for cluster, c in f.groupby("cluster"):
            classes.append({"input_sd": sigma, "storage_fraction": fraction, "cluster": cluster,
                "n_receivers": c.target.nunique(), "n_connections": int(c.n_connections.sum()),
                "n_systems": c.component.nunique(), **{m: c[m].mean() for m in METRICS}})
    return receivers, pd.DataFrame(rows), pd.DataFrame(classes)


def numerical_comparison(coarse, fine):
    keys = ["pair_id", "input_sd", "branch_condition", "storage_fraction"]
    left, right = [f.set_index(keys).sort_index() for f in (coarse, fine)]
    assert left.index.equals(right.index)
    columns = ["pulse_peak", "peak_time", "pulse_sd", "pulse_centroid", "duration_80", "anomaly_area_fraction"]
    return pd.DataFrame({"metric": columns, "max_absolute_difference": [abs(left[c]-right[c]).max() for c in columns],
                         "coarse_dt": DT, "fine_dt": DT/2})


def main():
    a = ROOT/"analysis"
    a.mkdir(parents=True, exist_ok=True)
    footprints = pd.read_csv(FOOTPRINT, dtype=DTYPES)
    examples = pd.read_csv(REPRESENTATIVES, dtype=DTYPES)
    examples["selection"] = "existing geometry-median example"
    storage = footprints.sort_values(["common_storage_fraction", "pair_id"], ascending=[False, True]).iloc[0]
    extra = pd.DataFrame([{"example_group": 5, "pair_id": storage.pair_id, "n_connections": 1,
                           "selection": "maximum mapped common-trunk storage fraction"}])
    examples = pd.concat([examples, extra], ignore_index=True)
    scenarios, contrasts, curves = simulate(footprints, examples)
    fine, _, _ = simulate(footprints, examples, dt=DT/2, keep_curves=False)
    receivers, cohort, classes = summaries(contrasts)
    convergence = numerical_comparison(scenarios, fine)
    opportunity = footprints[footprints.common_storage_fraction.gt(0)].copy()
    dominance = footprints.copy()
    dominance["relative_branch_sd"] = dominance.total_path_cv
    dominance["storage_fraction_for_equal_variance"] = dominance.total_path_cv/dominance.common_fraction
    dominance["storage_can_dominate_within_budget"] = dominance.storage_fraction_for_equal_variance.le(1)
    tables = {"scenario_metrics": scenarios, "structural_contrasts": contrasts, "receiver_contrasts": receivers,
        "cohort_summary": cohort, "outline_context": classes, "representatives": examples,
        "numerical_convergence": convergence, "mapped_storage_opportunity": opportunity,
        "variance_dominance": dominance}
    for name, frame in tables.items():
        frame.to_csv(a/f"{name}.csv", index=False)
    curves.to_parquet(a/"representative_responses.parquet", index=False)
    config = {"input_pulse_sds": PULSE_SDS, "common_storage_fractions": FRACTIONS, "dt": DT,
        "mean_arrival": 1., "storage_kernel": "causal unit-gain exponential",
        "mixture_weights": "fixed drainage-area shares; constant positive flow",
        "storage_parameter_from_mapped_waterbody": False, "cohort_aggregation": "pairs within receiver, receivers equal",
        "chemical_loss": False, "time_unit": "relative scenario time, not days",
        "previous_results_seen": True, "new_model_training": False}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    summary = {"n_connections": len(footprints), "n_receivers": footprints.target.nunique(),
        "n_systems": footprints.component.nunique(), "n_scenarios": len(scenarios),
        "n_common_storage_connections": len(opportunity), "n_common_storage_receivers": opportunity.target.nunique(),
        "n_common_storage_systems": opportunity.component.nunique(),
        "max_centroid_error": float(abs(scenarios.pulse_centroid-1).max()),
        "max_area_error": float(abs(scenarios.anomaly_area_fraction-1).max()),
        "max_sd_error": float(abs(scenarios.pulse_sd-scenarios.analytic_sd).max()),
        "n_storage_can_dominate_within_budget": int(dominance.storage_can_dominate_within_budget.sum()),
        "median_fraction_for_equal_variance": float(dominance.storage_fraction_for_equal_variance.median()),
        "outline_cohort_counts": footprints.groupby("cluster").agg(connections=("pair_id", "size"),
            receivers=("target", "nunique"), systems=("component", "nunique")).reset_index().to_dict("records")}
    (a/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    for path in CODE:
        dest = ROOT/"code_snapshot"/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    sources = {"source_hashes": {str(p): sha256_file(p) for p in (FOOTPRINT, REPRESENTATIVES,
        ROOT/"study_plan.md", ROOT/"config.json", *CODE)},
        "product_hashes": {str(p): sha256_file(p) for p in sorted(a.iterdir()) if p.is_file()}}
    (ROOT/"analysis_sources.json").write_text(json.dumps(sources, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    print(cohort.query("input_sd == .15 and metric in ['branch_peak_reduction_pct', 'storage_peak_reduction_pct', 'storage_duration_change_pct']").to_string(index=False))


if __name__ == "__main__":
    main()
