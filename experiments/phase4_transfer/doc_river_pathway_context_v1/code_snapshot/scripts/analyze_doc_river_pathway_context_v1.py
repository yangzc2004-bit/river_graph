"""Connect complete river form to monitored paths and receiving DOC variance."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_monitored_arrivals_v1 import (
    CELLS,
    DATASET,
    MAPPING,
    MASKS,
    NODES,
    PANEL,
    TYPES,
    VAA,
    load_context,
    routed_network,
)

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_monitored_arrivals import (
    association_table,
    availability_ok,
    form_contrasts,
)
from river_graph.analysis.river_pathway_context import (
    comparison_opportunities,
    corridor_geometry,
    variance_budget,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_pathway_context_v1")
PRIOR = Path("experiments/phase4_transfer/doc_river_monitored_arrivals_v1")
FOCALS = ("monitored_path_cv", "monitored_common_fraction", "monitored_storage_length_fraction",
          "entry_branch_covariance_fraction")
OUTCOMES = ("outlet_mix_log_sd_ratio", "unexplained_outlet_variance_fraction")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_pathway_context_v1.py",
    "scripts/plot_doc_river_pathway_context_v1.py", "scripts/verify_doc_river_pathway_context_v1.py",
    "src/river_graph/analysis/river_pathway_context.py", "tests/test_river_pathway_context.py",
    "scripts/analyze_doc_river_monitored_arrivals_v1.py",
    "src/river_graph/analysis/river_monitored_arrivals.py",
    "src/river_graph/analysis/river_junction_layout.py",
    "src/river_graph/analysis/river_whole_storage.py",
    "src/river_graph/analysis/river_form_process.py",
    "src/river_graph/analysis/river_mechanisms.py")))


def calculate(*, draws=5000, shortest=False):
    context = load_context()
    data, nodes, _, panel, lookup, visible, dates, vaa = context
    prior = PRIOR/"shortest_route_sensitivity" if shortest else PRIOR
    inv = pd.read_csv(prior/"analysis/network_inventory.csv", dtype=TYPES)
    gauges = pd.read_csv(prior/"analysis/candidate_gauges.csv", dtype=TYPES)
    old = pd.read_csv(prior/"analysis/receiver_signals.csv", dtype=TYPES)
    old = old[old.version.eq("full_monthly")].set_index("target")
    series = pd.read_parquet(prior/"analysis/monthly_series.parquet")
    rows, paths_long, budgets, flow_rows, traces, input_files = [], [], [], [], [], []
    hydro_index = list(data["feature_channels"]).index("discharge")
    x, xm = np.asarray(data["x"]), np.asarray(data["x_mask"])
    for target, group in series.groupby("target", sort=True):
        row = panel[panel.station.eq(target)].iloc[0]
        paths, layout, _, files = routed_network(row, vaa, shortest)
        input_files.extend(files)
        g = gauges[gauges.target.eq(target) & gauges.selected].sort_values("source_order").copy()
        geo, reaches = corridor_geometry(paths, g, int(row.comid), nodes.loc[target, "measure"],
            vaa.loc[paths.comids, "wbareatype"].fillna("").to_numpy(),
            vaa.loc[paths.comids, "wbareacomi"].fillna(0).to_numpy(np.int64), layout)
        paths_long.append(reaches.assign(target=target))
        group = group.sort_values("date")
        slots = group.month_index.to_numpy(int)
        ids = [lookup[s] for s in g.station]
        a, y = visible[ids][:, slots].T, visible[lookup[target], slots]
        w, time = g.area_weight.to_numpy(float), dates[slots]
        point, trace = variance_budget(a, y, w, time)
        np.testing.assert_allclose(trace.doc_receiver, group.doc_receiver, atol=1e-12)
        np.testing.assert_allclose(trace.doc_mixture, group.doc_mixture, atol=1e-12)
        np.testing.assert_allclose(point["outlet_mix_log_sd_ratio"], old.loc[target, "outlet_mix_log_sd_ratio"], atol=1e-10)
        meta = {"target": target, "cluster": int(row.cluster), "huc4": row.huc4,
            "component": int(old.loc[target, "component"])}
        selected_inv = inv[inv.station.eq(target)].iloc[0].drop(labels=["station"]).to_dict()
        names = nodes.loc[[*g.station, target], "station_nm"].astype(str)
        documented = target == "401733105392404"
        context_label = ("documented lake; NHD corridor tags also present" if documented and geo["n_monitored_corridor_waterbodies"]
                         else "documented lake; no corridor reach tag" if documented
                         else "mapped lake/reservoir corridor present" if geo["n_monitored_corridor_waterbodies"]
                         else "no mapped corridor tag; absence not established")
        rows.append({**selected_inv, **meta, **geo, **point,
            "station_name": names.loc[target], "documented_lake_context": documented,
            "storage_context": context_label})
        budgets.append({**meta, "subset": "full_monthly", "weighting": "fixed_area", **point})
        traces.append(trace.assign(**meta, subset="full_monthly", weighting="fixed_area"))
        q = x[[*ids, lookup[target]]][:, slots, hydro_index].T
        observed_q = xm[[*ids, lookup[target]]][:, slots, hydro_index].T
        positive = (observed_q & np.isfinite(q) & (q > 0)).all(axis=1)
        closure = np.full(len(time), np.nan)
        closure[positive] = q[positive, :-1].sum(axis=1)/q[positive, -1]
        for subset, keep in (("positive_common_flow", positive),
                             ("flow_closure_80_120", positive & (closure >= .8) & (closure <= 1.2))):
            usable = availability_ok(keep, time)
            flow_rows.append({**meta, "subset": subset, "n_common_months": int(keep.sum()),
                "eligible": usable, "median_source_receiver_flow_ratio": np.nanmedian(closure[keep]) if keep.any() else np.nan,
                "area_coverage": selected_inv["covered_area_fraction"],
                "flow_unit": "dataset monthly discharge, cfs; shares unit independent"})
            if not usable:
                continue
            for label, weight in (("fixed_area", w), ("monthly_flow", q[keep, :-1])):
                result, trace = variance_budget(a[keep], y[keep], weight, time[keep])
                budgets.append({**meta, "subset": subset, "weighting": label, **result})
                traces.append(trace.assign(**meta, subset=subset, weighting=label))
    receiver = pd.DataFrame(rows)
    budget = pd.DataFrame(budgets)
    coverage, summaries, associations, contrasts = [], [], [], []
    metrics = (*OUTCOMES, "monitored_common_fraction", "monitored_path_cv",
               "mixture_variance_share", "mismatch_variance_share", "covariance_variance_share")
    for cut in (0., .5, .7, .8, .9):
        f = receiver[receiver.covered_area_fraction.ge(cut)]
        for c in (1, 2, 3):
            a = f[f.cluster.eq(c)]
            coverage.append({"minimum_coverage": cut, "cluster": c, "n_receivers": len(a),
                "n_systems": a.component.nunique(), "n_huc4": a.huc4.nunique()})
        for group_name, g in (("all", f), *( (f"class_{c}", f[f.cluster.eq(c)]) for c in (1, 2, 3))):
            for metric in metrics:
                result = cluster_mean(g, metric, "component", draws=draws)
                if result is not None:
                    summaries.append({"minimum_coverage": cut, "group": group_name, **result})
        contrasts.append(form_contrasts(f, metrics, draws=draws).assign(minimum_coverage=cut))
        associations.append(association_table(f, FOCALS, OUTCOMES, draws=draws).assign(minimum_coverage=cut))
    flow_contrasts = []
    for subset, f in budget[budget.subset.ne("full_monthly")].groupby("subset"):
        a = f[f.weighting.eq("fixed_area")].set_index("target")
        b = f[f.weighting.eq("monthly_flow")].set_index("target")
        for metric in OUTCOMES + ("outlet_mix_correlation",):
            change = a.reset_index()[["target", "huc4", "component"]].copy()
            change[metric] = (b[metric]-a[metric]).reindex(change.target).to_numpy()
            result = cluster_mean(change, metric, "component", draws=draws)
            if result:
                flow_contrasts.append({"subset": subset, "contrast": "monthly_flow_minus_fixed_area", **result})
    opportunities = comparison_opportunities(inv)
    tables = {"receiver_pathway_panel": receiver, "source_corridor_reaches": pd.concat(paths_long, ignore_index=True),
        "receiving_variance_budgets": budget, "variance_series": pd.concat(traces, ignore_index=True),
        "coverage_opportunities": pd.DataFrame(coverage), "signal_summary": pd.DataFrame(summaries),
        "form_contrasts": pd.concat(contrasts, ignore_index=True),
        "geometry_associations": pd.concat(associations, ignore_index=True),
        "flow_availability": pd.DataFrame(flow_rows), "flow_paired_contrasts": pd.DataFrame(flow_contrasts),
        "same_region_form_pairs": opportunities}
    summary = {"n_receivers": len(receiver), "n_systems": receiver.component.nunique(),
        "n_high_coverage_receivers": int(receiver.covered_area_fraction.ge(.8).sum()),
        "n_same_huc4_shape_pairs": len(opportunities),
        "n_area_comparable_shape_pairs": int(opportunities.area_comparable.sum()),
        "n_both_observed_area_comparable_pairs": int((opportunities.area_comparable & opportunities.both_observed).sum()),
        "n_usable_high_coverage_form_pairs": int(opportunities.usable_form_pair.sum()),
        "n_mapped_corridor_storage_receivers": int(receiver.n_monitored_corridor_waterbodies.gt(0).sum()),
        "n_documented_lake_without_corridor_tags": int((receiver.documented_lake_context & receiver.n_monitored_corridor_waterbodies.eq(0)).sum()),
        "n_flow_eligible_receivers": int(tables["flow_availability"].query("subset == 'positive_common_flow' and eligible").shape[0]),
        "n_flow_closure_eligible_receivers": int(tables["flow_availability"].query("subset == 'flow_closure_80_120' and eligible").shape[0]),
        "bootstrap_draws": draws, "new_training": False}
    files = [prior/"analysis/network_inventory.csv", prior/"analysis/candidate_gauges.csv",
        prior/"analysis/receiver_signals.csv", prior/"analysis/monthly_series.parquet", DATASET, NODES, MAPPING, VAA,
        CELLS, PANEL, *(MASKS/f"split{seed}.npz" for seed in (142, 143, 144)),
        *input_files]
    return tables, summary, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables, summary, sources = calculate(draws=args.bootstrap_draws, shortest=args.shortest_routes)
    outputs = []
    for name, f in tables.items():
        path = out/f"{name}.{'parquet' if name == 'variance_series' else 'csv'}"
        f.to_parquet(path, index=False) if name == "variance_series" else f.to_csv(path, index=False)
        outputs.append(path)
    (out/"summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item())+"\n")
    outputs.append(out/"summary.json")
    config = {"previous_results_seen": True, "minimum_coverage": .8,
        "coverage_sensitivity": [0., .5, .7, .8, .9], "same_huc4_maximum_area_ratio": 2.,
        "flow_closure_range": [.8, 1.2], "minimum_months": 24, "minimum_years": 3,
        "minimum_calendar_months": 6, "bootstrap_draws": args.bootstrap_draws,
        "routing": "saved shortest" if args.shortest_routes else "original geometric mainstem",
        "interpretation": "exploratory observed concentration variance; not DOC mass balance or event transit"}
    (root/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    for path in CODE:
        target = root/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    sources.extend([ROOT/"study_plan.md", *CODE,
        Path("scripts/analyze_doc_river_monitored_arrivals_v1.py"),
        Path("src/river_graph/analysis/river_monitored_arrivals.py"),
        Path("src/river_graph/analysis/river_junction_layout.py"),
        Path("src/river_graph/analysis/river_form_process.py")])
    (root/"analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(path): sha256_file(path) for path in sources},
        "output_hashes": {str(path): sha256_file(path) for path in outputs},
        "new_model_fit": False}, indent=2)+"\n")
    print(json.dumps(summary, indent=2, default=lambda x: x.item()), flush=True)


if __name__ == "__main__":
    main()
