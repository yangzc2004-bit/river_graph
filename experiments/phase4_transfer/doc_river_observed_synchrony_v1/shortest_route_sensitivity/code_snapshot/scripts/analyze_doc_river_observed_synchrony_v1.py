"""Observed source timing, independent receiving response and whole river form."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_monitored_arrivals import (
    association_table,
    availability_ok,
    form_contrasts,
)
from river_graph.analysis.river_observed_synchrony import (
    coordination_statistics,
    excursion_block_interval,
    within_receiver_association,
)
from river_graph.experiments.provenance import sha256_file

BASE = Path("experiments/phase4_transfer")
ROOT = BASE/"doc_river_observed_synchrony_v1"
ARRIVALS = BASE/"doc_river_monitored_arrivals_v1"
PATHWAYS = BASE/"doc_river_pathway_context_v1"
TYPES = {"station": str, "target": str, "huc4": str, "source_station": str}
METRICS = ("source_coherence", "outlet_sync_log_sd_ratio", "outlet_mix_log_sd_ratio",
           "peak_risk_difference", "coincident_date_fraction", "outlet_mix_correlation")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_observed_synchrony_v1.py",
    "scripts/plot_doc_river_observed_synchrony_v1.py", "scripts/verify_doc_river_observed_synchrony_v1.py",
    "src/river_graph/analysis/river_observed_synchrony.py", "tests/test_river_observed_synchrony.py",
    "src/river_graph/analysis/river_monitored_arrivals.py", "src/river_graph/analysis/river_form_process.py")))


def _matrix(group, value, *, date="month", station="station"):
    if group.duplicated([date, "source_order"]).any():
        raise ValueError("one source/receiver row per common date required")
    orders = sorted(group.source_order.unique())
    if orders[0] != -1 or orders[1:] != list(range(len(orders)-1)):
        raise ValueError("consecutive source columns and one receiver required")
    pivot = group.pivot(index=date, columns="source_order", values=value).sort_index()
    if not np.isfinite(pivot.to_numpy(float)).all():
        raise ValueError("complete common-date panel required")
    identities = group.groupby("source_order")[station].nunique()
    if identities.ne(1).any():
        raise ValueError("source identities must stay fixed")
    sources = group[group.source_order.ge(0)].groupby("source_order").area_weight.first().sort_index()
    if group[group.source_order.ge(0)].groupby("source_order").area_weight.nunique().ne(1).any():
        raise ValueError("fixed area shares required")
    return pivot[orders[1:]].to_numpy(float), pivot[-1].to_numpy(float), sources.to_numpy(float), pivot.index


def _check_frozen_product(path, ledger):
    record = json.loads(ledger.read_text())
    if record["output_hashes"].get(str(path)) != sha256_file(path):
        raise ValueError(f"frozen source product differs: {path}")


def calculate(*, draws=5000, shortest=False):
    arrivals = ARRIVALS/"shortest_route_sensitivity" if shortest else ARRIVALS
    pathways = PATHWAYS/"shortest_route_sensitivity" if shortest else PATHWAYS
    files = [arrivals/"analysis/selected_activities.parquet", arrivals/"analysis/monthly_series.parquet",
        arrivals/"analysis/weekly_case_selected_activities.parquet", arrivals/"analysis/weekly_case_stations.csv",
        pathways/"analysis/receiver_pathway_panel.csv"]
    for path in files:
        _check_frozen_product(path, (pathways if path.parent.parent == pathways else arrivals)/"analysis_sources.json")
    selected = pd.read_parquet(files[0])
    selected["month"] = pd.to_datetime(selected.month)
    reference = pd.read_parquet(files[1])
    panel = pd.read_csv(files[4], dtype=TYPES).set_index("target", drop=False)
    # Shortest routes have their independently retained sampling population.
    if not shortest and (len(panel) != 32 or selected.target.nunique() != 32 or len(reference) != 1333):
        raise ValueError("retain the fixed 32-network, 1333-month primary population")
    points, traces, availability, periods, period_inventory = [], [], [], [], []
    for target, group in selected.groupby("target", sort=True):
        row = panel.loc[target]
        meta = {k: row[k] for k in ("target", "huc4", "cluster", "component", "covered_area_fraction",
            "monitored_path_cv", "monitored_common_fraction", "monitored_storage_length_fraction")}
        a, y, w, time = _matrix(group, "doc_monthly")
        old = reference[reference.target.eq(target)].sort_values("date")
        np.testing.assert_array_equal(pd.DatetimeIndex(old.date), time)
        np.testing.assert_allclose(y, old.doc_receiver, atol=1e-10)
        np.testing.assert_allclose(a@(w/w.sum()), old.doc_mixture, atol=1e-10)
        slots = [(-1, "monthly", group)]
        for cut in (0, 1, 3, 7):
            restricted = group[group.sample_span_days.le(cut)]
            dates = pd.DatetimeIndex(restricted.month.unique()).sort_values()
            eligible = availability_ok(np.ones(len(dates), bool), dates)
            availability.append({**meta, "maximum_span_days": cut, "n_dates": len(dates),
                "n_years": dates.year.nunique(), "n_calendar_months": dates.month.nunique(), "eligible": eligible})
            if eligible:
                slots.extend([(cut, "activity", restricted), (cut, "monthly_same_dates", restricted)])
        full_trace = None
        for cut, version, g in slots:
            a, y, w, dates = _matrix(g, "doc_activity" if version == "activity" else "doc_monthly")
            for adjustment in ("calendar_year", "raw"):
                for quantile in (.75, .9):
                    result, trace = coordination_statistics(a, y, w, dates, adjustment=adjustment, quantile=quantile)
                    labels = {**meta, "sample_cut_days": cut, "version": version, "adjustment": adjustment}
                    points.append({**labels, **result})
                    traces.append(trace.assign(**labels, excursion_quantile=quantile))
                    if version == "monthly" and adjustment == "calendar_year" and quantile == .75:
                        full_trace = trace
        assert full_trace is not None
        block = (full_trace.date.dt.year//5)*5
        for begin, t in full_trace.groupby(block):
            ok = availability_ok(np.ones(len(t), bool), t.date)
            period_inventory.append({**meta, "period_start_year": int(begin), "n_dates": len(t),
                "n_years": t.date.dt.year.nunique(), "n_calendar_months": t.date.dt.month.nunique(), "eligible": ok})
            if ok:
                source_columns = [f"source_{j}_anomaly" for j in range(len(w))]
                result, _ = coordination_statistics(t[source_columns], t.receiver_anomaly, w, t.date, projected=True)
                periods.append({**meta, "period_start_year": int(begin), **result})
    network = pd.DataFrame(points)
    series = pd.concat(traces, ignore_index=True)
    summaries, contrasts, associations, paired = [], [], [], []
    for (cut, version, adjustment, quantile), f in network.groupby(
            ["sample_cut_days", "version", "adjustment", "excursion_quantile"]):
        labels = {"sample_cut_days": cut, "version": version, "adjustment": adjustment,
            "excursion_quantile": quantile}
        for coverage in (0., .8):
            eligible = f[f.covered_area_fraction.ge(coverage)]
            for group_name, g in (("all", eligible), *((f"class_{c}", eligible[eligible.cluster.eq(c)]) for c in (1, 2, 3))):
                for metric in METRICS:
                    result = cluster_mean(g, metric, "component", draws=draws)
                    if result:
                        summaries.append({**labels, "minimum_coverage": coverage, "group": group_name, **result})
            contrasts.append(form_contrasts(eligible, METRICS, draws=draws).assign(**labels, minimum_coverage=coverage))
            associations.append(association_table(eligible,
                ("source_coherence", "monitored_path_cv", "monitored_common_fraction"),
                ("outlet_sync_log_sd_ratio", "peak_risk_difference"), draws=draws).assign(**labels, minimum_coverage=coverage))
    for (cut, adjustment, quantile), f in network[network.version.ne("monthly")].groupby(
            ["sample_cut_days", "adjustment", "excursion_quantile"]):
        a = f[f.version.eq("activity")].set_index("target")
        b = f[f.version.eq("monthly_same_dates")].set_index("target").reindex(a.index)
        for metric in METRICS:
            delta = a.reset_index()[["target", "huc4", "cluster", "component"]].copy()
            delta[metric] = (a[metric]-b[metric]).to_numpy()
            result = cluster_mean(delta, metric, "component", draws=draws)
            if result:
                paired.append({"sample_cut_days": cut, "adjustment": adjustment,
                    "excursion_quantile": quantile, "contrast": "activity_minus_monthly_on_same_dates", **result})
    period_frame = pd.DataFrame(periods)
    within, centered = within_receiver_association(period_frame, draws=draws)
    case, case_years, case_series, case_associations = dense_case(files[2], files[3], panel, draws=draws)
    tables = {"network_metrics": network, "observed_series": series,
        "sampling_availability": pd.DataFrame(availability), "signal_summary": pd.DataFrame(summaries),
        "form_contrasts": pd.concat(contrasts, ignore_index=True),
        "geometry_synchrony_associations": pd.concat(associations, ignore_index=True),
        "same_date_averaging_contrasts": pd.DataFrame(paired), "period_inventory": pd.DataFrame(period_inventory),
        "network_periods": period_frame, "within_network_periods": centered,
        "within_network_association": pd.DataFrame([within]), "dense_case_metrics": case,
        "dense_case_years": case_years, "dense_case_series": case_series,
        "dense_case_annual_associations": case_associations}
    primary = network.query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75")
    strict = network.query("version == 'activity' and sample_cut_days == 0 and adjustment == 'calendar_year' and excursion_quantile == .75")
    summary = {"n_receivers": len(primary), "n_systems": primary.component.nunique(),
        "n_monthly_dates": int(primary.n_dates.sum()), "n_strict_same_day_receivers": len(strict),
        "n_strict_same_day_dates": int(strict.n_dates.sum()),
        "n_high_coverage_receivers": int(primary.covered_area_fraction.ge(.8).sum()),
        "n_monthly_peak_comparisons": int(primary.peak_comparison_eligible.sum()),
        "n_eligible_five_year_blocks": len(period_frame),
        "n_receivers_with_repeated_blocks": within["n_receivers"],
        "dense_case_target": str(case.target.iloc[0]),
        "bootstrap_draws": draws, "new_training": False, "new_measurements": False}
    return tables, summary, files+[arrivals/"analysis_sources.json", pathways/"analysis_sources.json"]


def dense_case(events_path, stations_path, panel, *, draws):
    events = pd.read_parquet(events_path)
    events["date"] = pd.to_datetime(events.date)
    stations = pd.read_csv(stations_path, dtype=TYPES)
    target = str(stations.target.iloc[0])
    weights = stations[stations.source_order.ge(0)].sort_values("source_order").area_weight.to_numpy(float)
    events = events.merge(stations[["station", "area_weight"]], left_on="site_no", right_on="station", validate="many_to_one")
    meta = {k: panel.loc[target, k] for k in ("target", "huc4", "cluster", "component")}
    rows, traces, years, associations = [], [], [], []
    for version, value in (("selected_activity", "doc"), ("daily_mean", "doc_daily_mean")):
        a, y, w, time = _matrix(events, value, date="date", station="site_no")
        np.testing.assert_allclose(w, weights)
        dense = pd.Series(time).groupby(time.to_period("M")).transform("size").to_numpy() >= 3
        for adjustment, keep in (("calendar_year", np.ones(len(time), bool)), ("within_month", dense)):
            for quantile in (.75, .9):
                result, trace = coordination_statistics(a[keep], y[keep], weights, time[keep], adjustment=adjustment, quantile=quantile)
                labels = {**meta, "version": version, "adjustment": adjustment, "excursion_quantile": quantile}
                rows.append({**labels, **result, **excursion_block_interval(trace, pd.DatetimeIndex(trace.date).year, draws=draws)})
                traces.append(trace.assign(**labels))
                if quantile != .75:
                    continue
                annual = []
                for year, t in trace.groupby(trace.date.dt.year):
                    eligible = len(t) >= 12 and t.date.dt.month.nunique() >= 3
                    row = {**labels, "year": int(year), "n_dates": len(t), "n_months": t.date.dt.month.nunique(), "eligible": eligible}
                    if eligible:
                        columns = [f"source_{j}_anomaly" for j in range(len(weights))]
                        point, _ = coordination_statistics(t[columns], t.receiver_anomaly, weights, t.date, projected=True)
                        row.update(point)
                        annual.append(row)
                    years.append(row)
                f = pd.DataFrame(annual)
                if len(f) >= 3:
                    result = association_table(f.assign(component=f.year), ("source_coherence",),
                        ("outlet_sync_log_sd_ratio",), draws=draws)
                    result = result.rename(columns={"n_receivers": "n_years", "n_components": "n_year_blocks"})
                    associations.append(result.assign(**labels, interval_unit="calendar_year_in_one_fixed_network"))
    return (pd.DataFrame(rows), pd.DataFrame(years), pd.concat(traces, ignore_index=True),
            pd.concat(associations, ignore_index=True) if associations else pd.DataFrame())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables, summary, files = calculate(draws=args.bootstrap_draws, shortest=args.shortest_routes)
    outputs = []
    for name, frame in tables.items():
        path = out/f"{name}.{'parquet' if name.endswith('series') else 'csv'}"
        frame.to_parquet(path, index=False) if path.suffix == ".parquet" else frame.to_csv(path, index=False)
        outputs.append(path)
    for name, value in (("analysis/summary", summary), ("config", {
            "previous_results_seen": True, "bootstrap_draws": args.bootstrap_draws,
            "routing": "saved shortest" if args.shortest_routes else "original geometric mainstem",
            "peak_quantiles": [.75, .9], "minimum_peak_group": 5,
            "period_years": 5, "minimum_period_dates_years_months": [24, 3, 6],
            "dense_case_minimum_dates_months": [12, 3], "source_shares": "fixed non-overlapping catchment areas",
            "interpretation": "observed coordination, not physical transit-time fitting or causal form effects"})):
        path = root/f"{name}.json"
        path.write_text(json.dumps(value, indent=2, default=lambda x: x.item())+"\n")
        outputs.append(path)
    for path in CODE:
        copy = root/"code_snapshot"/path
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
    (root/"analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(path): sha256_file(path) for path in [ROOT/"study_plan.md", *files, *CODE]},
        "output_hashes": {str(path): sha256_file(path) for path in outputs},
        "new_model_fit": False}, indent=2)+"\n")
    print(json.dumps(summary, indent=2, default=lambda x: x.item()), flush=True)
    print(tables["signal_summary"].query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75 and group == 'all'").to_string(index=False))
    print(tables["dense_case_metrics"].query("excursion_quantile == .75").to_string(index=False))


if __name__ == "__main__":
    main()
