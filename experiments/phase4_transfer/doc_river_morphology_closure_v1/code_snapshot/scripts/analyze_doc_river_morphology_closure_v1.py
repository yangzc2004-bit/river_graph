"""Complete the river-form evidence chain without fitting transport to DOC."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_monitored_arrivals_v1 import load_context, routed_network

from river_graph.analysis.river_cosampling_geometry import within_month_dates
from river_graph.analysis.river_cosampling_pairs import disjoint_pair_inventory
from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_monitored_arrivals import monitored_frontier
from river_graph.analysis.river_morphology_closure import (
    junction_translation,
    pulse_response,
    source_dates,
)
from river_graph.analysis.river_observed_synchrony import coordination_statistics
from river_graph.analysis.river_sampling_resolution import doc_activities
from river_graph.analysis.river_signal_timescale import source_geometry
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file

BASE = Path("experiments/phase4_transfer")
ROOT = BASE / "doc_river_morphology_closure_v1"
PRIOR = BASE / "doc_river_cosampling_geometry_v1"
PATHS = BASE / "doc_river_pathway_context_v1"
OBSERVED = BASE / "doc_river_observed_synchrony_v1"
TYPES = {"station": str, "target": str, "source_station": str, "site_no": str,
         "elongated_station": str, "broad_station": str, "huc4": str}
SIGMAS = (.025, .1, .5, 2.)
METRICS = ("pulse_peak", "pulse_sd", "pulse_central80", "pulse_peak_time")
CODE = tuple(map(Path, ("src/river_graph/analysis/river_morphology_closure.py",
    "scripts/recover_doc_river_morphology_closure_v1.py",
    "scripts/discover_doc_river_morphology_closure_v1.py",
    "scripts/extend_doc_river_morphology_closure_v1.py",
    "scripts/analyze_doc_river_morphology_closure_v1.py",
    "scripts/plot_doc_river_morphology_closure_v1.py",
    "scripts/verify_doc_river_morphology_closure_v1.py",
    "tests/test_river_morphology_closure.py")))


def recover_analysis():
    manifest_path = ROOT / "recovery/retrieval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    accepted, inventory, raw_files = [], [], []
    for entry in manifest["records"]:
        station = entry["station"]
        name = entry.get("path", entry.get("fallback_path"))
        if name is None:
            inventory.append({"station": station, "archive_status": "unavailable", "n_activities": 0})
            continue
        path = Path(name)
        expected = entry.get("sha256", entry.get("fallback_sha256"))
        if sha256_file(path) != expected:
            raise ValueError(f"recorded raw archive changed: {path}")
        raw_files.append(path)
        raw = load_station_results(path)
        f = extract_doc_obs(raw)
        f = f[f.site_no.eq(station) & f.date.notna() & f.doc.ge(0) & np.isfinite(f.doc)].copy()
        meta = raw.loc[f.index, ["Activity_ActivityIdentifier", "Activity_StartTime", "Activity_StartTimeZone"]].rename(
            columns={"Activity_ActivityIdentifier": "event_id", "Activity_StartTime": "start_time", "Activity_StartTimeZone": "time_zone"})
        events = doc_activities(f.join(meta))
        accepted.append(events)
        inventory.append({"station": station, "archive_status": entry["status"],
            "uses_old_cache": "path" not in entry, "n_activities": len(events),
            "n_dates": events.date.nunique(), "first_date": events.date.min(), "last_date": events.date.max(),
            "known_clocks": int(events.timestamp_known.sum())})
    events = pd.concat(accepted, ignore_index=True)
    dates = source_dates(events)
    _, nodes, mapping, panel, _, _, _, vaa = load_context()
    requested = pd.read_csv(ROOT / "recovery/receivers.csv", dtype=TYPES)
    all_pairs, receiver_rows, selected_rows, point_rows, traces = [], [], [], [], []
    for row in panel[panel.station.isin(requested.station)].itertuples():
        paths, _, hydro, _ = routed_network(row, vaa)
        inside = nodes[nodes.comid.isin(paths.comids) & nodes.comid.ne(row.comid)]
        rows = []
        receiving = dates.get(row.station, pd.DatetimeIndex([]))
        for station, node in inside.iterrows():
            if station in dates and mapping.loc[station, "mapping_status"] != "gross_area_mismatch":
                rows.append({"station": station, "comid": int(node.comid), "measure": float(node.measure),
                    "n_common": len(receiving.intersection(dates[station])),
                    "reported_area_km2": mapping.loc[station, "drainage_area_km2"]})
        gauges = pd.DataFrame(rows, columns=["station", "comid", "measure", "n_common", "reported_area_km2"])
        gauges = gauges.sort_values(["n_common", "station"], ascending=[False, True]).drop_duplicates("comid")
        gauges = monitored_frontier(paths, hydro, gauges)
        pairs = disjoint_pair_inventory(gauges, dates, receiving, receiver_area=mapping.loc[row.station, "drainage_area_km2"])
        valid = pairs[pairs.official_area_consistent] if len(pairs) else pairs
        all_pairs.append(pairs.assign(target=row.station, huc4=row.huc4, cluster=int(row.cluster)))
        receiver_rows.append({"target": row.station, "cluster": int(row.cluster), "huc4": row.huc4,
            "n_pairs": len(pairs), "n_common_days_max": int(valid.n_common_days.max()) if len(valid) else 0,
            "n_dense_days_max": int(valid.n_within_month_days.max()) if len(valid) else 0,
            "same_day_eligible": bool(valid.same_day_eligible.any()) if len(valid) else False,
            "within_month_eligible": bool(valid.within_month_eligible.any()) if len(valid) else False})
        for mode, column in (("same_day", "same_day_eligible"), ("within_month", "within_month_eligible")):
            if valid.empty or not valid[column].any():
                continue
            pick = valid[valid[column]].sort_values(["represented_area_km2", "n_common_days", "source_a", "source_b"],
                ascending=[False, False, True, True]).iloc[0]
            sites = [pick.source_a, pick.source_b, row.station]
            common = receiving.intersection(dates[sites[0]]).intersection(dates[sites[1]])
            if mode == "within_month":
                common = within_month_dates(common)
            chosen = events[events.site_no.isin(sites) & events.date.isin(common)].sort_values(
                ["site_no", "date", "timestamp_utc", "event_id"], na_position="last").drop_duplicates(["site_no", "date"])
            pivot = chosen.pivot(index="date", columns="site_no", values="doc").sort_index()
            area = gauges.set_index("station").loc[sites[:2], "routed_area_km2"].to_numpy(float)
            stat, trace = coordination_statistics(pivot[sites[:2]], pivot[sites[2]], area, pivot.index,
                adjustment="within_month" if mode == "within_month" else "calendar_year")
            labels = {"target": row.station, "cluster": int(row.cluster), "huc4": row.huc4,
                "mode": mode, "source_a": sites[0], "source_b": sites[1]}
            point_rows.append({**labels, **stat})
            traces.append(trace.assign(**labels))
            selected_rows.append(chosen.assign(**labels))
    receiver = pd.DataFrame(receiver_rows)
    selected_pairs = pd.read_csv(ROOT / "recovery/selected_form_pairs.csv", dtype=TYPES)
    lookup = receiver.set_index("target")
    for form in ("elongated", "broad"):
        for column in ("n_common_days_max", "n_dense_days_max", "same_day_eligible", "within_month_eligible"):
            selected_pairs[form+"_recovered_"+column] = selected_pairs[form+"_station"].map(lookup[column])
    selected_pairs["both_recovered_same_day"] = selected_pairs.elongated_recovered_same_day_eligible & selected_pairs.broad_recovered_same_day_eligible
    selected_pairs["both_recovered_within_month"] = selected_pairs.elongated_recovered_within_month_eligible & selected_pairs.broad_recovered_within_month_eligible
    tables = {"recovered_activity_inventory": pd.DataFrame(inventory), "recovered_activities": events,
        "recovered_source_pairs": pd.concat(all_pairs, ignore_index=True), "recovered_receivers": receiver,
        "recovered_form_pairs": selected_pairs, "recovered_signal_metrics": pd.DataFrame(point_rows),
        "recovered_selected_activities": pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame(),
        "recovered_signal_series": pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()}
    summary = {"targeted_form_pairs": len(selected_pairs), "targeted_receivers": len(receiver),
        "requested_stations": len(manifest["records"]), "accepted_activities": len(events),
        "downloaded_stations": sum(r["status"] in ("downloaded", "saved_download") for r in manifest["records"]),
        "old_cache_fallback_stations": sum("path" not in r and "fallback_path" in r for r in manifest["records"]),
        "complete_form_pairs_same_day": int(selected_pairs.both_recovered_same_day.sum()),
        "complete_form_pairs_within_month": int(selected_pairs.both_recovered_within_month.sum()),
        "same_day_receivers_by_class": receiver.groupby("cluster").same_day_eligible.sum().to_dict(),
        "within_month_receivers_by_class": receiver.groupby("cluster").within_month_eligible.sum().to_dict(),
        "continuous_catalog_status": manifest["continuous_doc_catalog"]["status"]}
    return tables, summary, raw_files+[manifest_path]


def mechanism_analysis(*, shortest=False, draws=5000):
    root = PATHS / "shortest_route_sensitivity" if shortest else PATHS
    reaches_path = root / "analysis/source_corridor_reaches.csv"
    context_path = root / "analysis/receiver_pathway_panel.csv"
    reaches = pd.read_csv(reaches_path, dtype=TYPES)
    panel = pd.read_csv(context_path, dtype=TYPES)
    rows, waves, identities = [], [], []
    for r in panel.itertuples():
        sources, geometry = source_geometry(reaches[reaches.target.eq(r.target)])
        d, w = sources.normalized_delay.to_numpy(), sources.area_weight.to_numpy()
        mean, common = float(w@d), geometry["common_fraction"]
        meta = {"target": r.target, "huc4": r.huc4, "cluster": int(r.cluster), "component": int(r.component),
            "covered_area_fraction": r.covered_area_fraction, "path_cv": geometry["path_cv"], "common_fraction": common}
        for sigma in SIGMAS:
            scenarios = [("actual_paths", d, 0., np.zeros(len(d))),
                ("aligned_arrivals", np.full(len(d), mean), 0., np.zeros(len(d))),
                ("common_translation", d+1, 0., np.zeros(len(d))),
                ("compensated_source_clock", d, 0., mean-d)]
            scenarios.extend((f"actual_shared_{a:g}", d, a*common, np.zeros(len(d))) for a in (.25, .5))
            for fraction in (.2, .8):
                moved_common = fraction*d.min()
                moved = junction_translation(d, moved_common)
                scenarios.append((f"junction_translation_{fraction:g}", moved, 0., np.zeros(len(d))))
                scenarios.append((f"junction_spreading_{fraction:g}", moved, .5*moved_common, np.zeros(len(d))))
            for name, delay, memory, offset in scenarios:
                result, pdf, limits = pulse_response(delay, w, sigma=sigma, memory=memory, offsets=offset)
                rows.append({**meta, "sigma": sigma, "scenario": name, **result})
                if r.target == "401733105392404" and sigma == .1 and name in (
                        "actual_paths", "aligned_arrivals", "actual_shared_0.5", "compensated_source_clock"):
                    time = np.linspace(*limits, 2000)
                    waves.append(pd.DataFrame({"target": r.target, "sigma": sigma, "scenario": name,
                        "time": time, "response": pdf(time)}))
            # Check the null through its real complete-path decomposition,
            # rather than assigning identical output metrics by construction.
            identities.append({**meta, "sigma": sigma,
                "partition_error": float(np.max(np.abs(junction_translation(d, .2*d.min())-junction_translation(d, .8*d.min()))))})
    scenarios = pd.DataFrame(rows)
    keys = ["target", "sigma"]
    comparisons = {"path_difference": ("actual_paths", "aligned_arrivals"),
        "shared_spreading": ("actual_shared_0.5", "actual_paths"),
        "source_alignment": ("compensated_source_clock", "actual_paths"),
        "junction_only": ("junction_translation_0.8", "junction_translation_0.2"),
        "junction_with_spreading": ("junction_spreading_0.8", "junction_spreading_0.2"),
        "common_translation": ("common_translation", "actual_paths")}
    contrasts = []
    for comparison, (candidate, reference) in comparisons.items():
        a = scenarios[scenarios.scenario.eq(candidate)].set_index(keys)
        b = scenarios[scenarios.scenario.eq(reference)].set_index(keys).reindex(a.index)
        f = a.reset_index()[[*keys, "huc4", "cluster", "component", "covered_area_fraction"]]
        for metric in METRICS:
            f[metric+"_delta"] = (a[metric]-b[metric]).to_numpy()
            f[metric+"_relative"] = (a[metric]/b[metric]-1).to_numpy()
        contrasts.append(f.assign(comparison=comparison))
    contrast = pd.concat(contrasts, ignore_index=True)
    summary_rows, form_rows = [], []
    for cut in (0., .8):
        for (sigma, comparison), f in contrast[contrast.covered_area_fraction.ge(cut)].groupby(["sigma", "comparison"]):
            for metric in [m+suffix for m in METRICS for suffix in ("_delta", "_relative")]:
                result = cluster_mean(f, metric, "component", draws=draws)
                summary_rows.append({"sigma": sigma, "comparison": comparison, "minimum_coverage": cut, **result})
        for (sigma, scenario), f in scenarios[scenarios.covered_area_fraction.ge(cut)].groupby(["sigma", "scenario"]):
            for metric in ("pulse_peak", "pulse_central80"):
                result = cluster_mean(f, metric, "component", draws=draws, contrast=True)
                if result:
                    form_rows.append({"sigma": sigma, "scenario": scenario, "minimum_coverage": cut, **result})
    tables = {"mechanism_scenarios": scenarios, "mechanism_paired_changes": contrast,
        "mechanism_summary": pd.DataFrame(summary_rows), "mechanism_form_contrasts": pd.DataFrame(form_rows),
        "junction_null": pd.DataFrame(identities), "mechanism_example": pd.concat(waves, ignore_index=True)}
    summary = {"receivers": len(panel), "systems": panel.component.nunique(), "scenarios": len(scenarios),
        "max_junction_partition_error": float(pd.DataFrame(identities).partition_error.max()),
        "max_variance_decomposition_error": float(abs(scenarios.pulse_sd**2-scenarios.input_variance-scenarios.arrival_variance-scenarios.corridor_variance).max()),
        "max_gain_error": float(abs(scenarios.anomaly_mass_fraction-1).max()),
        "max_passive_peak": float(scenarios.pulse_peak.max())}
    return tables, summary, [reaches_path, context_path]


def observational_replication(*, draws=5000):
    path = OBSERVED / "analysis/network_metrics.csv"
    f = pd.read_csv(path, dtype=TYPES)
    f = f.query("version == 'monthly' and adjustment == 'calendar_year' and excursion_quantile == .75")
    rows = []
    for coverage in (0., .8):
        eligible = f[f.covered_area_fraction.ge(coverage) & f.peak_comparison_eligible].copy()
        for omitted in (-1, *sorted(eligible.component.unique())):
            g = eligible if omitted == -1 else eligible[eligible.component.ne(omitted)]
            result = cluster_mean(g, "peak_risk_difference", "component", draws=draws)
            if result:
                rows.append({"minimum_coverage": coverage, "omitted_system": int(omitted),
                    "n_positive_receivers": int(g.peak_risk_difference.gt(0).sum()),
                    "coincident_probability": g.receiver_high_given_coincident.mean(),
                    "solo_probability": g.receiver_high_given_solo.mean(), **result})
    return {"observed_leave_system_out": pd.DataFrame(rows)}, [path]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    args = p.parse_args()
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    tables, recovered, files = recover_analysis()
    primary, mechanism, inputs = mechanism_analysis(draws=args.bootstrap_draws)
    tables.update(primary)
    files.extend(inputs)
    shortest, shortest_summary, inputs = mechanism_analysis(shortest=True, draws=args.bootstrap_draws)
    files.extend(inputs)
    tables.update({"shortest_"+k: v for k, v in shortest.items()})
    observed, inputs = observational_replication(draws=args.bootstrap_draws)
    tables.update(observed)
    files.extend(inputs)
    for name in ("catalog_discovery.json", "additional_retrieval.json", "additional_summary.json"):
        files.append(ROOT / "recovery" / name)
    discovery = json.loads((ROOT / "recovery/catalog_discovery.json").read_text())
    additional = json.loads((ROOT / "recovery/additional_retrieval.json").read_text())
    for record in [*discovery["records"], *additional["records"], additional["continuous_doc_catalogue"]]:
        if "path" in record:
            files.append(Path(record["path"]))
    for path in sorted(out.glob("additional_*")):
        files.append(path)
    written = []
    for name, frame in tables.items():
        path = out / f"{name}.{'parquet' if name in ('recovered_activities', 'recovered_selected_activities', 'recovered_signal_series', 'mechanism_example', 'shortest_mechanism_example') else 'csv'}"
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        written.append(path)
    summary = {"recovery": recovered,
        "additional_recovery": json.loads((ROOT / "recovery/additional_summary.json").read_text()),
        "current_continuous_doc_catalogue": additional["continuous_doc_catalogue"],
        "mechanism": mechanism, "shortest_mechanism": shortest_summary,
        "bootstrap_draws": args.bootstrap_draws, "new_model_training": False,
        "new_field_measurements": False, "existing_field_records_retrieved": True,
        "time_unit": "cropped mean path divided by imposed uniform speed, never estimated from DOC"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int)+"\n")
    written.append(out / "summary.json")
    for path in CODE:
        target = ROOT / "code_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    ledger = {"source_hashes": {str(p): sha256_file(p) for p in [*set(files), ROOT / "study_plan.md", *CODE]},
        "output_hashes": {str(p): sha256_file(p) for p in written},
        "sigma_grid": SIGMAS, "receiver_weighting": "equal", "bootstrap_unit": "overlapping catchment system",
        "data_scope_change": "new full public activity archive, not prediction-model training or evaluation", "models_trained": False}
    (ROOT / "analysis_sources.json").write_text(json.dumps(ledger, indent=2)+"\n")
    print(json.dumps(summary, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
