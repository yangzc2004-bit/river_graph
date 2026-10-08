"""Measure full mainstem tributary layout and separate arrival-spreading mechanisms."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_junction_layout import (
    arrival_scenarios,
    condition_geometric_mainstem,
    partition_mainstem,
)
from river_graph.analysis.river_morphology_effect import (
    group_descriptor_summary,
    paired_response_summary,
)
from river_graph.analysis.river_whole_storage import StoragePaths
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_junction_layout_v1")
BASE = Path("experiments/phase4_transfer")
PANEL = BASE/"doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv"
PAIRS = PANEL.parent/"covariate_selected_pairs.csv"
REPS = BASE/"doc_river_storage_placement_v1/analysis/representatives.csv"
VAA = Path("cache/nldplus_vaa.parquet")
ROUTES = Path("data/raw/river_source_placement_v1/routing")
MEMBERS = Path("data/raw/river_planform_v1/members_full")
COLS = ["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm",
        "wbareatype", "wbareacomi", "arbolatesu"]
TYPES = {"station": str, "huc4": str, "station_a": str, "station_b": str}
CODE = tuple(map(Path, (
    "src/river_graph/analysis/river_junction_layout.py",
    "src/river_graph/analysis/river_whole_storage.py",
    "src/river_graph/analysis/river_routing_mechanisms.py",
    "src/river_graph/analysis/river_morphology_effect.py",
    "src/river_graph/topology/river_planform.py",
    "scripts/analyze_doc_river_junction_layout_v1.py",
    "scripts/plot_doc_river_junction_layout_v1.py",
    "scripts/verify_doc_river_junction_layout_v1.py",
    "scripts/plot_doc_river_planform_v1.py", "tests/test_river_junction_layout.py")))
GEOMETRY_METRICS = ("lateral_area_fraction", "lateral_entry_mean", "lateral_entry_sd",
    "lower_third_lateral_share", "effective_area_weighted_entries", "path_cv",
    "within_unit_variance_fraction", "between_unit_variance_fraction",
    "entry_trunk_variance_fraction", "mean_branch_variance_fraction",
    "entry_branch_covariance_fraction", "entry_branch_mean_correlation")
RESPONSE_METRICS = ("remove_within_peak_change_pct", "align_means_peak_change_pct",
                    "remove_within_duration_change_pct", "align_means_duration_change_pct")
DOC_METRICS = ("doc_median", "doc_cv", "harmonic_amplitude", "q90_fraction")


def context():
    panel = pd.read_csv(PANEL, dtype=TYPES)
    pairs = pd.read_csv(PAIRS, dtype=TYPES)
    pairs = pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].reset_index(drop=True)
    reps = pd.read_csv(REPS, dtype=TYPES)
    reps = reps[reps.example_group.le(3)].copy()
    if len(panel) != 297 or panel.station.duplicated().any() or len(pairs) != 22 or len(reps) != 3:
        raise ValueError("preserve original 297 networks / 22 matches / 3 representatives")
    return panel, pairs, reps


def load_layout(row, vaa, *, geometric_mainstem=False):
    route = ROUTES/f"comid_{int(row.comid)}.npz"
    members = MEMBERS/f"comid_{int(row.comid)}.npz"
    with np.load(route) as z:
        cid, distance = z["comids"], z["distance_km"]
    with np.load(members) as z:
        original = z["mainstem"]
        if not np.array_equal(np.sort(cid), np.sort(z["comids"])):
            raise ValueError("complete mapped and routed membership must agree")
    reachable = np.isfinite(distance)
    frame = vaa.loc[cid[reachable]]
    paths = StoragePaths(frame, int(row.comid), distance[reachable])
    conditioned = None
    if geometric_mainstem:
        conditioned = condition_geometric_mainstem(paths, frame.hydroseq, original, frame.dnhydroseq, frame.dnminorhyd)
    layout = partition_mainstem(paths, frame.hydroseq.to_numpy(), frame.arbolatesu.to_numpy(),
                                geometric_mainstem=original)
    np.testing.assert_allclose(layout.descriptors["path_cv"], row.route_distance_cv, atol=1e-9)
    if conditioned is not None:
        saved_mismatch = layout.descriptors["original_mainstem_link_mismatches"]
        layout = partition_mainstem(conditioned, frame.hydroseq.to_numpy(), frame.arbolatesu.to_numpy(),
                                    geometric_mainstem=original, prescribed_mainstem=original)
        weight = paths.area/paths.area.sum()
        layout.descriptors.update({"saved_mainstem_link_mismatches": saved_mismatch,
            "changed_successor_links": int(np.sum(paths.successor != conditioned.successor)),
            "mean_path_increase_fraction": float(weight@(conditioned.distance-paths.distance)/(weight@paths.distance)),
            "area_fraction_with_changed_route": float(weight@(conditioned.distance > paths.distance+1e-7))})
    layout.descriptors["routed_area_fraction"] = frame.areasqkm.sum()/vaa.loc[cid].areasqkm.sum()
    layout.descriptors["n_unreachable_reaches"] = int((~reachable).sum())
    return layout, [route, members]


def build_analysis(panel, pairs, reps, vaa, *, draws=5000, geometric_mainstem=False):
    rows, units, junctions, mains, responses, fine, curves, files = [], [], [], [], [], [], [], []
    for i, row in enumerate(panel.itertuples()):
        layout, source_files = load_layout(row, vaa, geometric_mainstem=geometric_mainstem)
        files.extend(source_files)
        meta = {"station": row.station, "comid": int(row.comid), "huc4": row.huc4, "cluster": int(row.cluster)}
        pulse, trace = arrival_scenarios(layout, keep_curves=row.station in set(reps.station))
        sensitivity, _ = arrival_scenarios(layout, dt=.0025)
        for f, output in ((layout.units, units), (layout.junctions, junctions), (pulse, responses), (sensitivity, fine)):
            output.append(f.assign(**meta))
        mains.append(pd.DataFrame({"station": row.station, "comid": row.comid,
                                  "mainstem_comid": layout.mainstem_comids,
                                  "sequence_outlet_to_head": np.arange(len(layout.mainstem_comids))}))
        if not trace.empty:
            curves.append(trace.assign(**meta))
        by_scenario = pulse.set_index("scenario")
        response = {}
        for scenario, prefix in (("within_unit_collapsed", "remove_within"), ("unit_means_aligned", "align_means")):
            response[prefix+"_peak_change_pct"] = by_scenario.at[scenario, "peak_change_vs_actual_pct"]
            response[prefix+"_duration_change_pct"] = by_scenario.at[scenario, "duration_change_vs_actual_pct"]
        rows.append({**meta, **layout.descriptors, **response, **{m: getattr(row, m) for m in DOC_METRICS}})
        if (i+1) % 40 == 0 or i+1 == len(panel):
            print(f"Partitioned and routed {i+1}/{len(panel)} complete networks", flush=True)
    descriptor = pd.DataFrame(rows)
    metrics = (*GEOMETRY_METRICS, *RESPONSE_METRICS, *DOC_METRICS)
    grouped = group_descriptor_summary(descriptor, metrics, draws=draws)
    matched, evidence = paired_response_summary(pairs, descriptor, dict.fromkeys(metrics, "raw"), draws=draws)
    for f in (grouped, matched, evidence):
        f["evidence_type"] = np.where(f.metric.isin(DOC_METRICS), "copied observed monthly DOC context",
                                      np.where(f.metric.isin(RESPONSE_METRICS), "controlled same-input timing scenario",
                                               "mapped whole-network routing geometry"))
    responses = pd.concat(responses, ignore_index=True)
    fine = pd.concat(fine, ignore_index=True)
    sensitivity = responses.merge(fine, on=["station", "scenario"], suffixes=("", "_fine"), validate="one_to_one")
    sensitivity = sensitivity[["station", "scenario", "pulse_peak", "pulse_peak_fine", "duration_80", "duration_80_fine"]]
    sensitivity["peak_abs_change"] = abs(sensitivity.pulse_peak_fine-sensitivity.pulse_peak)
    sensitivity["duration_abs_change"] = abs(sensitivity.duration_80_fine-sensitivity.duration_80)
    return {"station_junction_layout": descriptor, "tributary_units": pd.concat(units, ignore_index=True),
            "entry_junctions": pd.concat(junctions, ignore_index=True), "route_mainstems": pd.concat(mains, ignore_index=True),
            "arrival_scenarios": responses, "fine_resolution_scenarios": fine, "numerical_sensitivity": sensitivity,
            "form_summary": grouped, "matched_form_contrasts": matched, "matched_pair_evidence": evidence,
            "representative_curves": pd.concat(curves, ignore_index=True),
            "representatives": reps[["station", "selection", "example_group"]].merge(descriptor, on="station", validate="one_to_one")}, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--geometric-mainstem", action="store_true", help="separate original mapped-trunk routing sensitivity")
    args = parser.parse_args()
    root = ROOT/"original_mainstem_sensitivity" if args.geometric_mainstem else ROOT
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    config = {"previous_results_seen": True, "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
        "bootstrap_unit": "whole HUC4", "aggregation": "networks or original matched pairs equally weighted",
        "mainstem_definition": "largest arbolatesu parent on the saved single-successor routing tree; COMID tie break",
        "source_weights": "unique incremental catchment areas", "entry_coordinate": "remaining mainstem length / full mainstem length",
        "pulse_sd": .15, "time_unit": "original network mean path / imposed uniform speed",
        "dt": .005, "sensitivity_dt": .0025, "new_DOC_measurements": False, "neural_training": False}
    config["geometric_mainstem_conditioned"] = args.geometric_mainstem
    if args.geometric_mainstem:
        config["mainstem_definition"] = "original geometric mainstem; original real downstream links forced only on trunk; off-trunk saved successors fixed"
        config["design_revision_after_preliminary_result"] = "old-versus-routing mainstem overlap median 0.429; original trunk routing added as a separate sensitivity, no result discarded"
    (root/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    panel, pairs, reps = context()
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    tables, files = build_analysis(panel, pairs, reps, vaa, draws=args.bootstrap_draws, geometric_mainstem=args.geometric_mainstem)
    products = []
    for name, f in tables.items():
        path = out/f"{name}.parquet" if name in ("tributary_units", "representative_curves") else out/f"{name}.csv"
        if path.suffix == ".parquet":
            f.to_parquet(path, index=False)
        else:
            f.to_csv(path, index=False)
        products.append(path)
    f = tables["station_junction_layout"]
    summary = {"n_networks": len(f), "n_unique_networks": int(f.comid.nunique()), "n_huc4": int(f.huc4.nunique()),
        "class_counts": {str(k): int(v) for k, v in f.groupby("cluster").size().items()},
        "n_matched_pairs": len(pairs), "n_matched_huc4": int(pairs.huc4.nunique()),
        "n_original_mainstem_links_mismatched": int(f.original_mainstem_link_mismatches.sum()),
        "n_networks_with_original_mainstem_mismatch": int(f.original_mainstem_link_mismatches.gt(0).sum()),
        "n_networks_without_lateral_units": int(f.n_lateral_units.eq(0).sum()),
        "n_total_lateral_units": len(tables["tributary_units"].query("unit_kind == 'lateral' and area_km2 > 0")),
        "n_entry_junction_instances": len(tables["entry_junctions"]),
        "n_physical_entry_junctions": int(tables["entry_junctions"].entry_comid.nunique()),
        "max_variance_partition_error_km2": float(abs(f.path_variance_km2-f.within_unit_variance_km2-f.between_unit_variance_km2).max()),
        "max_nested_variance_error_km2": float(abs(f.between_unit_variance_km2-f.entry_trunk_variance_km2-f.mean_branch_variance_km2-f.twice_entry_branch_covariance_km2).max()),
        "max_peak_resolution_change": float(tables["numerical_sensitivity"].peak_abs_change.max()),
        "max_duration_resolution_change": float(tables["numerical_sensitivity"].duration_abs_change.max())}
    if args.geometric_mainstem:
        summary.update({"n_saved_mainstem_links_mismatched": int(f.saved_mainstem_link_mismatches.sum()),
                        "mean_changed_route_area_fraction": float(f.area_fraction_with_changed_route.mean()),
                        "median_mean_path_increase_fraction": float(f.mean_path_increase_fraction.median())})
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    products.append(out/"summary.json")
    for path in CODE:
        destination = root/"code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    sources = [ROOT/"study_plan.md", PANEL, PAIRS, REPS, VAA, *files, *CODE]
    ledger = {"source_hashes": {str(p): sha256_file(p) for p in sources},
              "config_sha256": sha256_file(root/"config.json"),
              "output_hashes": {str(p): sha256_file(p) for p in products}, "new_training": False}
    (root/"analysis_sources.json").write_text(json.dumps(ledger, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    print(tables["form_summary"].query("metric in ['lateral_entry_mean', 'lateral_entry_sd', 'between_unit_variance_fraction', 'entry_branch_mean_correlation']").to_string(index=False))
    print(tables["matched_form_contrasts"].to_string(index=False))


if __name__ == "__main__":
    main()
