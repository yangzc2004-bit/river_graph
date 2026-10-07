"""Connect fixed river forms, real storage positions and existing DOC evidence."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_morphology_effect import paired_response_summary
from river_graph.analysis.river_storage_placement import selected_corridor
from river_graph.analysis.river_structure_profiles import (
    assemble_profiles,
    corridor_storage_profile,
    storage_position_counts,
    summarize_profiles,
)
from river_graph.analysis.river_whole_storage import StoragePaths
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_structure_profiles_v1")
BASE = Path("experiments/phase4_transfer")
MORPH = BASE/"doc_river_morphology_effect_v1/analysis"
WHOLE = BASE/"doc_river_whole_storage_v1/analysis"
PLACEMENT = BASE/"doc_river_storage_placement_v1/analysis"
INTERNAL = BASE/"doc_river_internal_structure_v1/analysis"
MONITORED = BASE/"doc_monitored_river_footprint_v1/analysis"
FLOW = BASE/"doc_river_form_flow_response_v1/analysis"
SAMPLING = BASE/"doc_river_sampling_resolution_v1/analysis"
VAA = Path("cache/nldplus_vaa.parquet")
ROUTES = Path("data/raw/river_source_placement_v1/routing")
COLS = ["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm", "wbareatype", "wbareacomi"]
CODE = tuple(Path(p) for p in (
    "src/river_graph/analysis/river_structure_profiles.py", "src/river_graph/analysis/river_storage_placement.py",
    "src/river_graph/analysis/river_whole_storage.py", "src/river_graph/analysis/river_morphology_effect.py",
    "scripts/analyze_doc_river_structure_profiles_v1.py", "scripts/plot_doc_river_structure_profiles_v1.py",
    "scripts/verify_doc_river_structure_profiles_v1.py", "tests/test_river_structure_profiles.py"))
PROFILE_METRICS = ("arrival_dispersion", "major_common_share", "major_arrival_dispersion", "major_pair_area_share",
                   "mapped_storage_path_share", "mapped_storage_exposure", "corridor_storage_path_share",
                   "common_storage_fraction", "shared_storage_budget_fraction", "relative_time_pulse_peak",
                   "relative_time_pulse_sd", "doc_cv", "doc_median", "harmonic_amplitude", "q90_fraction")
MATCH_METRICS = ("arrival_dispersion", "major_common_share", "mapped_storage_path_share",
                 "mapped_storage_exposure", "relative_time_pulse_peak", "relative_time_pulse_sd",
                 "doc_cv", "doc_median", "harmonic_amplitude", "q90_fraction")
SOURCES = (MORPH/"station_morphology_doc_panel.csv", MORPH/"covariate_selected_pairs.csv",
           WHOLE/"network_descriptors.csv", PLACEMENT/"selected_footprints.csv", PLACEMENT/"representatives.csv",
           INTERNAL/"normalized_routing.csv", MONITORED/"observed_receivers.csv", MONITORED/"structural_associations.csv",
           FLOW/"class_flow_responses.csv", FLOW/"station_flow_responses.csv", SAMPLING/"summary.json",
           ROOT/"study_plan.md", VAA)


def read_context():
    types = {"station": str, "huc4": str, "station_a": str, "station_b": str}
    panel = pd.read_csv(SOURCES[0], dtype=types)
    pairs = pd.read_csv(SOURCES[1], dtype=types)
    pairs = pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].reset_index(drop=True)
    whole = pd.read_csv(SOURCES[2], dtype=types)
    selected = pd.read_csv(SOURCES[3], dtype=types)
    selected = selected[selected.cohort.eq("whole_confluence")].reset_index(drop=True)
    routing = pd.read_csv(INTERNAL/"normalized_routing.csv")
    if len(panel) != 297 or len(selected) != 295 or len(pairs) != 22:
        raise ValueError("retain the fixed 297 / 295 / 22 cohorts")
    return panel, pairs, whole, selected, routing


def partition_selected(selected, vaa):
    rows, receipts = [], []
    for i, row in enumerate(selected.itertuples()):
        path = ROUTES/f"comid_{int(row.comid)}.npz"
        source = np.load(path)
        frame = vaa.loc[source["comids"]]
        valid = np.isfinite(source["distance_km"])
        frame = frame[valid]
        paths = StoragePaths(frame, int(row.comid), source["distance_km"][valid], scale=1.)
        selection = {name: int(getattr(row, name)) for name in ("junction_index", "source_a_index", "source_b_index")}
        # The original selection and representatives are reused, not reselected.
        for index_name, id_name in (("junction_index", "junction_comid"), ("source_a_index", "source_a_comid"),
                                   ("source_b_index", "source_b_comid")):
            if int(paths.comids[selection[index_name]]) != int(getattr(row, id_name)):
                raise ValueError("saved source/junction identity must reproduce on the saved routes")
        corridor = selected_corridor(paths, selection)
        part = corridor_storage_profile(corridor, frame.reset_index(drop=True), row.weight_a)
        np.testing.assert_allclose([part[n] for n in ("branch_a_km", "branch_b_km", "common_km")],
                                   [row.branch_a_km, row.branch_b_km, row.common_km], atol=1e-8, rtol=1e-10)
        rows.append({"station": row.station, "comid": int(row.comid), "cluster": int(row.cluster),
                     "huc4": row.huc4, **part})
        receipts.append(path)
        if (i+1) % 50 == 0 or i+1 == len(selected):
            print(f"Mapped actual storage on {i+1}/{len(selected)} fixed confluences", flush=True)
    return pd.DataFrame(rows), receipts


def build_tables(panel, whole, selected, partition, routing, pairs, draws):
    profiles = assemble_profiles(panel, whole, selected, partition, routing)
    summary = summarize_profiles(profiles, PROFILE_METRICS, draws=draws)
    fields = {"doc_cv", "doc_median", "harmonic_amplitude", "q90_fraction"}
    summary["evidence_type"] = np.where(summary.metric.isin(fields), "observed monthly DOC",
        np.where(summary.metric.str.startswith("relative_time"), "identical-input geometry scenario", "mapped geometry"))
    matched, evidence = paired_response_summary(pairs, profiles, dict.fromkeys(MATCH_METRICS, "raw"), draws=draws)
    matched["evidence_type"] = np.where(matched.metric.isin(fields), "observed monthly DOC",
        np.where(matched.metric.str.startswith("relative_time"), "identical-input geometry scenario", "mapped geometry"))
    field = pd.read_csv(FLOW/"class_flow_responses.csv")
    field = field[field.population.eq("observed_flow") & field.metric.eq("adjusted_log_contrast")].copy()
    receivers = pd.read_csv(MONITORED/"observed_receivers.csv", dtype={"target": str, "huc4": str})
    associations = pd.read_csv(MONITORED/"structural_associations.csv")
    return {"station_mechanism_profiles": profiles, "mapped_storage_partitions": partition,
            "form_profile_summary": summary, "storage_position_counts": storage_position_counts(profiles),
            "matched_chain_summary": matched, "matched_chain_evidence": evidence,
            "field_flow_context": field, "monitored_receiver_context": receivers,
            "monitored_structural_context": associations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--tables-only", action="store_true", help="reuse mapped partitions to rebuild statistical tables")
    args = parser.parse_args()
    output = ROOT/"analysis"
    output.mkdir(parents=True, exist_ok=True)
    config = {"bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
              "bootstrap_unit": "HUC4", "aggregation": "station instances / matched pairs equal weight",
              "fixed_outline_classes": {"1": "elongated tributary-rich", "2": "mainstem-dominated sparse", "3": "broad tributary-rich"},
              "previous_results_seen": True, "new_outcome_regression": False, "neural_training": False,
              "mapped_storage_types": ["LakePond", "Reservoir"], "source_weights": "fixed incremental catchment area proxies",
              "selected_path_source": "previous geometry-selected representative midpoints and major junctions",
              "routing_source": "saved all-source mean-delay-one Gaussian SD0.15 scenarios",
              "field_scope": "source-role ST357 monthly DOC; no event travel-time fit or external validation"}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    panel, pairs, whole, selected, routing = read_context()
    if args.tables_only:
        partition = pd.read_csv(output/"mapped_storage_partitions.csv", dtype={"station": str, "huc4": str})
        paths = [ROUTES/f"comid_{int(r.comid)}.npz" for r in selected.itertuples()]
    else:
        vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
        partition, paths = partition_selected(selected, vaa)
    tables = build_tables(panel, whole, selected, partition, routing, pairs, args.bootstrap_draws)
    for name, frame in tables.items():
        frame.to_csv(output/f"{name}.csv", index=False)
    reps = pd.read_csv(PLACEMENT/"representatives.csv", dtype={"station": str})
    reps = reps[reps.example_group.le(3)][["station", "selection", "example_group"]].merge(
        tables["station_mechanism_profiles"], on="station", validate="one_to_one")
    reps.to_csv(output/"representatives.csv", index=False)
    sampling = json.loads((SAMPLING/"summary.json").read_text())
    (output/"sampling_context.json").write_text(json.dumps(sampling, indent=2)+"\n")
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    outputs = sorted(output.glob("*.csv"))+[output/"sampling_context.json"]
    hashes = {str(p): sha256_file(p) for p in (*SOURCES, *paths, *CODE)}
    ledger = {"source_hashes": hashes, "config_sha256": sha256_file(ROOT/"config.json"),
              "output_hashes": {str(p): sha256_file(p) for p in outputs},
              "n_station_instances": len(panel), "n_unique_networks": panel.comid.nunique(),
              "n_selected_corridors": len(partition), "n_huc4": panel.huc4.nunique(),
              "n_matched_pairs": len(pairs), "n_matched_huc4": pairs.huc4.nunique(),
              "n_monitored_receivers": len(tables["monitored_receiver_context"]), "new_model_training": False}
    (ROOT/"analysis_sources.json").write_text(json.dumps(ledger, indent=2)+"\n")
    print(tables["form_profile_summary"].query("metric in ['arrival_dispersion', 'major_common_share', 'mapped_storage_path_share', 'doc_cv']").to_string(index=False))
    print(tables["storage_position_counts"].to_string(index=False))
    print(tables["matched_chain_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
