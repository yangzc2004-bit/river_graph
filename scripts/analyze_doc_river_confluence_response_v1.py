"""Recompute two complementary real-confluence observations from public files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_confluence_response import (
    channel_geometry,
    confluence_chemistry,
    descriptive_geometry_relations,
    summarize_water_window,
)
from river_graph.analysis.river_event_observations import read_sites_csv
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_confluence_response_v1")
PARENT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
RAW = Path("data/raw/river_confluence_response_v1/plont")


def build():
    manifest = json.loads((ROOT / "retrieval_manifest.json").read_text())
    windows = pd.read_csv(PARENT / "analysis/confluence_event_coverage.csv")
    configurations = pd.read_csv(PARENT / "analysis/monitored_confluences.csv")
    lab = pd.read_parquet(PARENT / "analysis/laboratory_doc.parquet")
    flow, flow_audit = {}, []
    for item in manifest["objects"]:
        if item["kind"] != "flow_30min":
            continue
        path = Path(item["path"])
        preamble = path.read_text().split("TIMESTAMP,")[0]
        if "TIME RESOLUTION: 30 MINUTES" not in preamble:
            raise ValueError(f"Unexpected water resolution: {path}")
        methods = [line for line in preamble.splitlines() if line.lower().startswith("q -")]
        if len(methods) != 1:
            raise ValueError(f"Missing discharge method: {path}")
        frame = read_sites_csv(path, "Q")
        off_grid = (frame.timestamp_local.dt.minute.mod(30).ne(0)
                    | frame.timestamp_local.dt.second.ne(0))
        flow_audit.append({"site": item["site"], "n_source_rows": len(frame),
                           "n_valid_q": int(frame.value.notna().sum()), "n_off_halfhour_grid": int(off_grid.sum()),
                           "first_time": frame.timestamp_local.min(), "last_time": frame.timestamp_local.max(),
                           "discharge_method": methods[0], "source_url": item["url"]})
        if off_grid.any():
            raise ValueError(f"Off-grid flow timestamps require explicit resolution: {path}")
        flow[item["site"]] = frame.set_index("timestamp_local").value
    water_rows, clocks = [], []
    for window in windows.to_dict("records"):
        wid = f"{window['source_a']}__{window['source_b']}__{window['receiver']}__{window['year']}"
        start = pd.Timestamp(window["start_date"])
        stop = pd.Timestamp(window["end_date"])+pd.Timedelta(days=1)
        clock = pd.date_range(start, stop, freq="30min", inclusive="left")
        aligned = [flow[window[key]].reindex(clock).to_numpy() for key in ("source_a", "source_b", "receiver")]
        water_rows.append({**window, "window_id": wid, **summarize_water_window(clock, *aligned)})
        clocks.append(pd.DataFrame({"window_id": wid, "timestamp_local": clock,
                                    "q_a_m3s": aligned[0], "q_b_m3s": aligned[1], "q_receiver_m3s": aligned[2]}))
    water = pd.DataFrame(water_rows)
    config_rows = []
    for config in configurations.to_dict("records"):
        sub = water
        for key in ("source_a", "source_b", "receiver"):
            sub = sub[sub[key].eq(config[key])]
        ready = sub[sub.primary_coverage_subset]
        complete = sub[sub.full_joint_clock]
        config_rows.append({**config, "n_preserved_windows": len(sub), "n_primary_water_windows": len(ready),
                            "n_full_clock_windows": len(complete),
                            "median_wave_overlap": ready.wave_overlap.median(),
                            "median_incoming_peak_coincidence": ready.incoming_peak_coincidence.median(),
                            "median_amplitude_balanced_peak_coincidence": ready.amplitude_balanced_peak_coincidence.median(),
                            "median_branch_flow_share": ready.branch_flow_share_median.median(),
                            "full_clock_median_overlap": complete.wave_overlap.median(),
                            "full_clock_median_peak_coincidence": complete.incoming_peak_coincidence.median()})
    chemistry, lateral = confluence_chemistry(pd.read_csv(RAW / "Plontetal_WRR_database.csv"))
    geometry_raw = pd.read_csv(RAW / "Plontetal_WRR_Fall2021_widthdepth_notrib.csv")
    geometry, transects = channel_geometry(geometry_raw)
    author = pd.read_csv(RAW / "Plontetal_WRR_Fall2021_sumdata.csv").rename(columns={
        "site": "confluence", "rtnorm.min": "receiver_residence_100m_min",
        "rtnorm.main.min": "main_residence_100m_min", "width.cv": "author_receiver_width_cv_pct",
        "width.main.cv": "author_main_width_cv_pct"})
    geometry = geometry.merge(author[["confluence", "receiver_residence_100m_min", "main_residence_100m_min",
                                     "author_receiver_width_cv_pct", "author_main_width_cv_pct"]],
                              on="confluence", validate="one_to_one")
    geometry["receiver_main_residence_ratio"] = geometry.receiver_residence_100m_min/geometry.main_residence_100m_min
    geometry["width_cv_replay_difference_receiver"] = geometry.receiver_width_cv_pct-geometry.author_receiver_width_cv_pct
    geometry["width_cv_replay_difference_main"] = geometry.main_width_cv_pct-geometry.author_main_width_cv_pct
    fall = chemistry[chemistry.season.eq("fall")].merge(geometry, on="confluence", validate="one_to_one")
    relations = descriptive_geometry_relations(fall)
    seasons = chemistry.pivot(index="confluence", columns="season", values="doc_receiver_minus_mix_pct").reset_index()
    seasons["sign_changed"] = np.sign(seasons.fall) != np.sign(seasons.summer)
    primary = water[water.primary_coverage_subset]
    summary = {
        "n_flow_sites": len(flow_audit), "n_original_water_windows": len(water),
        "n_primary_water_windows": len(primary), "n_full_clock_water_windows": int(water.full_joint_clock.sum()),
        "n_water_configurations_with_primary_coverage": int(primary.receiver.nunique()),
        "median_water_wave_overlap": float(primary.wave_overlap.median()),
        "median_water_peak_coincidence": float(primary.incoming_peak_coincidence.median()),
        "n_field_confluences": int(chemistry.confluence.nunique()), "n_field_campaigns": len(chemistry),
        "n_field_lateral_positions": len(lateral), "n_channel_depth_points": len(geometry_raw),
        "n_channel_transects": len(transects),
        "n_missing_survey_depth_points": int(transects.n_missing_depth_points.sum()),
        "n_survey_units_with_conflicting_distance": int(transects.conflicting_distance.sum()),
        "n_receiver_wider_than_main": int(geometry.receiver_main_width_ratio.gt(1).sum()),
        "n_receiver_deeper_than_main": int(geometry.receiver_main_depth_ratio.gt(1).sum()),
        "n_receiver_longer_normalized_residence": int(geometry.receiver_main_residence_ratio.gt(1).sum()),
        "n_confluences_with_doc_sign_change": int(seasons.sign_changed.sum()),
        "n_doc_mix_inside_lateral_range": int(chemistry.doc_mix_within_lateral_range.sum()),
        "n_source_weight_campaigns_differ_from_normalized": int(chemistry.source_weights_differ_from_normalized.sum()),
        "doc_deviation_min_pct": float(chemistry.doc_receiver_minus_mix_pct.min()),
        "doc_deviation_max_pct": float(chemistry.doc_receiver_minus_mix_pct.max()),
        "water_scope": "observed seasonal water-wave overlap at partial mapped branches; not DOC travel times",
        "field_scope": "five serial confluences in one stream network; lateral arithmetic means are not flux-weighted cross-sections",
    }
    tables = {"water_window_comparison": water, "flow_source_quality": pd.DataFrame(flow_audit),
              "water_configuration_summary": pd.DataFrame(config_rows), "confluence_chemistry": chemistry,
              "lateral_doc_points": lateral, "channel_geometry": geometry, "surveyed_transects": transects,
              "fall_geometry_doc": fall, "descriptive_geometry_relations": relations, "season_comparison": seasons}
    products = {"water_clocks": pd.concat(clocks, ignore_index=True), "laboratory_doc_points": lab}
    return tables, products, summary


def main():
    tables, products, summary = build()
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in products.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    sources = [PARENT / "analysis" / name for name in ("confluence_event_coverage.csv", "monitored_confluences.csv", "laboratory_doc.parquet")]
    sources += [Path(item["path"]) for item in json.loads((ROOT / "retrieval_manifest.json").read_text())["objects"]]
    sources += [ROOT / "retrieval_manifest.json", ROOT / "study_plan.md", Path("scripts/analyze_doc_river_confluence_response_v1.py"),
                Path("src/river_graph/analysis/river_confluence_response.py"), Path("src/river_graph/analysis/river_event_observations.py")]
    for path in sources:
        if str(path).startswith(("scripts/", "src/")):
            target = ROOT / "code_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    outputs = list(output.glob("*.csv"))+list(output.glob("*.parquet"))+[output / "summary.json"]
    (ROOT / "analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(p): sha256_file(p) for p in sources},
        "output_hashes": {str(p): sha256_file(p) for p in outputs}}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
