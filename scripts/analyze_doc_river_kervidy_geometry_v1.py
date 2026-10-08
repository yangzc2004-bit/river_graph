"""Join Kervidy UTC records and locate its gauge in the actual mapped basin."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

from river_graph.analysis.river_kervidy_observations import (
    FLOW_CAP_DM3_S,
    flow_selected_windows,
    join_doc_flow,
    occupied_record_coverage,
    prepare_flow,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
RAW = Path("data/raw/river_kervidy_geometry_v1")
CHEM = Path("data/raw/river_external_event_catalog_v1/kervidy_spectro.txt")


def main():
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    chemistry = pd.read_csv(CHEM, sep=";", decimal=",")
    doc = chemistry.loc[np.isfinite(chemistry.DOCcor), ["time", "DOCcor"]].rename(
        columns={"time": "timestamp_utc", "DOCcor": "doc_mg_l"})
    doc["timestamp_utc"] = pd.to_datetime(doc.timestamp_utc, format="mixed", utc=True)
    flow = prepare_flow(pd.read_parquet(RAW / "discharge_quarter_hour.parquet"))
    joined = join_doc_flow(doc, flow)
    joined.to_parquet(out / "corrected_doc_flow_matches.parquet", index=False)
    windows = flow_selected_windows(flow, doc.timestamp_utc.min(), doc.timestamp_utc.max())
    rows = []
    for w in windows.itertuples():
        d = doc.loc[doc.timestamp_utc.between(w.window_start_utc, w.window_end_utc)]
        q = flow.loc[flow.flow_valid & flow.flow_timestamp_utc.between(w.window_start_utc, w.window_end_utc)]
        j = joined.loc[joined.flow_match & joined.timestamp_utc.between(w.window_start_utc, w.window_end_utc)]
        metrics = {**w._asdict()}
        for name, clock in (("doc", d.timestamp_utc), ("flow", q.flow_timestamp_utc), ("joint", j.timestamp_utc)):
            metrics.update({f"{name}_{k}": v for k, v in occupied_record_coverage(
                clock, w.window_start_utc, w.window_end_utc).items()})
        metrics.update({"n_corrected_doc": len(d), "n_flow_records": len(q), "n_joint_records": len(j),
            "doc_observed_window_peak_mg_l": float(d.doc_mg_l.max()) if len(d) else np.nan,
            "n_flow_cap_records": int(q.flow_at_reported_cap.sum()),
            "window_inside_flow_scope": w.window_start_utc >= flow.flow_timestamp_utc.min()
                and w.window_end_utc <= flow.flow_timestamp_utc.max(),
            "window_inside_corrected_doc_span": w.window_start_utc >= doc.timestamp_utc.min()
                and w.window_end_utc <= doc.timestamp_utc.max()})
        metrics["dense_joint_window"] = all(metrics[f"{key}_coverage"] >= .9 and
            metrics[f"{key}_max_gap_hours"] <= 1 for key in ("doc", "flow", "joint"))
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(out / "flow_selected_windows.csv", index=False)
    yearly = []
    for year, q in flow.groupby(flow.flow_timestamp_utc.dt.year):
        d, j = doc.loc[doc.timestamp_utc.dt.year.eq(year)], joined.loc[joined.timestamp_utc.dt.year.eq(year)]
        yearly.append({"year": year, "n_flow_records": len(q), "n_valid_flow": int(q.flow_valid.sum()),
            "n_zero_flow": int(q.q_dm3_s.eq(0).sum()), "n_negative_flow": int(q.q_dm3_s.lt(0).sum()),
            "n_flow_at_source_cap": int(q.flow_at_reported_cap.sum()),
            "n_flow_above_method_cap_rounding_interval": int(q.q_dm3_s.gt(FLOW_CAP_DM3_S+.0005).sum()),
            "q_min_m3_s": float(q.q_m3_s.min()), "q_max_m3_s": float(q.q_m3_s.max()),
            "n_corrected_doc": len(d), "n_exact_flow_matches": int(j.flow_exact_match.sum()),
            "n_flow_matches_within_2min": int(j.flow_match.sum()),
            "n_doc_at_zero_flow": int(j.q_m3_s.eq(0).sum())})
    pd.DataFrame(yearly).to_csv(out / "year_inventory.csv", index=False)

    rivers = gpd.read_file("zip://" + str(RAW / "river_network.zip")).to_crs(2154)
    basins = gpd.read_file("zip://" + str(RAW / "catchment_boundary.zip")).to_crs(2154)
    if not rivers.is_valid.all() or not basins.is_valid.all():
        raise ValueError("Inspect invalid source geometry before analysis")
    network, basin = unary_union(list(rivers.geometry)), unary_union(list(basins.geometry))
    location = json.loads((RAW / "outlet_locations.json").read_text())["value"]
    if len(location) != 1 or location[0]["location"]["type"] != "Point":
        raise ValueError("Expected the one published Kervidy outlet point")
    lon, lat = location[0]["location"]["coordinates"]
    gauge = Point(*Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat))
    inside = network.intersection(basin)
    geom = {"crs": "EPSG:2154", "river_network": mapping(network), "catchment": mapping(basin),
        "clipped_river_network": mapping(inside), "gauge": mapping(gauge),
        "source_credit": "Source : UMR 1069 SAS INRA - Agrocampus Ouest"}
    (out / "mapped_geometry.json").write_text(json.dumps(geom, indent=2) + "\n")
    inventory = {"outlet_longitude": lon, "outlet_latitude": lat, "catchment_area_km2": basin.area/1e6,
        "river_source_features": len(rivers), "river_attributes_delivered": [c for c in rivers.columns if c != "geometry"],
        "full_archive_river_length_km": network.length/1000,
        "within_catchment_river_length_km": inside.length/1000,
        "gauge_inside_official_boundary": bool(basin.covers(gauge)),
        "gauge_distance_to_mapped_river_m": network.distance(gauge),
        "gauge_distance_to_basin_boundary_m": basin.boundary.distance(gauge),
        "direction_available_in_delivered_river_attributes": False,
        "form_class_assigned": False, "independent_form_replication": False}
    (out / "geometry_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    summary = {"n_flow_records": len(flow), "n_corrected_doc": len(doc),
        "n_exact_flow_matches": int(joined.flow_exact_match.sum()),
        "n_flow_matches_within_2min": int(joined.flow_match.sum()),
        "n_doc_without_flow_match": int((~joined.flow_match).sum()),
        "n_flow_selected_windows": len(windows), "n_dense_joint_windows": int(sum(r["dense_joint_window"] for r in rows)),
        "n_selected_censored_flow_maxima": int(windows.reported_flow_max_censored.sum()),
        "flow_source_cap_m3_s": FLOW_CAP_DM3_S*.001,
        "n_flow_above_method_cap_rounding_interval": int(flow.q_dm3_s.gt(FLOW_CAP_DM3_S+.0005).sum()),
        "n_flow_slightly_above_literal_cap": int(flow.q_dm3_s.gt(FLOW_CAP_DM3_S).sum()),
        "utc_matching": "exact and nearest within 2 min; source clocks retained, no interpolation or fitted shift",
        "window_selection": "reported maximum flow per observed calendar year; 3 days before and 4 after; no DOC-value selection",
        "event_interpretation": "windows are process illustrations, not automatically complete rise/recovery events",
        "new_model_training": False, "independent_form_effect_estimated": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    inputs = [CHEM, RAW / "discharge_quarter_hour.parquet", RAW / "river_network.zip", RAW / "catchment_boundary.zip",
              RAW / "outlet_locations.json", ROOT / "study_plan.md", ROOT / "acquisition_notes.md"]
    code = [Path(__file__), Path("src/river_graph/analysis/river_kervidy_observations.py")]
    (ROOT / "analysis_sources.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in inputs},
        "code_hashes": {str(p): sha256_file(p) for p in code},
        "outputs": {str(p): sha256_file(p) for p in sorted(out.iterdir())}}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
