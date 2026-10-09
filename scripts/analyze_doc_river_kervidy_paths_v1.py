"""Measure connected BD Topage paths in the monitored Kervidy basin."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, mapping, shape

from river_graph.analysis.river_rooted_paths import rooted_paths
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_kervidy_pulses_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_kervidy_geometry_v1")
RAW = Path("data/raw/river_kervidy_geometry_v1/topage_network.zip")
WFS = "https://geosas.fr/geoserver/wfs?service=WFS&request=GetFeature&version=2.0.0&typeName=ore%3Anaizin_topage&outputFormat=SHAPE-ZIP"
RULES = "https://www.sandre.eaufrance.fr/ftp/documents/fr/DocAdmin/ETH/1/sandre_administration_topage_1.pdf"


def main():
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    known = json.loads((PREVIOUS / "analysis/mapped_geometry.json").read_text())
    basin, gauge = shape(known["catchment"]), shape(known["gauge"])
    source = gpd.read_file("zip://"+str(RAW)).to_crs(2154)
    reaches, geometries, raw_directions = [], {}, {}
    for row in source.itertuples():
        line = row.geometry.intersection(basin)
        if line.is_empty or line.length == 0:
            continue
        if line.geom_type != "LineString":
            raise ValueError("A clipped reach needs manual geometric inspection")
        reaches.append((row.CdOH, line))
        geometries[row.CdOH] = mapping(line)
        raw_directions[row.CdOH] = str(row.SensEcoule)
    endpoints = {tuple(line.coords[i]) for _, line in reaches for i in (0, -1)}
    root = min(endpoints, key=lambda x: Point(x).distance(gauge))
    root_distance = Point(root).distance(gauge)
    if root_distance > 10:
        raise ValueError("Mapped terminal point is too far from the published outlet")
    edges, paths, inventory = rooted_paths(reaches, root)
    inventory.update({"crs": "EPSG:2154", "full_layer_n_reaches": len(source),
        "catchment_area_km2": basin.area/1e6, "mapped_terminal_to_gauge_m": root_distance,
        "root_xy": root, "raw_direction_values": sorted(set(raw_directions.values())),
        "direction_basis": "Unique paths towards the known outlet; all source coordinate directions agree. Published BD Topage production convention also digitizes in flow direction.",
        "numeric_direction_code_decoded": False,
        "path_scope": "Mapped river headwaters to mapped outlet terminal, not hillslope paths or measured travel times",
        "morphology_class_assigned": False, "independent_catchments": 1})
    pd.DataFrame([{**p, "edge_ids": ";".join(p["edge_ids"])} for p in paths]).to_csv(out / "mapped_outlet_paths.csv", index=False)
    pd.DataFrame([{**e, "raw_SensEcoule": raw_directions[e["edge_id"]]} for e in edges]).to_csv(out / "topage_reaches.csv", index=False)
    (out / "topage_path_inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    (out / "topage_path_geometry.json").write_text(json.dumps({"crs": "EPSG:2154", "reaches": geometries,
        "catchment": known["catchment"], "gauge": known["gauge"], "mapped_terminal": mapping(Point(root)),
        "paths": paths, "common_terminal_edge_ids": sorted(set.intersection(*(set(p["edge_ids"]) for p in paths)))}, indent=2)+"\n")
    (ROOT / "topage_sources.json").write_text(json.dumps({"wfs_url": WFS,
        "geometry_source_sha256": sha256_file(RAW), "production_convention_url": RULES,
        "convention_section": "Annex 2, page 40: digitization follows flow, apart from bilateral flow exceptions",
        "numeric_nomenclature_access": "The official 776 endpoint could not be retrieved; numeral 2 is preserved, not decoded.",
        "script_sha256": sha256_file(Path(__file__)),
        "path_module_sha256": sha256_file(Path("src/river_graph/analysis/river_rooted_paths.py")),
        "previous_geometry_sha256": sha256_file(PREVIOUS / "analysis/mapped_geometry.json")}, indent=2)+"\n")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
