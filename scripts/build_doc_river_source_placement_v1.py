"""Locate potential DOC source landscapes along actual whole-river paths."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_source_placement import (
    UpstreamDistances,
    source_placement,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
CACHE = Path("data/raw/river_source_placement_v1")
MEMBERS = Path("data/raw/river_planform_v1/members_full")


def main():
    nodes_path = Path("data/processed/graph_nodes_graphfix_st357.csv")
    vaa_path = Path("cache/nldplus_vaa.parquet")
    landscape_path = CACHE/"reach_landscape.parquet"
    audit_path = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/station_mapping_audit.csv")
    nodes = pd.read_csv(nodes_path, dtype={"site_no": str, "huc_cd": str})
    audit = pd.read_csv(audit_path, dtype={"site_no": str}).set_index("site_no")
    landscape = pd.read_parquet(landscape_path).set_index("comid")
    memberships = {}
    hashes = {str(p): sha256_file(p) for p in (nodes_path, vaa_path, landscape_path, audit_path,
             Path(__file__), Path("src/river_graph/analysis/river_source_placement.py"), ROOT/"study_plan.md")}
    for cid in nodes.comid.unique():
        p = MEMBERS/f"comid_{int(cid)}.npz"
        hashes[str(p)] = sha256_file(p)
        with np.load(p) as z:
            memberships[int(cid)] = np.unique(z["comids"])
    union = np.unique(np.concatenate(list(memberships.values())))
    vaa = pd.read_parquet(vaa_path, columns=["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm"])
    vaa = vaa[vaa.comid.isin(union)].reset_index(drop=True)
    network = UpstreamDistances(vaa)
    area_lookup = vaa.set_index("comid").areasqkm
    (CACHE/"routing").mkdir(exist_ok=True)
    summaries = {}
    rows = []
    for n, row in enumerate(nodes.itertuples(), 1):
        cid = int(row.comid)
        members = memberships[cid]
        if cid not in summaries:
            path = CACHE/"routing"/f"comid_{cid}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["comids"], members):
                        raise ValueError("routing membership cache changed")
                    distance = z["distance_km"]
            else:
                distance = network.distances(cid, members)
                np.savez_compressed(path, comids=members, distance_km=distance)
            area = area_lookup.reindex(members).to_numpy(float)
            f = landscape.reindex(members)
            complete = f.nlcd2019_catpctfull.to_numpy(float) >= 95
            # All component columns must be known. pandas sum would turn all
            # missing forest/wetland classes into fictitious source-free land.
            wetland = f.pctwdwet2019cat.to_numpy(float)+f.pcthbwet2019cat.to_numpy(float)
            forest = f.pctdecid2019cat.to_numpy(float)+f.pctconif2019cat.to_numpy(float)+f.pctmxfst2019cat.to_numpy(float)
            wetland[~complete], forest[~complete] = np.nan, np.nan
            rp_wetland = f.pctwdwet2019catrp100.to_numpy(float)+f.pcthbwet2019catrp100.to_numpy(float)
            rp_forest = f.pctdecid2019catrp100.to_numpy(float)+f.pctconif2019catrp100.to_numpy(float)+f.pctmxfst2019catrp100.to_numpy(float)
            summary = source_placement(area, distance, wetland, forest,
                riparian_area=f.catareasqkmrp100.to_numpy(float), riparian_wetland=rp_wetland,
                riparian_forest=rp_forest)
            summary.update(comid=cid, n_upstream_reaches=len(members),
                reachable_reaches=int(np.isfinite(distance).sum()),
                catchment_area_ratio=float(f.catareasqkm.sum()/area.sum()) if area.sum() > 0 else np.nan,
                area_unit="km2", distance_unit="channel km; reach midpoint to receiving outlet")
            summaries[cid] = summary
        summary = summaries[cid].copy()
        summary.update(station=row.site_no, huc_cd=row.huc_cd, mapping_status=audit.loc[row.site_no, "mapping_status"])
        if summary["mapping_status"] == "gross_area_mismatch":
            status = "mapping_discrepancy_excluded"
        elif summary["represented_area_fraction"] < .95 or not np.isfinite(summary["represented_area_fraction"]):
            status = "landscape_or_routing_coverage_insufficient"
        else:
            status = "included_landscape"
        summary["inclusion_status"] = status
        rows.append(summary)
        if n % 25 == 0:
            print(f"Source placement: {n}/{len(nodes)} stations", flush=True)
    out = ROOT/"analysis"
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"station_source_placement.csv", index=False)
    (ROOT/"placement_sources.json").write_text(json.dumps({"source_hashes": hashes,
        "distance_definition": "shortest upstream primary-or-secondary channel route from reach midpoint to receiving reach outlet",
        "NLCD_year": 2019, "nodes": len(nodes), "unique_reach_basins": len(summaries),
        "inclusion": frame.inclusion_status.value_counts().to_dict(), "reads_DOC_labels": False,
        "historical_graph_modified": False}, indent=2)+"\n")
    print(frame.inclusion_status.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
