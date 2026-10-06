"""Verify complete real shapes and classification-to-map source correspondence."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_planform_typology_v1")
CACHE = Path("data/raw/river_planform_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()
    analysis = args.analysis_dir or ROOT / ("provisional_analysis" if args.provisional else "analysis")
    summary = json.loads((analysis / "classification_summary.json").read_text())
    source = analysis / summary["source_snapshot"] if "source_snapshot" in summary else ROOT / "analysis" / "station_planform.csv"
    assert summary["source_sha256"] == sha256_file(source), "Measured cohort changed: rerun classification"
    assert summary["study_plan_sha256"] == sha256_file(ROOT / "study_plan.md")
    records = json.loads((ROOT / "analysis" / "acquisition_records.json").read_text())
    classes = pd.read_csv(analysis / "classes.csv", dtype={"station": str})
    features = pd.read_csv(analysis / "features.csv")
    assert np.isfinite(features.to_numpy()).all()
    assert len(features) == summary["classified_networks"]
    assert not classes.comid.duplicated().any()
    assert classes.loc[classes.cluster.notna(), "n_reaches"].ge(5).all()
    connection = sqlite3.connect(f"file:{CACHE / 'flowlines.sqlite'}?mode=ro", uri=True)
    available = {v for (v,) in connection.execute("SELECT comid FROM lines")}
    catchment_area = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "areasqkm"]).set_index("comid").areasqkm
    sources, area_checks = [], []
    for row in classes.itertuples():
        record = records[row.station]
        assert record["status"] == "complete" and record["geometry_coverage"] == 1
        member_path = CACHE / "members_full" / f"comid_{row.comid}.npz"
        member = np.load(member_path)
        assert len(member["comids"]) == row.n_reaches
        assert set(map(int, member["comids"])).issubset(available)
        assert set(map(int, member["mainstem"])).issubset(set(map(int, member["comids"])))
        assert int(member["mainstem"][0]) == row.comid
        assert row.mainstem_gap_max_m < 1., "Mainstem stitching gap requires inspection"
        unique_area = float(catchment_area.loc[member["comids"]].sum())
        ratio = row.basin_area_km2/unique_area
        assert .99 <= ratio <= 1.01, "Basin polygon differs materially from unique contributing catchments"
        area_checks.append({"station": row.station, "comid": row.comid,
            "polygon_area_km2": row.basin_area_km2, "unique_catchment_area_km2": unique_area,
            "polygon_to_unique_area": ratio,
            "polygon_to_cumulative_vaa_area": row.basin_area_km2/row.vaa_area_km2})
        basin_path = CACHE / "basins" / f"comid_{row.comid}.json"
        assert basin_path.exists()
        sources.append({"station": row.station, "comid": row.comid, "members_sha256": sha256_file(member_path),
                        "basin_sha256": sha256_file(basin_path), "flowline_url": record["flowline_url"],
                        "basin_url": record["basin_url"]})
    connection.close()
    pd.DataFrame(area_checks).to_csv(analysis / "basin_area_validation.csv", index=False)
    reps = pd.read_csv(analysis / "representatives.csv", dtype={"station": str})
    assert reps.station.isin(classes.station).all()
    assert reps.groupby("cluster").size().eq(3).all()
    result = {"verified": True, "measured_unique_networks": len(classes), "classified_networks": len(features),
              "source_nodes_sha256": sha256_file("data/processed/graph_nodes_graphfix_st357.csv"),
              "source_vaa_sha256": sha256_file("cache/nldplus_vaa.parquet"),
              "classification_status": summary["status"], "source_geometry": sources}
    result["maximum_relative_area_difference"] = max(abs(r["polygon_to_unique_area"]-1) for r in area_checks)
    (analysis / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Verified {len(classes)} complete geometric units and {len(features)} classified real networks")


if __name__ == "__main__":
    main()
