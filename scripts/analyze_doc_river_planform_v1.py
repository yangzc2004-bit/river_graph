"""Classify complete real river shapes without opening DOC labels."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from river_graph.analysis.river_planform_typology import classify
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_planform_typology_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    progress = json.loads((ROOT / "progress.json").read_text())
    records = json.loads((ROOT / "analysis" / "acquisition_records.json").read_text())
    acquisition_finished = len(records) == progress["n_stations"]
    if not acquisition_finished and not args.allow_partial:
        raise SystemExit("Acquisition incomplete; use --allow-partial only for a labelled provisional analysis")
    source = ROOT / "analysis" / "station_planform.csv"
    raw = source.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), dtype={"station": str, "huc_cd": str})
    # Same receiving reach is one geometric unit even if monitored twice.
    unique = frame.sort_values("station").drop_duplicates("comid").reset_index(drop=True)
    result = classify(unique, draws=args.draws)
    out = ROOT / "analysis" if acquisition_finished else ROOT / "provisional_analysis" / f"cohort_n{len(frame)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "measured_cohort.csv").write_bytes(raw)
    for name in ("classes", "features", "candidates", "stability", "centroids", "representatives", "candidate_assignments"):
        result[name].to_csv(out / f"{name}.csv", index=False)
    full = frame.merge(result["classes"][["comid", "cluster", "classification_status"]], on="comid", validate="many_to_one")
    full.to_csv(out / "station_classes.csv", index=False)
    status = ("complete" if progress["full_cohort_complete"] else "complete_for_available_geometry") if acquisition_finished else "provisional_partial_cohort"
    summary = {"status": status,
               "attempted_stations": len(records), "requested_stations": progress["n_stations"],
               "acquisition_failures": {s: r["status"] for s, r in records.items() if r["status"] != "complete"},
               "measured_stations": len(frame), "unique_networks": len(unique),
               "classified_networks": len(result["features"]), "selected_k": result["selected_k"],
               "features": result["selected_features"], "removed": result["removed"],
               "counts": {str(k): int(v) for k, v in result["classes"].cluster.value_counts().items()},
               "source_sha256": sha256_file(out / "measured_cohort.csv"), "source_snapshot": "measured_cohort.csv",
               "study_plan_sha256": sha256_file(ROOT / "study_plan.md"),
               "geometry_unit": "full upstream network at receiving COMID outlet",
               "method": "Ward; maximum silhouette with minimum class n=10; equal feature-block weight",
               "stability_draws": args.draws, "stability_seed": 42,
               "doc_used_for_classification": False,
               "stability_note": "80% unique-network subsets; nested basins are dependent, not independent replicates"}
    (out / "classification_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    rows = result["classes"].dropna(subset=["cluster"]).groupby("cluster")
    description = rows[["basin_aspect", "network_axis_ratio", "mainstem_share", "drainage_density",
                        "junction_frequency", "side_imbalance", "tributary_alignment", "mainstem_sinuosity",
                        "basin_area_km2", "n_reaches", "hierarchy_order"]].median()
    description.to_csv(out / "class_physical_medians.csv")
    # Report scale associations; excluding area as a clustering feature does
    # not remove size dependence from hierarchy or mainstem share.
    scale = unique.loc[result["features"].index, "basin_area_km2"]
    diagnostics = [{"feature": name, "spearman_with_basin_area": column.corr(scale, method="spearman")}
                   for name, column in result["features"].items()]
    pd.DataFrame(diagnostics).to_csv(out / "scale_diagnostics.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
