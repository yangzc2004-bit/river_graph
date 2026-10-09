"""Recompute all selected measured-confluence DOC and geometry results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_measured_confluences import (
    campaign_summary,
    descriptive_associations,
    geometry_tables,
    mixing_ledger,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4_transfer/doc_river_measured_confluences_v1"
RAW = ROOT / "data/raw/river_measured_confluences_v1"


def main():
    inputs = [RAW / name for name in ["Plontetal_WRR_database.csv", "Plontetal_WRR_database_metadata.csv",
              "Plontetal_WRR_Fall2021_widthdepth_notrib.csv", "Plontetal_WRR_Fall2021_sumdata.csv"]]
    manifest = json.loads((OUT / "retrieval_manifest.json").read_text())
    for obj in manifest["objects"]:
        if sha256_file(ROOT / obj["path"]) != obj["sha256"]:
            raise ValueError(f"Changed archived input: {obj['path']}")
    meta = pd.read_csv(inputs[1]).set_index("Parameter")
    for field, unit in {"Q.Ls": "L/s", "DOC.mgL": "mg/L", "SpC": "uS/cm", "v.mmin": "m/min"}.items():
        if meta.loc[field, "Units"] != unit:
            raise ValueError(f"Unrecognized source unit: {field}")
    frame, geom = pd.read_csv(inputs[0]), pd.read_csv(inputs[2])
    if set(frame.season) != {"summer", "fall"} or frame.location.nunique() != 5:
        raise ValueError("Incomplete selected confluence/campaign inventory")
    ledger = mixing_ledger(frame)
    transects, geometry = geometry_tables(geom)
    summary = campaign_summary(ledger, geometry)
    association = descriptive_associations(summary)
    published = pd.read_csv(inputs[3]).rename(columns={"site": "location"})
    comparison = summary[summary.season == "fall"].merge(published, on="location", validate="one_to_one")
    comparison["width_cv_recalc_minus_published_pp"] = comparison.downstream_width_cv_pct - comparison["width.cv"]
    changed = ledger[(ledger.provided_main_fraction_error.abs() > 1e-7) |
                     (ledger.provided_trib_fraction_error.abs() > 1e-7)].copy()
    audit = {
        "downstream_position_rows": len(frame), "confluence_campaigns": len(summary),
        "confluences": frame.location.nunique(), "networks": 1,
        "geometry_raw_rows": len(geom), "geometry_width_transects": len(transects),
        "geometry_missing_depth_rows": int(geom["depth.m"].isna().sum()),
        "provided_fraction_disagreements": len(changed),
        "max_provided_fraction_sum_departure": float((ledger.provided_fraction_sum - 1).abs().max()),
        "tracer_out_of_bounds_rows": int((~ledger.tracer_in_bounds).sum()),
        "water_closure_pct_min": float(summary.water_closure_pct.min()),
        "water_closure_pct_max": float(summary.water_closure_pct.max()),
        "mixing_zone_chemistry": "Not in selected database; three fully mixed downstream positions only",
        "geometry_resolution": "Fall transects, equal width-transect weights; no complete rooted whole-network map",
        "units": "DOC mg/L * Q L/s = mg C/s; snapshots, not integrated event kg",
        "analysis": "All rows retained; fractions recalculated from discharge; no source R execution",
    }
    tables = {"point_ledger": ledger, "geometry_transects": transects,
              "geometry_reaches": geometry, "campaigns": summary,
              "descriptive_associations": association, "published_geometry_comparison": comparison,
              "provided_fraction_disagreements": changed}
    folder = OUT / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        data.to_csv(folder / f"{name}.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    outputs = [folder / f"{name}.csv" for name in tables] + [folder / "summary.json"]
    source_code = [Path(__file__), ROOT / "src/river_graph/analysis/river_measured_confluences.py"]
    (OUT / "analysis_sources.json").write_text(json.dumps({
        "inputs": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
        "code": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_code},
        "study_plan_sha256": sha256_file(OUT / "study_plan.md"),
        "outputs": {str(p.relative_to(OUT)): sha256_file(p) for p in outputs},
    }, indent=2) + "\n")
    if not np.isfinite(summary[["doc_departure_pct", "doc_flux_discrepancy_pct"]].to_numpy()).all():
        raise ValueError("Nonfinite core results")
    print(json.dumps(audit, indent=2))
    print(summary[["location", "season", "doc_departure_pct", "doc_flux_discrepancy_pct",
                   "spc_departure_pct", "wrt_downstream_main_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
