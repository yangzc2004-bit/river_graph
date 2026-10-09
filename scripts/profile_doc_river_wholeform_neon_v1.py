"""Profile public lab-DOC replication candidates without fitting a form effect."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

RAW = Path("data/raw/river_wholeform_evidence_v1")
OUT = Path("experiments/phase4_transfer/doc_river_wholeform_evidence_v1")
LAKES = {"BARC", "CRAM", "LIRO", "PRLA", "PRPO", "SUGG", "TOOK"}


def main():
    paths = [RAW / "absNEON_output.csv", RAW / "absNEON_output_older_samples.csv"]
    records = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"sampleID", "domainID", "siteID", "collectDate", "DOC"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing sample fields: {path}")
        frame["archive_table"] = path.name
        records.append(frame)
    frame = pd.concat(records, ignore_index=True)
    if frame.sampleID.duplicated().any():
        raise ValueError("Overlapping sample IDs across archive tables; adjudicate before merging")
    frame["clock"] = pd.to_datetime(frame.collectDate, errors="raise")
    frame["sample_location"] = frame.sampleID.str.split(".").str[1]
    if not frame.sampleID.str.split(".").str[0].eq(frame.siteID).all():
        raise ValueError("Site ID differs from sample ID site")
    frame["sample_base_id"] = frame.sampleID.str.replace(r"\.\d+$", "", regex=True)
    frame["waterbody_scope"] = np.where(frame.siteID.isin(LAKES), "lake_site", "stream_or_river")
    frame["primary_surface_location"] = frame.sample_location.isin({"SS", "C0"})
    frame["doc_finite_nonnegative"] = np.isfinite(frame.DOC) & frame.DOC.ge(0)
    # Numeric suffixes are field replicates under the NEON water-chemistry guide.
    # They provide uncertainty at one occasion, not additional river/event units.
    groups = frame.groupby(["siteID", "sample_base_id"], sort=True)
    consistency = groups.agg(n_clocks=("collectDate", "nunique"),
                             n_locations=("sample_location", "nunique"))
    if consistency.n_clocks.gt(1).any() or consistency.n_locations.gt(1).any():
        raise ValueError("Replicate base ID spans clocks/locations; retain for source adjudication")
    occasions = groups.agg(
        domain=("domainID", "first"), collectDate=("collectDate", "first"),
        sample_location=("sample_location", "first"), waterbody_scope=("waterbody_scope", "first"),
        primary_surface_location=("primary_surface_location", "first"),
        n_replicates=("sampleID", "size"), doc_mean_mgl=("DOC", "mean"),
        doc_min_mgl=("DOC", "min"), doc_max_mgl=("DOC", "max"),
        all_finite_nonnegative=("doc_finite_nonnegative", "all")).reset_index()
    occasions["clock"] = pd.to_datetime(occasions.collectDate)
    occasions["month"] = occasions.clock.dt.strftime("%Y-%m")
    rows = []
    metadata_path = RAW / "neon_sites.json"
    site_metadata = {}
    if metadata_path.exists():
        site_metadata = {s["siteCode"]: s for s in json.loads(metadata_path.read_text())["data"]}
    for site, group in occasions.groupby("siteID", sort=True):
        primary = group[group.primary_surface_location & group.all_finite_nonnegative]
        clocks = primary.clock.sort_values().drop_duplicates()
        steps = clocks.diff().dt.total_seconds().dropna() / 86400
        raw = frame[frame.siteID.eq(site)]
        meta = site_metadata.get(site, {})
        rows.append({"site": site, "domain": group.domain.iloc[0],
                     "waterbody_scope": group.waterbody_scope.iloc[0],
                     "raw_doc_rows": len(raw), "sampling_occasions": len(group),
                     "primary_surface_occasions": len(primary),
                     "relocated_occasions_without_exact_coordinates": int(group.sample_location.eq("RE").sum()),
                     "months": primary.month.nunique(),
                     "first_clock": str(clocks.min()), "last_clock": str(clocks.max()),
                     "median_sampling_interval_days": steps.median(),
                     "doc_median_mgl": primary.doc_mean_mgl.median(),
                     "site_center_latitude": meta.get("siteLatitude"),
                     "site_center_longitude": meta.get("siteLongitude"),
                     "chemistry_outlet_location_verified": False,
                     "whole_network_geometry_verified": False,
                     "independence_from_other_networks_verified": False})
    output = OUT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "neon_site_inventory.csv", index=False)
    occasions.drop(columns="clock").to_csv(output / "neon_doc_occasions.csv", index=False)
    frame.groupby(["siteID", "sample_location"]).size().rename("raw_rows").reset_index().to_csv(
        output / "neon_sample_location_inventory.csv", index=False)
    lotic = inventory[inventory.waterbody_scope.eq("stream_or_river")]
    summary = {"archive_raw_doc_rows": len(frame), "archive_sites": frame.siteID.nunique(),
               "archive_unique_sample_ids": frame.sampleID.nunique(),
               "archive_missing_doc": int(frame.DOC.isna().sum()),
               "archive_negative_doc": int(frame.DOC.lt(0).sum()),
               "archive_zero_doc": int(frame.DOC.eq(0).sum()),
               "stream_river_sites": len(lotic), "stream_river_raw_rows": int(lotic.raw_doc_rows.sum()),
               "stream_river_sampling_occasions": int(lotic.sampling_occasions.sum()),
               "stream_river_primary_surface_occasions": int(lotic.primary_surface_occasions.sum()),
               "stream_river_relocated_occasions": int(lotic.relocated_occasions_without_exact_coordinates.sum()),
               "median_site_sampling_interval_days": lotic.median_sampling_interval_days.median(),
               "first_clock": str(frame.clock.min()), "last_clock": str(frame.clock.max()),
               "replicate_handling": "Mean within exact site/base-sample ID; clock and location consistency verified",
               "relocated_sample_handling": "Retained in inventory; excluded from primary-location summaries until alt coordinates are provided",
               "event_transport_eligible": False,
               "source_population": "DOC observations represented in a NEON-authored absorbance-derived archive; not all NEON DOC samples",
               "unavailable_fields": ["original quality flags", "sample condition", "detection flags/limits",
                                      "exact sampling coordinates", "relocated alt coordinates", "discharge",
                                      "whole-network geometry", "event branch inputs"],
               "effect_estimation": "None; chemistry coverage supports preparing independent field validation"}
    (output / "neon_profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths += [metadata_path] if metadata_path.exists() else []
    (OUT / "neon_profile_sources.json").write_text(json.dumps(
        {"inputs": {str(p): sha256_file(p) for p in paths}, "script": sha256_file(Path(__file__)),
         "method_guide": "NEON_waterChem_userGuide_vF, sections 3.6, 3.9, 3.10"}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
