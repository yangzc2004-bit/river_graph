"""Separate conservative timing from labelled DOC using all paired additions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_tracer_process import (
    atom_fraction,
    bounded_response,
    clock_seconds,
    paired_joint_area,
    pulse_core_metrics,
    resolve_clock_points,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_tracer_process_v1")
RAW = Path("data/raw/river_tracer_process_v1")
PULSES = {
    "g1": ("Glucose_20190808", "2019-08-08 10:11:00"),
    "l": ("Leachate_20190809", "2019-08-09 10:15:00"),
    "g2": ("Glucose_20190815", "2019-08-15 09:00:00"),
}
BACKGROUNDS = {
    "b_g1_up": (25, 25), "b_g1_down": (75, 75),
    "b_l_up": (30, 30), "b_l_down": (75, 75),
    "b_g2_up": (35, 35), "b_g2_down": (75, 50),
}


def source_points():
    all_rows, inventory = [], []
    info = pd.read_csv(RAW / "blaine/site_data.csv")
    for pulse, (suffix, start) in PULSES.items():
        source = RAW / "blaine" / f"13CAdditionData_Creston{suffix}_analyzed.csv"
        frame = pd.read_csv(source)
        date_col = "SampleDate" if "SampleDate" in frame else "Date"
        dates = pd.to_datetime(frame[date_col] + " " + frame.SampleTime, format="mixed")
        salt_col = "SpC.us.cm" if "SpC.us.cm" in frame else "SpC.us.cm_1"
        for site in ("Up", "Down"):
            select = frame.SiteID.eq(site)
            original = frame[select].copy()
            code = f"b_{pulse}_{site.lower()}"
            salt_stop, isotope_stop = BACKGROUNDS[code]
            time = (dates[select] - pd.Timestamp(start)).dt.total_seconds() / 60
            original["time_min"] = time
            bkg_salt = original.loc[time.between(0, salt_stop), salt_col].dropna()
            bkg_delta = original.loc[time.between(0, isotope_stop), "Delta 13C"].dropna()
            if bkg_salt.empty or bkg_delta.empty:
                raise ValueError(f"No measured background at {code}")
            af = atom_fraction(original["Delta 13C"])
            bkg_af = atom_fraction(bkg_delta)
            series = info[info.code.eq(code)]
            if len(series) != 1:
                raise ValueError("Ambiguous tracer site metadata")
            metadata = series.iloc[0]
            ratio = metadata.doc_mass_13C_mg / (metadata.salt_mass * 2100)
            lab = original["DOC.ID"].notna() & original["DOC.ID"].ne("NotEntered")
            if pulse != "g2":
                lab[:] = True
            lab |= original.DOC_conc.notna() | original["Delta 13C"].notna()
            points = pd.DataFrame({
                "pulse": pulse, "code": code, "site": site.lower(), "source_file": str(source),
                "record_id": original.index + 1, "timestamp_local": dates[select].to_numpy(),
                "release_timestamp_local": pd.Timestamp(start), "time_min": time.to_numpy(),
                "spc_raw": original[salt_col].to_numpy(),
                "doc_total_raw": original.DOC_conc.to_numpy(),
                "delta13c_raw": original["Delta 13C"].to_numpy(),
                "atom_fraction_raw": af, "is_lab_record": lab.to_numpy(),
                "source_notes": original.NOTES.to_numpy(), "reference_per_spc": ratio,
                "background_spc": bkg_salt.mean(), "background_atom_fraction": bkg_af.mean(),
                "background_spc_median": bkg_salt.median(), "background_atom_fraction_median": np.median(bkg_af),
            })
            points["doc_label_raw"] = points.doc_total_raw * (af - bkg_af.mean())
            points["doc_label_median_bkg"] = points.doc_total_raw * (af - np.median(bkg_af))
            all_rows.append(points)
            inventory.append({"code": code, "pulse": pulse, "site": site.lower(),
                              "substrate": metadata.doc_release, "date": pd.Timestamp(start).date(),
                              "release_clock": start, "distance_from_release_m": metadata.length,
                              "n_original_rows": len(points), "n_unique_times": points.time_min.nunique(),
                              "n_lab_scheduled_rows": int(lab.sum()),
                              "n_lab_coobserved_rows": int(points.doc_label_raw.notna().sum()),
                              "n_conductivity_rows": int(points.spc_raw.notna().sum()),
                              "n_background_salt": len(bkg_salt), "n_background_isotope": len(bkg_delta),
                              "background_salt_stop_min": salt_stop, "background_isotope_stop_min": isotope_stop,
                              "background_spc": bkg_salt.mean(), "background_spc_median": bkg_salt.median(),
                              "background_atom_fraction": bkg_af.mean(), "reference_per_spc": ratio,
                              "source_manual_travel_min": metadata.travel_time,
                              "source_summary_nominal_travel_min": metadata["NomTT.min"],
                              "source_summary_peak_min": metadata["PeakT.min"],
                              "source_flow_ls": metadata["Qest.Ls"], "source_notes": metadata.Notes})
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(inventory)


def build():
    raw, inventory = source_points()
    author = pd.read_csv(RAW / "author/data_doc.csv")
    frames, metrics, sensitivity, comparison = [], [], [], []
    for code, original in raw.groupby("code", sort=True):
        metadata = inventory[inventory.code.eq(code)].iloc[0]
        points = resolve_clock_points(original)
        for col in ("code", "pulse", "site"):
            points[col] = metadata[col]
        points["reference_raw"] = (points.spc_raw - metadata.background_spc) * metadata.reference_per_spc
        for bkg in ("mean", "median"):
            work = original.copy()
            if bkg == "median":
                work["doc_label_raw"] = work.doc_label_median_bkg
            resolved = resolve_clock_points(work)
            baseline = metadata.background_spc if bkg == "mean" else metadata.background_spc_median
            ref = (resolved.spc_raw - baseline) * metadata.reference_per_spc
            for cutoff in (0.10, 0.25, 0.50):
                sensitivity.append({"code": code, "policy": "original_points", "background": bkg,
                                    "core_fraction": cutoff,
                                    **pulse_core_metrics(resolved.time_min, ref, resolved.doc_label_raw, cutoff)})
        for gap in (20, 30, 45):
            salt_shape = bounded_response(points.time_min, points.reference_raw, max_gap=gap)
            lab = points[points.n_lab_records.gt(0)]
            joint = paired_joint_area(lab.time_min, lab.reference_raw, lab.doc_label_raw, max_gap=gap)
            lab_shape = bounded_response(lab.time_min, lab.doc_label_raw, max_gap=gap)
            row = {**metadata.to_dict(), "max_gap_min": gap,
                   "n_conflicting_salt_times": int(points.salt_conflict.sum()),
                   **pulse_core_metrics(points.time_min, points.reference_raw, points.doc_label_raw),
                   **{f"salt_{k}": v for k, v in salt_shape.items()},
                   **{f"doc_{k}": v for k, v in lab_shape.items()}, **joint}
            metrics.append(row)
        processed = author[author.code.eq(code)].copy()
        n_author_rows = len(processed)
        # The two real upstream leachate baseline replicates share a clock in
        # both archives. Their derived values are averaged at that clock only.
        processed["clock_second"] = clock_seconds(processed.MinFrom0)
        processed = processed.groupby("clock_second", as_index=False)[["MinFrom0", "SpC_corr", "doc_13C"]].mean()
        processed = processed.rename(columns={"MinFrom0": "author_time_min", "SpC_corr": "author_spc_anomaly",
                                              "doc_13C": "author_doc_label"})
        points["clock_second"] = clock_seconds(points.time_min)
        points = points.merge(processed, on="clock_second", how="outer", validate="one_to_one", indicator=True)
        points["time_min"] = points.time_min.fillna(points.author_time_min)
        points["code"], points["pulse"], points["site"] = code, metadata.pulse, metadata.site
        points["author_reference"] = points.author_spc_anomaly * metadata.reference_per_spc
        points["author_carbon_without_raw_lab"] = points.author_doc_label.notna() & points.doc_label_raw.isna()
        points["author_only_clock"] = points._merge.eq("right_only")
        paired_carbon = points.author_doc_label.notna() & points.doc_label_raw.notna()
        paired_salt = points.author_reference.notna() & points.reference_raw.notna()
        points["author_doc_differs"] = paired_carbon & ~np.isclose(points.author_doc_label, points.doc_label_raw, atol=1e-12, rtol=0)
        points["author_salt_differs"] = paired_salt & ~np.isclose(points.author_reference, points.reference_raw, atol=1e-12, rtol=0)
        comparison.append({"code": code, "n_author_rows": n_author_rows,
                           "n_author_unique_clocks": len(processed),
                           "n_author_carbon_without_raw_lab": int(points.author_carbon_without_raw_lab.sum()),
                           "n_author_only_clocks": int(points.author_only_clock.sum()),
                           "n_lab_points_omitted_in_author": int((points.doc_label_raw.notna() & points.author_doc_label.isna()).sum()),
                           "n_author_salt_differing": int(points.author_salt_differs.sum()),
                           "n_author_carbon_differing": int(points.author_doc_differs.sum())})
        for cutoff in (0.10, 0.25, 0.50):
            sensitivity.append({"code": code, "policy": "author_processed", "background": "author",
                                "core_fraction": cutoff,
                                **pulse_core_metrics(processed.author_time_min,
                                                     processed.author_spc_anomaly * metadata.reference_per_spc,
                                                     processed.author_doc_label, cutoff)})
        frames.append(points.drop(columns="_merge").sort_values("time_min"))
    metrics = pd.DataFrame(metrics)
    primary = metrics[metrics.max_gap_min.eq(30)]
    pairs = []
    for pulse, group in primary.groupby("pulse", sort=True):
        a, b = (group[group.site.eq(site)].iloc[0] for site in ("up", "down"))
        pairs.append({"pulse": pulse, "date": a.date, "substrate": a.substrate,
                      "segment_length_m": b.distance_from_release_m - a.distance_from_release_m,
                      "core_fraction_up": a.core_slope, "core_fraction_down": b.core_slope,
                      "down_up_core_fraction_ratio": b.core_slope / a.core_slope,
                      "salt_centroid_delay_min": b.salt_centroid_min - a.salt_centroid_min,
                      "salt_duration80_ratio": b.salt_duration80_min / a.salt_duration80_min,
                      "salt_peak_ratio": b.salt_sampled_peak / a.salt_sampled_peak,
                      "doc_peak_ratio": b.doc_sampled_peak / a.doc_sampled_peak,
                      "sampled_salt_peak_first_delay_min": b.salt_sampled_peak_first_min - a.salt_sampled_peak_first_min,
                      "joint_fraction_up": a.joint_doc_reference_fraction,
                      "joint_fraction_down": b.joint_doc_reference_fraction,
                      "down_up_joint_fraction_ratio": b.joint_doc_reference_fraction / a.joint_doc_reference_fraction,
                      "salt_boundary_peak_ratio_up": a.salt_boundary_peak_ratio,
                      "salt_boundary_peak_ratio_down": b.salt_boundary_peak_ratio,
                      "salt_covered_fraction_up": a.salt_covered_min / a.salt_span_min,
                      "salt_covered_fraction_down": b.salt_covered_min / b.salt_span_min})
    summary = {"n_original_pair_rows": len(raw), "n_series": len(inventory),
               "n_additions": len(PULSES), "n_independent_streams": 1,
               "n_original_lab_coobserved_rows": int(raw.doc_label_raw.notna().sum()),
               "n_original_lab_scheduled_rows": int(raw.is_lab_record.sum()),
               "n_conflicting_salt_times": int(primary.n_conflicting_salt_times.sum()),
               "n_series_core_below_conservative_reference": int(primary.core_slope.lt(1).sum()),
               "core_fraction_min": float(primary.core_slope.min()),
               "core_fraction_max": float(primary.core_slope.max()),
               "scope": "three paired additions in one short stream segment; nominal catalogue distance retained separately; original-point primary with identified author processing",
               "not_estimated": "whole-network class differences; complete mass recovery; DOC uptake/respiration rates; fitted geometry-to-storage coefficients"}
    return {"series_inventory": inventory, "response_metrics": metrics,
            "paired_responses": pd.DataFrame(pairs), "processing_sensitivity": pd.DataFrame(sensitivity),
            "author_processing_comparison": pd.DataFrame(comparison)}, {
                "original_sample_points": raw, "response_points": pd.concat(frames, ignore_index=True)}, summary


def main():
    tables, products, summary = build()
    output = ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in products.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest = json.loads((ROOT / "retrieval_manifest.json").read_text())
    sources = [Path(item["path"]) for item in manifest["objects"]]
    sources += [ROOT / "retrieval_manifest.json", ROOT / "study_plan.md", ROOT / "source_method_notes.md",
                Path("scripts/fetch_doc_river_tracer_process_v1.py"),
                Path("scripts/analyze_doc_river_tracer_process_v1.py"),
                Path("src/river_graph/analysis/river_tracer_process.py")]
    for path in sources:
        if str(path).startswith(("scripts/", "src/")):
            destination = ROOT / "code_snapshot" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    receipts = {"sources": [{"path": str(path), "sha256": sha256_file(path)} for path in sources],
                "products": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(output.iterdir())]}
    (ROOT / "analysis_sources.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
