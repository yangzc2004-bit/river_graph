"""Inspect the public hourly Turbolo optical-DOC case, separately from lab DOC."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
FILE = Path("data/raw/river_event_observations_v1/turbolo/REPO_DOC_final_2.xlsx")
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_tables(path):
    """Read the first data columns directly; no spreadsheet dependency or editing."""
    with ZipFile(path) as archive:
        shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(e.itertext()) for e in shared.findall("x:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        tables = {}
        for sheet in workbook.findall("x:sheets/x:sheet", NS):
            number, name = sheet.attrib["sheetId"], sheet.attrib["name"]
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
            rows = []
            for row in root.findall("x:sheetData/x:row", NS):
                cells = {}
                for cell in row.findall("x:c", NS):
                    column = "".join(x for x in cell.attrib["r"] if x.isalpha())
                    value = cell.find("x:v", NS)
                    if column not in "ABCDEF" or len(column) != 1 or value is None:
                        continue
                    cells[column] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                rows.append(cells)
            columns = [c for c in "ABCDEF" if c in rows[0]]
            tables[name] = pd.DataFrame([[row.get(c) for c in columns] for row in rows[1:]],
                                        columns=[rows[0][c] for c in columns])
    return tables


def build():
    tables = read_tables(FILE)
    summary_rows, hourly, events, duplicates = [], {}, [], []
    for name, raw in tables.items():
        frame = raw.copy()
        date_col = "Date" if "Date" in frame else "date"
        if "Hourly" in name:
            dates = pd.to_datetime(frame[date_col], errors="coerce")
        else:
            dates = pd.to_datetime(pd.to_numeric(frame[date_col], errors="coerce"), unit="D", origin="1899-12-30")
        frame["date_time_source_clock"] = dates
        for col in ("Turbidity (FNU)", "Tw (°C)", "Discharge (m3 s-1)", "DOC (mg l-1)"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        valid = frame.dropna(subset=["date_time_source_clock", "DOC (mg l-1)", "Discharge (m3 s-1)"])
        summary_rows.append({"sheet": name, "n_rows": len(frame), "n_valid_doc_q": len(valid),
                             "n_duplicate_times": int(valid.date_time_source_clock.duplicated().sum()),
                             "start": valid.date_time_source_clock.min(), "end": valid.date_time_source_clock.max(),
                             "n_missing_doc": int(frame["DOC (mg l-1)"].isna().sum()),
                             "n_turbidity_gt40": int(valid["Turbidity (FNU)"].gt(40).sum()),
                             "n_negative_doc_q": int(((valid["DOC (mg l-1)"] < 0) |
                                                       (valid["Discharge (m3 s-1)"] < 0)).sum())})
        if "Hourly" not in name:
            continue
        site = "upstream_SN" if name.endswith("_SN") else "receiver_FITT"
        clean = valid.copy()
        measured = ["Turbidity (FNU)", "Tw (°C)", "Discharge (m3 s-1)", "DOC (mg l-1)"]
        conflicting = []
        for timestamp, sub in valid[valid.date_time_source_clock.duplicated(keep=False)].groupby("date_time_source_clock"):
            conflict = sub[measured].nunique(dropna=False).gt(1).any()
            duplicates.append({"site": site, "timestamp": timestamp, "n_rows": len(sub),
                               "author_event_ids": ";".join(sub.event.astype(str)),
                               "conflicting_measurements": bool(conflict),
                               "action": "exclude_all" if conflict else "keep_one_identical_measurement"})
            if conflict:
                conflicting.append(timestamp)
        clean = clean[~clean.date_time_source_clock.isin(conflicting)].drop_duplicates("date_time_source_clock")
        hourly[site] = clean.set_index("date_time_source_clock")
        for event, sub in valid.groupby("event"):
            events.append({"site": site, "author_event_id": event, "n_hours": len(sub),
                           "start": sub.date_time_source_clock.min(), "end": sub.date_time_source_clock.max(),
                           "max_gap_hours": sub.sort_values("date_time_source_clock").date_time_source_clock.diff()
                           .dt.total_seconds().div(3600).max()})
    paired = hourly["upstream_SN"].join(hourly["receiver_FITT"], how="inner", lsuffix="_upstream", rsuffix="_receiver")
    paired = paired.reset_index().sort_values("date_time_source_clock").reset_index(drop=True)
    if paired.empty:
        raise ValueError("No paired hourly records")
    paired["block"] = paired.date_time_source_clock.diff().ne(pd.Timedelta(hours=1)).cumsum()
    paired["evidence_type"] = "corrected_fDOM_derived_DOC_not_laboratory_truth"
    blocks = paired.groupby("block").agg(start=("date_time_source_clock", "min"),
                                         end=("date_time_source_clock", "max"),
                                         n_paired_hours=("date_time_source_clock", "size")).reset_index()
    blocks["duration_hours"] = blocks.end.sub(blocks.start).dt.total_seconds().div(3600)
    summary = {
        "n_paired_hourly_records": len(paired), "n_contiguous_paired_blocks": len(blocks),
        "n_paired_blocks_ge24records": int(blocks.n_paired_hours.ge(24).sum()),
        "n_paired_blocks_ge24hours": int(blocks.duration_hours.ge(24).sum()),
        "largest_paired_block_records": int(blocks.n_paired_hours.max()),
        "largest_paired_block_duration_hours": float(blocks.duration_hours.max()),
        "clock_timezone": "Not stated in workbook/README; source clock retained, not relabelled UTC",
        "evidence_type": "Corrected optical DOC estimates; not independent laboratory concentration",
        "peak_reconstruction": "Paper reports statistical retrieval of some high-turbidity DOC peaks; workbook has no per-row flag",
        "selection": "Author-selected hourly events, not all events or a random sample",
        "river_relation": "Two nested sections described in the source paper; three-form morphology not assigned",
        "n_conflicting_site_timestamps_excluded": len([d for d in duplicates if d["conflicting_measurements"]]),
        "duplicate_policy": "Collapse identical overlapping-event measurements; exclude every conflicting site-time",
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(events), blocks, paired, pd.DataFrame(duplicates), summary


def main():
    sheets, events, blocks, paired, duplicates, summary = build()
    out = ROOT / "analysis"
    sheets.to_csv(out / "optical_sheet_audit.csv", index=False)
    events.to_csv(out / "optical_author_events.csv", index=False)
    blocks.to_csv(out / "optical_paired_blocks.csv", index=False)
    paired.to_parquet(out / "optical_paired_hours.parquet", index=False)
    duplicates.to_csv(out / "optical_duplicates.csv", index=False)
    (out / "optical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = [FILE, FILE.with_name("readme.txt"), ROOT / "alternative_retrieval.json",
             Path("scripts/audit_doc_river_optical_case_v1.py")]
    outputs = [out / name for name in ("optical_sheet_audit.csv", "optical_author_events.csv",
               "optical_paired_blocks.csv", "optical_paired_hours.parquet", "optical_duplicates.csv", "optical_summary.json")]
    receipt = {"source_hashes": {str(p): sha256_file(p) for p in paths},
               "output_hashes": {str(p): sha256_file(p) for p in outputs}}
    (ROOT / "optical_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
