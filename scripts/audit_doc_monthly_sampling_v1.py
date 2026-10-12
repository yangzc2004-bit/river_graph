"""Describe sampling support of frozen DOC labels without changing them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")


def main():
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    months = pd.DatetimeIndex(data["months"])
    lookup = {month: i for i, month in enumerate(months)}
    rows = []
    inputs = {}
    for station, site in enumerate(data["site_no"]):
        path = Path("data/raw/wqp_results") / f"{site}.csv"
        inputs[str(path)] = sha256_file(path)
        doc = extract_doc_obs(load_station_results(path))
        doc["month"] = doc.date.values.astype("datetime64[M]")
        for month, part in doc.groupby("month"):
            index = lookup.get(month)
            if index is None or not data["y_mask"][station, index]:
                continue
            distinct = part[["date", "doc"]].drop_duplicates()
            days = pd.to_datetime(part.date).dt.normalize()
            rows.append({"station": station, "site_no": str(site), "month": index,
                         "calendar_month": str(month.date()),
                         "sample_records": len(part), "unique_date_value_records": len(distinct),
                         "sample_days": days.nunique(),
                         "earliest_sample_day": int(days.dt.day.min()),
                         "latest_sample_day": int(days.dt.day.max()),
                         "sample_day_span": int((days.max()-days.min()).days),
                         "days_in_month": month.days_in_month,
                         "y_true": float(data["y"][station, index])})
    frame = pd.DataFrame(rows)
    assert len(frame) == int(np.asarray(data["y_mask"]).sum())
    frame["cell"] = frame.station * len(months) + frame.month
    groups = [("all", frame)]
    for mask in sorted((ROOT / "masks").glob("*.npz")):
        with np.load(mask, allow_pickle=False) as archive:
            groups.append((mask.stem, frame[frame.cell.isin(archive["test"])]))
    summaries = []
    for name, part in groups:
        summaries.append({"panel": name, "observed_months": len(part),
                          "single_record_months": int((part.sample_records == 1).sum()),
                          "single_sample_day_months": int((part.sample_days == 1).sum()),
                          "single_sample_day_percent": 100 * float((part.sample_days == 1).mean()),
                          "median_records_per_month": float(part.sample_records.median()),
                          "median_days_per_month": float(part.sample_days.median()),
                          "median_day_span": float(part.sample_day_span.median()),
                          "date_value_duplicate_records": int((part.sample_records-part.unique_date_value_records).sum())})
    out = ROOT / "sampling_audit"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "observed_month_sampling.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    report = {"dataset_sha256": sha256_file(DATASET), "raw_sha256": inputs,
              "script_sha256": sha256_file(__file__), "labels_modified": False,
              "definition": "Sample records and distinct sample dates of accepted DOC records in each frozen observed calendar-month cell. Distinct date/value records do not identify physical replicate samples.",
              "interpretation": "The target is the mean of available samples. Sampling support alone cannot establish measurement error or a cause of low temporal skill.",
              "all_observations": summaries[0],
              "outputs": {name: sha256_file(out / name) for name in ("observed_month_sampling.csv", "summary.csv")}}
    (out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries[0]))


if __name__ == "__main__":
    main()
