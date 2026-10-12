"""Audit daily DOC sources against the frozen monthly comparison cohort."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from prepare_aqua_oc_data_v1 import CACHE, OUTPUT, ROOT, digest, write_json


def main() -> None:
    protocol_path = ROOT / "experiments/phase4_transfer/doc_domain_work_v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    data_path = ROOT / protocol["dataset"]
    if digest(data_path) != protocol["dataset_sha256"]:
        raise ValueError("Frozen comparison dataset differs")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    local = pd.read_parquet(CACHE / "local_doc_samples.parquet")
    local["month"] = local.date.dt.to_period("M").astype(str)
    monthly = local.groupby(["site_no", "month"], as_index=False).doc.mean()
    monthly_values = monthly.set_index(["site_no", "month"]).doc
    sites = np.asarray(data["site_no"], str)
    months = pd.to_datetime(np.asarray(data["months"])).to_period("M").astype(str)
    truth = np.asarray(data["y"])
    observed = np.asarray(data["y_mask"], bool)
    cells = np.flatnonzero(observed.ravel())
    keys = pd.MultiIndex.from_arrays([sites[cells // len(months)], months[cells % len(months)]],
                                     names=["site_no", "month"])
    values = monthly_values.reindex(keys).to_numpy()
    errors = np.abs(values - truth.ravel()[cells])
    receipt = {"stage": "data_acquisition_and_quality_audit_only",
               "training_started": False, "complete_aqua_oc_reproduction": False,
               "dataset_sha256": digest(data_path),
               "local_sample_sha256": digest(CACHE / "local_doc_samples.parquet"),
               "canonical_observed_monthly_cells": len(cells),
               "monthly_cells_missing_in_current_raw": int(np.isnan(values).sum()),
               "max_monthly_absolute_difference_mg_l": float(np.nanmax(errors)),
               "monthly_cells_difference_over_1e_minus_4": int(np.sum(errors > 1e-4))}
    frame = pd.read_csv(CACHE / "raw/grqa/DOC_GRQA.csv", sep=";", dtype={"site_id": str}, low_memory=False,
                        usecols=["obs_id", "site_id", "lat_wgs84", "lon_wgs84", "obs_date",
                                 "obs_value", "unit", "source", "detection_limit_flag", "obs_iqr_outlier"])
    frame["date"] = pd.to_datetime(frame.obs_date, errors="coerce")
    frame["canonical_site"] = frame.site_id.str.removeprefix("USGS-")
    frame["month"] = frame.date.dt.to_period("M").astype(str)
    paper_window = frame.date.between("1984-01-01", "2020-12-31")
    overlapping = frame.canonical_site.isin(sites)
    receipt["grqa"] = {
        "sha256": digest(CACHE / "raw/grqa/DOC_GRQA.csv"), "records": len(frame),
        "unique_stations": frame.site_id.nunique(), "source_counts": frame.source.value_counts().to_dict(),
        "units": frame.unit.value_counts(dropna=False).to_dict(),
        "records_in_paper_window": int(paper_window.sum()),
        "duplicate_observation_ids": int(frame.obs_id.duplicated().sum()),
        "negative_doc": int((frame.obs_value < 0).sum()),
        "missing_coordinates": int(frame[["lat_wgs84", "lon_wgs84"]].isna().any(axis=1).sum()),
        "invalid_dates": int(frame.date.isna().sum()),
        "exact_id_overlap_with_our_stations": frame.loc[overlapping, "canonical_site"].nunique(),
        "records_at_our_station_ids": int(overlapping.sum()),
        "detection_flags": frame.detection_limit_flag.value_counts(dropna=False).to_dict(),
        "outlier_flags": frame.obs_iqr_outlier.value_counts(dropna=False).to_dict(),
        "scope": "Exact-ID overlap only; spatial/alias overlap not yet resolved. No training eligibility inferred."}
    query_rows = []
    for region in protocol["regions"]:
        path = ROOT / "experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks" / f"huc4_{region}.npz"
        if digest(path) != protocol["mask_sha256"][region]:
            raise ValueError("Frozen comparison mask differs")
        with np.load(path, allow_pickle=False) as saved:
            role_cells = {role: saved[role] for role in ("train", "val", "test")}
        role_sites = {role: sites[np.unique(indices // len(months))] for role, indices in role_cells.items()}
        for role, indices in role_cells.items():
            query_rows.extend({"region": region, "role": role, "cell": int(cell),
                               "site_no": sites[cell // len(months)], "month": months[cell % len(months)]}
                              for cell in indices)
        target_months = months[role_cells["test"] % len(months)]
        relevant_raw = local.site_no.isin(role_sites["test"])
        relevant_global = frame.canonical_site.isin(role_sites["test"])
        receipt.setdefault("by_target_basin", []).append({
            "huc4": region, "target_stations": len(role_sites["test"]),
            "target_monthly_cells": len(role_cells["test"]),
            "target_cells_in_paper_1984_2020_window": int(((target_months >= "1984-01") &
                                                          (target_months <= "2020-12")).sum()),
            "raw_samples_at_target_stations": int(relevant_raw.sum()),
            "grqa_exact_id_target_stations": frame.loc[relevant_global, "canonical_site"].nunique(),
            "grqa_records_at_target_station_ids": int(relevant_global.sum()),
            "leakage_requirement": "Exclude all target/validation basin observations from training, including aliases and nearby stations."})
    queries = pd.DataFrame(query_rows)
    queries.to_parquet(CACHE / "frozen_comparison_cells.parquet", index=False)
    receipt["comparison_cells_file_sha256"] = digest(CACHE / "frozen_comparison_cells.parquet")
    write_json(OUTPUT / "data_quality_audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
