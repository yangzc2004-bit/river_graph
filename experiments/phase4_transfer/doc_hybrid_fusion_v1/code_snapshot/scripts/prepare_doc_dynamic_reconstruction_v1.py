"""Audit alignment, cached DOC aggregation and daily inputs; never alter labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.data.aggregate import to_monthly
from river_graph.data.wqp import (
    extract_covariate_obs,
    extract_doc_obs,
    load_station_results,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DAILY = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1/daily_features.npz")


def main():
    data = torch.load(DATASET,weights_only=False,map_location="cpu")
    meta = json.loads(DAILY.with_suffix(".meta.json").read_text())
    if meta["dataset_hash"] != sha256_file(DATASET) or meta["feature_product_hash"] != sha256_file(DAILY):
        raise ValueError("daily descriptor product no longer matches its frozen inputs")
    with np.load(DAILY,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["site_no"].astype(str),np.asarray(data["site_no"],str))
        np.testing.assert_array_equal(archive["months"].astype(str),np.asarray(data["months"],str))
        daily=archive["full"].copy()
    assert daily.shape==(*data["y"].shape,8) and np.isfinite(daily).all()
    assert not daily[~np.asarray(data["x_mask"],bool)[...,1]].any()
    calendar=pd.DatetimeIndex(data["months"])
    assert calendar.to_period("M").is_unique and np.all(np.diff(calendar.to_period("M").asi8)==1)
    assert list(data["feature_channels"])==["temperature","discharge"]
    y=np.asarray(data["y"],float);mask=np.asarray(data["y_mask"],bool)
    rows=[];high=[];input_files={};value_errors=[];temperature_errors=[]
    for index,site in enumerate(data["site_no"]):
        path=Path("data/raw/wqp_results")/f"{site}.csv"
        if not path.exists():
            rows.append({"site_no":str(site),"raw_present":False});continue
        input_files[str(path)]=sha256_file(path)
        raw=load_station_results(path)
        doc=extract_doc_obs(raw)
        monthly=to_monthly(doc,"doc").set_index("month").doc
        expected=monthly.reindex(calendar).to_numpy(float)
        visible=mask[index] & np.isfinite(expected)
        errors=np.abs(expected[visible]-y[index,visible]);value_errors.extend(errors.tolist())
        t=to_monthly(extract_covariate_obs(raw,"temperature"),"temperature").set_index("month").temperature
        t_expected=t.reindex(calendar).to_numpy(float)
        actual=np.asarray(data["x"],float)[index,:,0]
        tmask=np.asarray(data["x_mask"],bool)[index,:,0] & np.isfinite(t_expected)
        temperature_errors.extend(np.abs(actual[tmask]-t_expected[tmask]).tolist())
        rows.append({"site_no":str(site),"raw_present":True,"stored_doc_months":int(mask[index].sum()),
                     "raw_reproduced_months":int(visible.sum()),"max_doc_absolute_difference":float(errors.max()) if len(errors) else None,
                     "additional_current_cache_months":int((~mask[index] & np.isfinite(expected)).sum()),
                     "exact_raw_duplicate_rows":int(raw.duplicated().sum())})
        accepted_code=(raw.usgs_pcode=="00681") | (raw.usgs_pcode.isna() & raw.fraction.str.lower().isin(["dissolved","filtered field and/or lab"]))
        selected=raw[(raw.variable=="organic_carbon") & (raw.value>=50) & accepted_code
                     & (raw.unit.str.lower()=="mg/l") & raw.detection_condition.isna()].copy()
        selected=selected[[c for c in ("site_no","date","value","unit","usgs_pcode","fraction","detection_condition") if c in selected]]
        high.extend(selected.assign(date=selected.date.astype(str)).to_dict("records"))
        if (index+1)%75==0:print(json.dumps({"stage":"raw_audit","stations":index+1}),flush=True)
    observed=y[mask]
    report={"dataset_sha256":sha256_file(DATASET),"daily_sha256":sha256_file(DAILY),
            "raw_file_hashes":input_files,"raw_stations_checked":len(input_files),
            "observed_doc_cells":len(observed),"raw_reproduced_doc_cells":len(value_errors),
            "doc_monthly_mean_max_difference":max(value_errors) if value_errors else None,
            "temperature_same_month_mean_max_difference":max(temperature_errors) if temperature_errors else None,
            "doc_quantiles":dict(zip(("min","q50","q90","q95","q99","q999","max"),np.quantile(observed,[0,.5,.9,.95,.99,.999,1]).tolist())),
            "calendar_start":str(calendar[0]),"calendar_end":str(calendar[-1]),
            "months":len(calendar),"stations":len(y),"daily_shape":list(daily.shape),
            "daily_quality":meta["quality_summary"],"original_labels_modified":False,
            "target":"mean of available filtered, uncensored mg/L DOC samples in a calendar month; not a continuously sampled monthly mean",
            "input_scope":"retrospective same-month daily descriptors; no future month, DOC or target-fitted scaling",
            "extreme_policy":"retain all frozen observations; raw high values reported for inspection, not deleted"}
    ROOT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT/"raw_station_audit.csv",index=False)
    pd.DataFrame(high).to_csv(ROOT/"raw_high_doc_records.csv",index=False)
    (ROOT/"preparation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="raw_file_hashes"}))


if __name__=="__main__":
    main()
