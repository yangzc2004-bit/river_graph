"""Build DOC-only external inputs from coordinates, StreamCat and hydro records.

No model is fitted or evaluated. External DOC values determine inventory and
the scoring table only; the label-free input archive is saved separately.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import write_json

from river_graph.data.aggregate import to_monthly
from river_graph.data.nwis import fetch_daily_discharge, load_daily_discharge
from river_graph.data.quality import apply_rules
from river_graph.data.streamcat import fetch_metrics
from river_graph.data.wqp import (
    _payload_looks_complete,
    extract_covariate_obs,
    load_station_results,
)
from river_graph.dataset import build_dataset
from river_graph.experiments.provenance import sha256_file
from river_graph.models.daily_flow_features import build_daily_flow_features
from river_graph.topology.nldi import get_station_hydrolocation, snap_point_to_comid

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1")


def fetch_temperature(code, raw, first_year, last_year):
    """Small calendar windows avoid the WQP whole-history truncation path."""
    paths = []
    for first in range(first_year, last_year+1, 3):
        last = min(first+2, last_year)
        path = raw / f"temperature_{first}_{last}.csv"
        paths.append(path)
        if path.exists() and _payload_looks_complete(path.read_bytes(), None):
            continue
        params = {"huc": code, "providers": "NWIS", "characteristicName": "Temperature, water",
                  "dataProfile": "narrow", "mimeType": "csv", "startDateLo": f"01-01-{first}",
                  "startDateHi": f"12-31-{last}"}
        url = "https://www.waterqualitydata.us/wqx3/Result/search?"+urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "river-graph-research",
                                                       "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(30_000_001)
            declared = response.headers.get("Content-Length")
        if len(payload) > 30_000_000 or not _payload_looks_complete(payload, declared):
            raise ValueError(f"external temperature window {first}-{last} incomplete or too large")
        path.write_bytes(payload)
        write_json(path.with_suffix(".request.json"), {"url": url, "sha256": sha256_file(path)})
        print(json.dumps({"stage": "temperature", "first_year": first, "last_year": last,
                          "bytes": len(payload)}), flush=True)
    return paths


def match_one(row, raw):
    result = {"site_no": row.site_no, "comid": None, "measure": None, "status": "pending"}
    try:
        result.update(get_station_hydrolocation(row.site_no, raw / "comid"))
        if result["comid"] is None:
            result["comid"] = snap_point_to_comid(row.dec_lat_va, row.dec_long_va, raw / "snap")
        result["status"] = "matched" if result["comid"] is not None else "unmatched"
    except Exception as error:  # noqa: BLE001 - preserve transient failures for resumption
        result.update(status="network_error", error=f"{type(error).__name__}: {error}")
    return result


def ecology_columns(row):
    def value(name):
        return pd.to_numeric(row.get(name, np.nan), errors="coerce")
    return [sum(value(name+"ws") for name in names) for names in
            (("pctconif2019", "pctdecid2019", "pctmxfst2019"), ("pctcrop2019", "pcthay2019"),
             ("pcturbhi2019", "pcturbmd2019", "pcturblo2019", "pcturbop2019"),
             ("pctwdwet2019", "pcthbwet2019"))] + [
                value("precip9120ws"), np.log1p(value("tmean9120ws")+20),
                value("omws"), value("elevws"), value("bfiws")]


def build(code, root):
    destination = root / "external_inputs" / code
    destination.mkdir(parents=True, exist_ok=True)
    raw = Path("data/raw/doc_external_inputs_v1") / code
    raw.mkdir(parents=True, exist_ok=True)
    inventory = root / "external_availability" / code
    screen = json.loads((inventory / "screen.json").read_text())
    if not screen["availability_pass"]:
        raise ValueError("DOC inventory does not support the external task")
    stations = pd.read_csv(inventory / "stations.csv", dtype={"site_no": str})
    nodes = pd.DataFrame({"site_no": stations.site_no,
        "dec_lat_va": pd.to_numeric(stations.Location_LatitudeStandardized, errors="coerce"),
        "dec_long_va": pd.to_numeric(stations.Location_LongitudeStandardized, errors="coerce")})
    if not np.isfinite(nodes[["dec_lat_va", "dec_long_va"]]).all().all() or nodes.site_no.duplicated().any():
        raise ValueError("external coordinates/station identities are incomplete")
    location_file = destination / "hydrolocations.csv"
    previous = pd.read_csv(location_file, dtype={"site_no": str, "comid": str}) if location_file.exists() else pd.DataFrame()
    saved = previous.set_index("site_no").to_dict("index") if len(previous) else {}
    rows, pending = [], []
    for row in nodes.itertuples(index=False):
        old = saved.get(row.site_no)
        if old and old["status"] in ("matched", "unmatched"):
            rows.append({"site_no": row.site_no, **old})
        else:
            pending.append(row)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(lambda row: match_one(row, raw), pending):
            rows.append(result)
            pd.DataFrame(rows).to_csv(location_file, index=False)
            print(json.dumps({"stage": "hydrolocation", "completed": len(rows), "stations": len(nodes),
                              "status": result["status"]}), flush=True)
    locations = pd.DataFrame(rows).set_index("site_no").loc[nodes.site_no].reset_index()
    if locations.status.eq("network_error").any():
        write_json(destination / "progress.json", {"stage": "retry_hydrolocations", "network_errors":
                                                    locations[locations.status.eq("network_error")].site_no.tolist()})
        raise RuntimeError("transient hydrolocation failures remain; resume the same inventory")
    comids = sorted(locations.comid.dropna().astype(str).unique())
    sc = fetch_metrics(comids, raw / "streamcat") if comids else pd.DataFrame(columns=["comid"])
    sc["comid"] = sc.comid.astype(str)
    sc.to_csv(destination / "streamcat.csv", index=False)
    lookup = sc.set_index("comid")
    vaa = pd.read_parquet("cache/nldplus_vaa.parquet", columns=["comid", "streamorde", "totdasqkm", "slope"])
    vaa["comid"] = vaa.comid.astype(str)
    reach = vaa.set_index("comid")
    regime = []
    for row in locations.itertuples(index=False):
        c = str(row.comid)
        hydraulic = ([reach.loc[c, "streamorde"], np.log1p(reach.loc[c, "totdasqkm"]),
                      reach.loc[c, "slope"], np.nan] if c in reach.index else [np.nan]*4)
        # Graph headwater is unknown until the optional river graph is built.
        regime.append(hydraulic+(ecology_columns(lookup.loc[c]) if c in lookup.index else [np.nan]*9))
    regime = np.nan_to_num(np.asarray(regime, dtype=np.float32), nan=-1)
    temperature_paths = fetch_temperature(code, raw, int(screen["min_month"][:4]), int(screen["max_month"][:4]))
    temperature = pd.concat([extract_covariate_obs(load_station_results(path), "temperature")
                             for path in temperature_paths], ignore_index=True)
    temperature = temperature[temperature.site_no.isin(nodes.site_no)]
    fetch_daily_discharge(nodes.site_no.tolist(), raw / "nwis_dv", batch_size=5)
    flow = load_daily_discharge(raw / "nwis_dv")
    flow = flow[flow.site_no.isin(nodes.site_no)].copy()
    candidates = flow[["site_no", "date"]].copy()
    candidates["value"], candidates["unit"] = flow.discharge_cfs.to_numpy(), "ft3/s"
    audited = apply_rules(candidates, "discharge")
    monthly_flow = to_monthly(audited[audited.qc_status.eq("accepted")][["site_no", "date", "value"]]
                             .rename(columns={"value": "discharge"}), "discharge")
    labels = pd.read_parquet(inventory / "monthly_doc.parquet")
    labels["month"] = pd.to_datetime(labels.month)
    months = pd.date_range(screen["min_month"], screen["max_month"], freq="MS")
    dataset = build_dataset(nodes, pd.DataFrame(columns=["source", "target"]), labels,
        {"temperature": to_monthly(temperature, "temperature"), "discharge": monthly_flow},
        months, edge_attr=np.empty((0, 6)), regime=regime)
    # Empty edges have shape [2,0] in this dataset builder.
    daily = build_daily_flow_features({key: dataset[key] for key in ("site_no", "months", "x_mask")},
                                      flow)
    torch.save(dataset, destination / "dataset_scoring_only.pt")
    np.savez_compressed(destination / "inputs.npz", x=np.asarray(dataset["x"]), x_mask=np.asarray(dataset["x_mask"]),
        regime=regime, static=np.asarray(dataset["static"]), daily=daily["full"],
        site_no=np.asarray(dataset["site_no"], str), months=np.asarray(dataset["months"], str))
    nodes.to_csv(destination / "nodes.csv", index=False)
    report = {"huc8": code, "station_count": len(nodes), "months": len(months),
        "valid_doc_cells": int(np.asarray(dataset["y_mask"]).sum()), "comid_matches": int(locations.comid.notna().sum()),
        "ecology_ge5_stations": int(((regime[:, 4:] != -1).sum(1) >= 5).sum()),
        "temperature_cells": int(np.asarray(dataset["x_mask"])[..., 0].sum()),
        "discharge_cells": int(np.asarray(dataset["x_mask"])[..., 1].sum()),
        "river_graph": "not constructed; optional separate transport ablation",
        "scope": "availability and feature construction only; no external prediction or calibration",
        "dataset_hash": sha256_file(destination / "dataset_scoring_only.pt"),
        "inputs_hash": sha256_file(destination / "inputs.npz"), "inventory": screen,
        "raw_inputs": {str(path): sha256_file(path) for path in raw.rglob("*") if path.is_file()},
        "builder_hash": sha256_file(__file__), "daily_policy": daily.get("policy", "existing daily feature function")}
    write_json(destination / "complete.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("raw_inputs", "inventory")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huc8", default="02040104")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.huc8, args.root)


if __name__ == "__main__":
    main()
