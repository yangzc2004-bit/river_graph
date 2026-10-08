"""Bounded public DOC recovery for fifteen metadata-selected river-form pairs."""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from analyze_doc_river_monitored_arrivals_v1 import load_context, routed_network

from river_graph.data.wqp import _payload_looks_complete, _query_url
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_morphology_closure_v1")
PRIOR = Path("experiments/phase4_transfer/doc_river_cosampling_geometry_v1")
RAW = Path("data/raw/river_morphology_closure_v1")
TYPES = {"station": str, "target": str, "huc4": str,
         "elongated_station": str, "broad_station": str}


def prepare():
    out = ROOT / "recovery"
    out.mkdir(parents=True, exist_ok=True)
    priority = pd.read_csv(PRIOR / "analysis/non_nested_form_pair_priorities.csv", dtype=TYPES)
    selected = priority.sort_values("observation_priority").head(15).copy()
    selected.to_csv(out / "selected_form_pairs.csv", index=False)
    targets = set(selected.elongated_station) | set(selected.broad_station)
    _, nodes, mapping, panel, _, _, _, vaa = load_context()
    inventories, gauges = [], []
    files = []
    for row in panel[panel.station.isin(targets)].itertuples():
        paths, _, hydro, inputs = routed_network(row, vaa)
        files.extend(inputs)
        inside = nodes[nodes.comid.isin(paths.comids) & nodes.comid.ne(row.comid)]
        inside = inside.loc[mapping.loc[inside.index, "mapping_status"].ne("gross_area_mismatch")]
        # All mapped gauges are retained; no DOC outcomes or date intersections
        # determine this expanded collection list.
        for station, g in inside.iterrows():
            gauges.append({"target": row.station, "station": station,
                "comid": int(g.comid), "measure": float(g.measure),
                "reported_area_km2": mapping.loc[station, "drainage_area_km2"]})
        inventories.append({"station": row.station, "cluster": row.cluster,
            "huc4": row.huc4, "comid": int(row.comid),
            "station_name": nodes.loc[row.station, "station_nm"],
            "receiver_area_km2": mapping.loc[row.station, "drainage_area_km2"],
            "n_mapped_upstream_gauges": len(inside), "n_routed_reaches": len(hydro)})
    f = pd.DataFrame(gauges, columns=["target", "station", "comid", "measure", "reported_area_km2"])
    f.to_csv(out / "mapped_gauges.csv", index=False)
    pd.DataFrame(inventories).to_csv(out / "receivers.csv", index=False)
    sites = sorted(targets | set(f.station))
    pd.DataFrame({"station": sites, "receiver": [s in targets for s in sites],
        "wqp_url": [_query_url(s, characteristics=["Organic carbon"]) for s in sites]}).to_csv(out / "station_requests.csv", index=False)
    (out / "selection_sources.json").write_text(json.dumps({
        "selected_pairs": len(selected), "unique_receivers": len(targets), "requested_stations": len(sites),
        "selection": "saved first fifteen non-nested priorities, metadata only",
        "scope": "full public discrete DOC archive, including cells outside earlier model source roles",
        "source_hashes": {str(p): sha256_file(p) for p in [
            PRIOR / "analysis/non_nested_form_pair_priorities.csv", ROOT / "study_plan.md", *set(files)]}}, indent=2) + "\n")
    print(json.dumps({"pairs": len(selected), "receivers": len(targets), "stations": len(sites)}), flush=True)


def fetch_one(item):
    station, url = item.station, item.wqp_url
    path = RAW / f"{station}.csv"
    record = {"station": station, "url": url, "attempted_at": datetime.now(timezone.utc).isoformat()}
    if path.exists():
        record.update(status="saved_download", path=str(path), sha256=sha256_file(path))
        return record
    try:
        response = requests.get(url, timeout=(8, 25), headers={"User-Agent": "river-graph-research", "Accept-Encoding": "identity"})
        record.update(http_status=response.status_code, response_bytes=len(response.content))
        if response.ok and _payload_looks_complete(response.content, response.headers.get("Content-Length")):
            # Validate CSV framing before accepting the archive. Values are
            # inspected only by the following analysis, after station selection.
            frame = pd.read_csv(io.BytesIO(response.content), dtype=str, low_memory=False)
            if "Location_Identifier" not in frame:
                raise ValueError("WQP response has no station identifier")
            path.write_bytes(response.content)
            record.update(status="downloaded", path=str(path), sha256=sha256_file(path), raw_rows=len(frame))
        else:
            record.update(status="provider_unavailable_or_incomplete",
                error=response.text[:250] if not response.ok else "Incomplete or unsupported CSV framing")
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        record.update(status="access_error", error=str(exc)[:400])
    old = Path("data/raw/wqp_results") / f"{station}.csv"
    if "path" not in record and old.exists():
        record.update(fallback_path=str(old), fallback_sha256=sha256_file(old))
    print(f"{station}: {record['status']}", flush=True)
    return record


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    inventory = ROOT / "recovery/retrieval_manifest.json"
    if inventory.exists():
        print("Retrieval already recorded; preserve access outcomes and downloaded bytes", flush=True)
        return
    sites = pd.read_csv(ROOT / "recovery/station_requests.csv", dtype={"station": str})
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, sites.itertuples()))
    # Official continuous DOC availability is a separate discovery source.
    continuous_url = ("https://waterservices.usgs.gov/nwis/site/?format=rdb&sites="
        + ",".join(sites.loc[sites.receiver, "station"]) + "&seriesCatalogOutput=true&parameterCd=00681&hasDataTypeCd=iv")
    continuous = {"url": continuous_url, "parameter": "00681 dissolved organic carbon"}
    try:
        response = requests.get(continuous_url, timeout=(8, 25))
        continuous.update(http_status=response.status_code, response_bytes=len(response.content))
        if response.ok:
            path = RAW / "continuous_doc_catalog.rdb"
            path.write_bytes(response.content)
            continuous.update(status="retrieved", path=str(path), sha256=sha256_file(path))
        else:
            continuous.update(status="provider_unavailable", error=response.text[:250])
    except requests.RequestException as exc:
        continuous.update(status="access_error", error=str(exc)[:400])
    inventory.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
        "records": records, "continuous_doc_catalog": continuous,
        "attempt_policy": "one bounded attempt per fixed station; preserve local archive fallback",
        "source_selection_uses_doc_outcomes": False}, indent=2) + "\n")
    print(pd.Series([r["status"] for r in records]).value_counts().to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare()
    if args.fetch:
        fetch()
    if not (args.prepare or args.fetch):
        p.error("choose --prepare or --fetch")


if __name__ == "__main__":
    main()
