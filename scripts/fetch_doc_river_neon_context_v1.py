"""Acquire full-watershed StreamCat context and its actual coverage fractions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from river_graph.data.streamcat import METRICS

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")
RAW = Path("data/raw/river_neon_form_validation_v1")
API = "https://api.epa.gov/StreamCat/streams/metrics"


def main():
    locations = pd.read_csv(ROOT / "analysis" / "registered_locations.csv")
    ids = sorted(locations.comid.dropna().astype(int).unique().tolist())
    parameters = {"name": ",".join(METRICS[:13]), "aoi": "ws", "comid": ",".join(map(str, ids)),
                  "showpctfull": "true", "showareasqkm": "true"}
    path = RAW / "streamcat_ws_coverage.json"
    if path.exists():
        value = json.loads(path.read_text())
    else:
        response = requests.post(API, data=parameters, timeout=(15, 120))
        response.raise_for_status()
        value = response.json()
        if not value.get("items"):
            raise ValueError("no watershed context returned")
        path.write_text(json.dumps(value, indent=2) + "\n")
    frame = pd.DataFrame(value["items"])
    if frame.comid.duplicated().any() or not set(frame.comid).issubset(ids):
        raise ValueError("StreamCat COMIDs differ from registered locations")
    ledger = {"endpoint": API, "parameters": parameters, "requested_reaches": len(ids),
              "returned_reaches": len(frame), "missing_comids": sorted(set(ids)-set(frame.comid)),
              "missing_is_not_zero_cover": True,
              "coverage_fields": [c for c in frame if "pctfull" in c],
              "interpretation": "CONUS watershed land cover 2019 and climate normals 1991–2020"}
    (ROOT / "context_retrieval.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
