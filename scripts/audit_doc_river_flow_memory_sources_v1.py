"""Inventory authorized independent discharge access and analysis dependencies."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("experiments/phase4_transfer/doc_river_flow_memory_v1")


def main():
    endpoint = "https://data.neonscience.org/api/v0/data/DP4.00130.001/BLUE/2024-06"
    try:
        response = requests.get(endpoint, timeout=20)
        status = response.status_code
    except requests.RequestException as exc:
        status = type(exc).__name__
    audit = {"official_discharge_product": "DP4.00130.001",
             "download_endpoint_test": endpoint, "http_status": status,
             "documentation": "https://data.neonscience.org/data-api/endpoints/data/",
             "public_viewer": "https://openflow.neonscience.org/",
             "auth_used": False, "raw_discharge_obtained": False,
             "boundary": "Official raw-data endpoint requires user-authorized authentication; no credentials sought. Public viewer is not an archived QC-complete raw extract.",
             "independent_sites_with_existing_form": len(pd.read_csv(
                 "experiments/phase4_transfer/doc_river_neon_form_validation_v1/analysis/site_panel_primary.csv")),
             "notebook_dependencies": {m: importlib.util.find_spec(m) is not None for m in ("nbformat", "nbclient", "ipykernel")}}
    (ROOT/"independent_flow_acquisition.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
