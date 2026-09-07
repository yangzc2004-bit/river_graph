"""Fetch the DOC station inventory + qw series catalog for the study region.

Resumable: existing downloads are reused. Run from the repo root:

    python scripts/fetch_doc_inventory.py
"""

from river_graph.config import load_config
from river_graph.data.nwis import fetch_doc_sites, fetch_qw_catalog


def main() -> None:
    cfg = load_config()
    sites = fetch_doc_sites(cfg["region"]["huc2"], "data/raw")
    print(f"{len(sites):,} DOC candidate sites in region")
    fetch_qw_catalog(sorted(sites["site_no"]), "data/raw/nwis_catalog")
    print("catalog done")


if __name__ == "__main__":
    main()
