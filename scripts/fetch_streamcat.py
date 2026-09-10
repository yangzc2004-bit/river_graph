"""Fetch StreamCat ecological context for all graph COMIDs.

Output: data/processed/streamcat_attributes.csv (one row per comid).

Usage: python scripts/fetch_streamcat.py
"""

import pickle
from pathlib import Path

from river_graph.data.streamcat import fetch_metrics

PROCESSED = Path("data/processed")


def main() -> None:
    with open(PROCESSED / "mississippi_graph.pkl", "rb") as fh:
        nodes = pickle.load(fh)["nodes"]
    comids = sorted({str(c) for c in nodes["comid"].dropna()})
    print(f"{len(comids)} unique comids")

    df = fetch_metrics(comids, "data/raw/streamcat")
    df.to_csv(PROCESSED / "streamcat_attributes.csv", index=False)
    print(f"fetched {len(df)} comids x {len(df.columns) - 1} metrics")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
