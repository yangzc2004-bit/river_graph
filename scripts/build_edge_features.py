"""Build edge features for the station graph -> data/processed/edge_features.csv

Usage: python scripts/build_edge_features.py
"""

import pickle
from pathlib import Path

from river_graph.topology.edge_features import compute_edge_features

PROCESSED = Path("data/processed")


def main() -> None:
    with open(PROCESSED / "mississippi_graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    out = compute_edge_features(graph["nodes"], graph["edges"],
                                Path("data/raw/nldi/flowlines"))
    out.to_csv(PROCESSED / "edge_features.csv", index=False)
    print(f"edges: {len(out)}; hop distance recovered: {out['hop_dist'].notna().sum()}")
    print(out["hop_dist"].describe().round(1))


if __name__ == "__main__":
    main()
