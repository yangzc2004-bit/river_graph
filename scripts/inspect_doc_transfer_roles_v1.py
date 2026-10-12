"""Inspect geographic split semantics before transfer protocol selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.models.transfer_robust_doc import huc_labels

if __name__ == "__main__":
    data = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str, "huc_cd": str})
    groups = huc_labels(nodes, data["site_no"])
    print(json.dumps({"unparsed_huc_examples": nodes.set_index("site_no").loc[np.asarray(data["site_no"])[groups["huc2"] == "unknown"], "huc_cd"].head(8).tolist()}))
    report = {"huc2_stations": dict(zip(*[x.tolist() for x in np.unique(groups["huc2"], return_counts=True)], strict=True)), "tasks": {}}
    for region in ("1013", "1019", "0708", "1030", "1101"):
        with np.load(Path("experiments/phase4_transfer/doc_river_relation_v2/masks")/f"space_{region}.npz") as saved:
            report["tasks"][region] = {role: {"stations": len(np.unique(saved[role]//data["y"].shape[1])),
                "huc4": np.unique(groups["huc4"][saved[role]//data["y"].shape[1]]).tolist()} for role in ("train", "val", "test")}
    print(json.dumps(report, indent=2))
