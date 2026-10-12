"""Label-free schema/topology consistency receipt for the transfer comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unified_transfer_v2")


def main():
    source = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    source_edges = np.asarray(source["edge_index"])
    flags = np.asarray(source["regime"])[:, 3]
    current_flag = np.bincount(source_edges[1], minlength=len(flags)) == 0
    source_nodes = pd.read_csv("data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str})
    external_nodes = pd.read_csv(ROOT/"external/nodes.csv", dtype={"site_no": str})
    source_comid = set(pd.to_numeric(source_nodes.comid, errors="coerce").dropna().astype(np.int64))
    external_comid = set(pd.to_numeric(external_nodes.comid, errors="coerce").dropna().astype(np.int64))
    report = {"source_stations": len(flags), "source_stream_order_positive": bool((np.asarray(source["regime"])[:, 0] > 0).all()),
        "stored_source_headwater_flags": int(np.sum(flags == 1)), "current_graph_zero_in_degree": int(current_flag.sum()),
        "source_flag_current_graph_mismatch": int(np.sum(flags != current_flag)),
        "source_flag_definition": "original build_dataset computes zero incoming station edges; frozen source values retained, never retrospectively changed",
        "receiving_flag_definition": "same construction formula on the independently constructed receiving station cohort",
        "source_receiving_comid_overlap": len(source_comid & external_comid),
        "channel_order": list(source["feature_channels"]), "temperature_unit": "degC", "discharge_unit": "ft3/s",
        "code_sha256": sha256_file(__file__), "external_input_sha256": sha256_file(ROOT/"external/inputs.npz")}
    (ROOT/"schema_inspection.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
