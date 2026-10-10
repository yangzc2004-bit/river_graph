"""Frozen-study contracts and a test of order-scanned message propagation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.models.river_architecture_comparison import (
    ARMS,
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/phase4_transfer/doc_river_architecture_comparison_v1"
DATASET = ROOT / "data/processed/mississippi_graph_graphfix_st357.pt"
_PROTOCOL = (json.loads((STUDY / "protocol.json").read_text())
             if (STUDY / "protocol.json").exists() else {})
_REQUIRED = [DATASET, STUDY / "protocol.json", ROOT / "data/processed/graph_nodes_graphfix_st357.csv"]
_REQUIRED += [ROOT / "experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks" / f"huc4_{region}.npz"
              for region in _PROTOCOL.get("regions", [])]


def test_order_scan_propagates_across_multiple_order_changes():
    torch.manual_seed(42)
    windows, env, season = torch.randn(1, 4, 12, 8), torch.randn(4, 15), torch.randn(1, 2)
    graph = graph_view(np.array([[0, 1, 2], [1, 2, 3]]), np.array([1, 2, 3, 4]), np.arange(4))
    changed = env.clone()
    changed[0] += 3
    flat = RiverArchitectureModel("directed_gnn", layers=1, dropout=0).eval()
    hierarchical = RiverArchitectureModel("hierarchical_gnn", layers=1, dropout=0).eval()
    hierarchical.load_state_dict(flat.state_dict())
    torch.testing.assert_close(flat(windows, env, season, graph)[:, 3],
                               flat(windows, changed, season, graph)[:, 3])
    delta = hierarchical(windows, changed, season, graph) - hierarchical(windows, env, season, graph)
    assert abs(float(delta[0, 3].detach())) > 1e-6


@pytest.mark.skipif(any(not path.is_file() for path in _REQUIRED),
                    reason="local ST357 dataset, node metadata, frozen protocol and geographical masks are required")
def test_frozen_masks_and_source_only_inputs():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert protocol["arms"] == list(ARMS)
    assert protocol["seeds"] == [42, 43, 44]
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    t = data["y"].shape[1]
    nodes = pd.read_csv(ROOT / "data/processed/graph_nodes_graphfix_st357.csv",
                        dtype={"site_no": str, "huc_cd": str}).set_index("site_no").reindex(data["site_no"])
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == protocol["dataset_hash"]
    for region in protocol["regions"]:
        path = ROOT / "experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks" / f"huc4_{region}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["mask_hashes"][region]
        with np.load(path, allow_pickle=False) as saved:
            source = np.unique(saved["train"] // t)
            target = np.unique(saved["test"] // t)
            validation = np.unique(saved["val"] // t)
            assert not np.intersect1d(source, np.r_[target, validation]).size
            assert (nodes.iloc[target].huc_cd.str.zfill(8).str[:4] == region).all()
        covariates = {key: value for key, value in data.items() if key not in ("y", "y_mask")}
        inputs = covariate_inputs(covariates, source)
        assert inputs["normalization"]["source_rows"] == source.tolist()
        graph = graph_view(data["edge_index"], inputs["order"], source)
        assert not np.intersect1d(graph["rows"], np.r_[target, validation]).size
