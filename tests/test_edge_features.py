"""Tests for edge feature extraction (synthetic cache, no network)."""

import json

import numpy as np
import pandas as pd

from river_graph.topology.edge_features import compute_edge_features


def _write_walk(cache, comid, walk, dist=500):
    feats = [{"properties": {"nhdplus_comid": c}} for c in walk]
    (cache / f"comid_{comid}_dm{dist}.json").write_text(json.dumps({"features": feats}))


def test_hop_distance_from_walk(tmp_path):
    _write_walk(tmp_path, "100", ["100", "200", "300"])
    nodes = pd.DataFrame(
        {
            "site_no": ["A", "B", "C"],
            "comid": ["100", "200", "300"],
            "dec_long_va": [-90.0, -90.5, -91.0],
            "dec_lat_va": [40.0, 40.0, 40.0],
        }
    )
    edges = pd.DataFrame({"source": ["A", "B", "C"], "target": ["B", "C", "C"]})
    out = compute_edge_features(nodes, edges, tmp_path)
    # A->B: B at index 1 in A's walk; B->C: walk for 200 not cached -> NaN;
    # C->C self-loop-ish: C's walk not cached -> NaN
    assert out["hop_dist"].tolist()[0] == 1.0
    assert np.isnan(out["hop_dist"].iloc[1])
    # geo distance A->B is 0.5 degrees
    assert abs(out["geo_dist_deg"].iloc[0] - 0.5) < 1e-9
