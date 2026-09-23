"""Protocol-contract tests for frozen K-shot v2 (no training)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "experiments" / "kshot_protocol_v2"
_PROTO_JSON = (
    json.loads((PROTO / "protocol.json").read_text(encoding="utf-8"))
    if (PROTO / "protocol.json").exists()
    else {}
)
# Data paths come from the frozen protocol itself so a re-freeze on another
# cohort cannot leave these tests pointing at a stale hard-coded file.
DATASET = ROOT / _PROTO_JSON.get(
    "dataset", "data/processed/mississippi_graph_graphfix_st357.pt"
)
NODES = ROOT / _PROTO_JSON.get(
    "nodes", "data/processed/graph_nodes_graphfix_st357.csv"
)


@pytest.fixture(scope="module")
def ds():
    if not DATASET.is_file():
        pytest.skip(f"local dataset for frozen k-shot v2 missing: {DATASET}")
    return torch.load(DATASET, weights_only=False)


@pytest.fixture(scope="module")
def regions():
    return json.loads((PROTO / "regions.json").read_text())


@pytest.fixture(scope="module")
def y_mask(ds):
    return np.asarray(ds["y_mask"])


def test_protocol_file_exists():
    assert (PROTO / "protocol.json").exists()
    assert (PROTO / "manifest.json").exists()
    assert (PROTO / "connectivity_audit.csv").exists()
    assert (PROTO / "candidate_rejections.md").exists()


def test_hide_rows_are_full_huc6(ds, regions):
    nodes = pd.read_csv(NODES, dtype={"site_no": str})
    sites = [str(s) for s in ds["site_no"]]
    meta = pd.DataFrame({"station": sites}).merge(
        nodes[["site_no", "huc_cd"]].rename(columns={"site_no": "station"}),
        on="station",
        how="left",
        validate="one_to_one",
    )
    huc6 = meta["huc_cd"].astype(str).str[:6]
    for r in regions["primary"]:
        code = r["code"]
        expect = sorted(i for i, v in enumerate(huc6) if v == code)
        assert r["hide_rows"] == expect, code


def test_task_component_is_largest_connected_subset(ds, regions):
    ei = ds["edge_index"]
    for r in regions["primary"]:
        hide = set(r["hide_rows"])
        comp = r["task_component_rows"]
        assert set(comp) <= hide
        g = nx.Graph()
        g.add_nodes_from(hide)
        for a, b in ei.T.tolist():
            a, b = int(a), int(b)
            if a in hide and b in hide:
                g.add_edge(a, b)
        largest = max(nx.connected_components(g), key=len)
        assert sorted(largest) == comp, r["code"]


def test_tasks_full_ladder_nested_fixed_query(y_mask):
    k_list = [0, 1, 3, 5]
    flat_obs = {int(i) for i in np.flatnonzero(y_mask.ravel())}
    for path in (PROTO / "tasks").glob("*/*.json"):
        payload = json.loads(path.read_text())
        rows = set(payload["task_component_rows"])
        assert payload["k_list"] == k_list
        for task in payload["tasks"]:
            q = set(task["query_cells"])
            assert len(q) == payload["min_query"] or len(q) >= payload["min_query"]
            assert set(task["support_cells_by_k"]) == {"0", "1", "3", "5"}
            prev: list[int] = []
            for k in k_list:
                s = task["support_cells_by_k"][str(k)]
                assert len(s) == k
                assert not set(s) & q
                if prev:
                    assert set(prev) <= set(s)
                assert set(s) <= flat_obs
                assert q <= flat_obs
                # all inside component
                assert {c // y_mask.shape[1] for c in s} <= rows
                assert {c // y_mask.shape[1] for c in q} <= rows
                prev = s


def test_manifest_relative_paths_unique():
    manifest = json.loads((PROTO / "manifest.json").read_text())
    keys = list(manifest["artifact_hashes"])
    assert len(keys) == len(set(keys))
    assert any(k.startswith("tasks/") for k in keys)
    # basename collision would only keep one 42.json — require 5 regions × 3 seeds
    task_keys = [k for k in keys if k.startswith("tasks/") and k.endswith(".json")]
    assert len(task_keys) == 15, task_keys
    assert manifest.get("generator_script_sha256")
    assert manifest.get("dataset_sha256")


def test_manifest_runtime_code_snapshot():
    manifest = json.loads((PROTO / "manifest.json").read_text())
    snap = manifest.get("runtime_code_snapshot")
    assert snap, "manifest must record a runtime code snapshot"
    for rel in (
        "src/river_graph/models/gcn.py",
        "src/river_graph/models/hydro.py",
        "src/river_graph/models/support_encoder.py",
        "src/river_graph/experiments/kshot.py",
    ):
        assert rel in snap, rel
        path = ROOT / rel
        assert path.is_file()
        # recorded hash must match the file currently on disk
        assert snap[rel] == hashlib.sha256(path.read_bytes()).hexdigest(), rel


def test_base_model_config_frozen():
    proto = json.loads((PROTO / "protocol.json").read_text())
    cfg = proto["base_model"]["config"]
    for key in (
        "hidden",
        "layers",
        "dropout",
        "lr",
        "max_epochs",
        "patience",
        "env_groups",
        "env_encoder",
        "train_split_seed",
    ):
        assert key in cfg, key
    assert cfg["architecture"] == "transport_enc"
    assert proto["success_gate"]["primary_endpoint"] == "K=5"
    assert proto["k_list"] == [0, 1, 3, 5]
