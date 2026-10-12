"""Frozen execution revision and common-cohort contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_network_work_v2")


@pytest.mark.skipif(not (ROOT/"protocol.json").exists(), reason="parallel DOC comparison not yet frozen")
def test_parallel_protocol_preserves_data_recipes_and_source_gate():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    old = json.loads(Path("experiments/phase4_transfer/doc_network_work_v1/protocol.json").read_text())
    assert protocol["workers"] == 5 and protocol["threads"] == 1
    assert protocol["runtime_benchmark"]["source_only"] is True
    assert protocol["runtime_benchmark"]["doc_labels_read"] is False
    assert protocol["native_reproductions"] is False
    assert protocol["unique_new_fits"] == 45
    for name in ("new_recipes", "neural_settings", "regions", "seeds", "models", "dataset_sha256", "nodes_sha256", "mask_sha256", "joint_primary", "selection"):
        assert protocol[name] == old[name]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT/"code_snapshot"/name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected


@pytest.mark.skipif(not (ROOT/"all_selections.json").exists(), reason="parallel source selections not complete")
def test_all_source_selections_bound_before_query_outputs():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    gate = json.loads((ROOT/"all_selections.json").read_text())
    assert gate["test_used"] is False
    for region in protocol["regions"]:
        run = ROOT/"regions"/f"huc4_{region}"
        assert sha256_file(run/"selection_complete.json") == gate["regional_selections"][region]
        assert json.loads((run/"selection.json").read_text())["test_used"] is False
