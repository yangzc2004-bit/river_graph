"""Frozen component protocol and whole-region mask contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")


@pytest.mark.skipif(not DATA.exists() or not (ROOT / "protocol.json").exists(),
                    reason="local ST357 data or frozen component protocol unavailable")
def test_frozen_runtime_and_whole_region_masks():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == manifest["protocol_hash"]
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == protocol["dataset_hash"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((ROOT / "code_snapshot" / name).read_bytes()).hexdigest() == expected
    months = torch.load(DATA, weights_only=False)["y"].shape[1]
    for region, expected in protocol["mask_hashes"].items():
        path = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks") / f"huc4_{region}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        with np.load(path) as split:
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                assert not np.intersect1d(split[left] // months, split[right] // months).size
    assert protocol["unique_fits"] == 165
    assert protocol["stages"]["environment"]["eligible"] == ["mlp", "linear", "residual_mlp"]
    assert "current" not in protocol["stages"]["time"]["eligible"]
    assert not protocol["old_protocols_modified"]
