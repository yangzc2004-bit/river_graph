"""T10/T11: run identity, artifact storage and the strict artifact audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.experiments.h3_runs import (
    ARTIFACT_KEYS,
    SMOKE_STATIONS,
    audit_run,
    dataset_scope,
    identity,
    payload_hash,
    run_stem,
    scenario_of,
    task_grid,
    tensor_digest,
    train_and_store,
)
from river_graph.experiments.h3_training import load_protocol

ROOT = Path(__file__).resolve().parents[1]
N_REGIME = 13
EDGE_DIM = 6
MASK_NAME = "e2b_demo"


def tiny_dataset(n: int = 8, t: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
    return {
        "site_no": [f"s{i}" for i in range(n)],
        "months": [f"2000-{m + 1:02d}" for m in range(t)],
        "edge_index": edge_index,
        "edge_attr": torch.rand(edge_index.shape[1], EDGE_DIM, generator=g),
        "y": torch.rand(n, t, generator=g) * 4 + 0.5,
        "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.rand(n, t, 2, generator=g) * 10,
        "x_mask": torch.ones(n, t, 2, dtype=torch.bool),
        "static": torch.rand(n, 2, generator=g),
        "regime": torch.rand(n, N_REGIME, generator=g),
        "feature_channels": ["temperature", "discharge"],
    }


def tiny_split() -> dict[str, np.ndarray]:
    used = np.array([0, 1, 2, 3, 20, 24, 25, 26, 28, 29], dtype=np.int64)
    return {
        "test": np.array([0, 1, 2, 3], dtype=np.int64),
        "train": np.setdiff1d(np.arange(32, dtype=np.int64), used),
        "val": np.array([24, 25, 26], dtype=np.int64),
        "val_context": np.array([20], dtype=np.int64),
        "context": np.array([28, 29], dtype=np.int64),
    }


def tiny_protocol(hidden: int = 16) -> dict:
    proto = copy.deepcopy(load_protocol())
    proto["training"]["max_epochs"] = 3
    proto["training"]["patience"] = 3
    proto["arms"]["env"]["hidden"] = hidden
    proto["arms"]["h2x"]["hidden"] = hidden
    proto["arms"]["h2x"]["env_emb"] = 8
    return proto


@pytest.fixture
def workspace(tmp_path):
    dataset_path = tmp_path / "tiny.pt"
    torch.save(tiny_dataset(), dataset_path)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    np.savez(masks_dir / f"{MASK_NAME}.npz", **tiny_split())
    return {
        "root": tmp_path / "root",
        "masks_dir": masks_dir,
        "dataset_path": str(dataset_path),
        "protocol": tiny_protocol(),
    }


def report():
    return SimpleNamespace(trained=0, skipped=0, pending=0)


def train_one(workspace, arm: str = "env", seed: int = 0, stage: str = "pilot",
              protocol=None, root=None):
    return train_and_store(
        arm,
        seed,
        stage,
        protocol or workspace["protocol"],
        MASK_NAME,
        workspace["masks_dir"],
        workspace["dataset_path"],
        root or workspace["root"],
        cpu_threads=1,
        report=report(),
    )


def manifest_path(workspace, arm: str = "env", seed: int = 0, protocol=None, root=None):
    proto = protocol or workspace["protocol"]
    return (root or workspace["root"]) / "runs" / (
        run_stem(arm, seed, MASK_NAME, proto["protocol_version"]) + ".json"
    )


# ------------------------------------------------------------------ helpers


def test_scenario_classification():
    assert scenario_of("e1_r20_seed42") == "E1"
    assert scenario_of("e2a_partial") == "E2a"
    assert scenario_of("e2b_partial") == "E2b"
    assert scenario_of("e3_internal_seed42") == "E3"
    with pytest.raises(ValueError):
        scenario_of("nonsense")


def test_task_grid_is_the_frozen_pilot_budget():
    protocol = load_protocol()
    grid = task_grid("pilot", protocol)
    assert len(grid) == 27
    assert len(set(grid)) == 27
    assert {arm for arm, _, _ in grid} == {"env", "h2x", "h3a"}
    assert {seed for _, seed, _ in grid} == {0, 1, 2}
    assert {mask for _, _, mask in grid} == {
        "e1_r20_seed42",
        "e2b_partial",
        "e3_internal_seed42",
    }
    assert len(task_grid("smoke", protocol)) == 3
    assert len(task_grid("expand", protocol)) == 8 * 5 * 3
    assert len(task_grid("pilot", protocol, masks=["e2b_partial"], seeds=[0])) == 3


def test_tensor_digest_tracks_content():
    a = tiny_dataset()
    b = tiny_dataset()
    assert tensor_digest(a) == tensor_digest(b)
    c = tiny_dataset()
    c["y"] = c["y"] + 1.0
    assert tensor_digest(c) != tensor_digest(a)


def test_identity_moves_with_every_decisive_input(workspace):
    protocol = workspace["protocol"]
    scope = dataset_scope(tiny_dataset(), workspace["dataset_path"], "pilot")
    base, base_hash = identity("env", 0, MASK_NAME, protocol, scope,
                               workspace["masks_dir"])
    variants = {
        "arm": identity("h2x", 0, MASK_NAME, protocol, scope, workspace["masks_dir"]),
        "seed": identity("env", 1, MASK_NAME, protocol, scope, workspace["masks_dir"]),
    }
    other_protocol = tiny_protocol(hidden=32)
    variants["hidden"] = identity("env", 0, MASK_NAME, other_protocol, scope,
                                  workspace["masks_dir"])
    smoke_scope = dataset_scope(tiny_dataset(), workspace["dataset_path"], "smoke")
    variants["stage"] = identity("env", 0, MASK_NAME, protocol, smoke_scope,
                                 workspace["masks_dir"])
    for label, (_, digest) in variants.items():
        assert digest != base_hash, label
    assert payload_hash(base) == base_hash
    # the same inputs always produce the same identity
    again, again_hash = identity("env", 0, MASK_NAME, protocol, scope,
                                 workspace["masks_dir"])
    assert again_hash == base_hash and again == base


# --------------------------------------------------------------- the run


def test_train_and_store_roundtrip(workspace):
    record = train_one(workspace)
    assert record["status"] == "complete"
    assert record["test_metrics"] is None
    assert set(record["artifacts"]) == set(ARTIFACT_KEYS)
    for item in record["artifacts"].values():
        assert (workspace["root"] / item["path"]).is_file()
    assert not list((workspace["root"]).rglob("*.tmp"))

    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    audited = audit_run(
        workspace["root"],
        manifest_path(workspace),
        dataset,
        tiny_split(),
        record["config_hash"],
        expected_config=record["config"],
    )
    assert audited["config_hash"] == record["config_hash"]
    frame = pd.read_parquet(
        workspace["root"] / record["artifacts"]["validation"]["path"]
    )
    assert set(frame.role.unique()) == {"val"}
    assert np.array_equal(frame.cell.to_numpy(), tiny_split()["val"])
    # an ENV run has no graph branch at all
    assert np.isnan(frame.correction_log.to_numpy()).all()
    assert np.isfinite(frame.base_log.to_numpy()).all()


def test_second_call_reuses_the_verified_record(workspace):
    first = train_one(workspace)
    second = train_one(workspace)
    assert second["config_hash"] == first["config_hash"]
    assert second["completed_at"] == first["completed_at"]


def test_a_different_configuration_refuses_to_overwrite(workspace):
    train_one(workspace)
    with pytest.raises(ValueError, match="identity conflict"):
        train_one(workspace, protocol=tiny_protocol(hidden=32))


def test_audit_rejects_a_tampered_artifact(workspace):
    record = train_one(workspace)
    path = workspace["root"] / record["artifacts"]["epochs"]["path"]
    path.write_text(path.read_text(encoding="utf-8") + "{\"epoch\": 99}\n",
                    encoding="utf-8")
    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    with pytest.raises(ValueError, match="corrupt artifact|epoch log"):
        audit_run(workspace["root"], manifest_path(workspace), dataset,
                  tiny_split(), record["config_hash"])


def test_audit_rejects_a_missing_checkpoint(workspace):
    record = train_one(workspace)
    (workspace["root"] / record["artifacts"]["checkpoint"]["path"]).unlink()
    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    with pytest.raises(ValueError, match="corrupt artifact"):
        audit_run(workspace["root"], manifest_path(workspace), dataset,
                  tiny_split(), record["config_hash"])


def test_audit_rejects_a_record_that_claims_test_metrics(workspace):
    record = train_one(workspace)
    path = manifest_path(workspace)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["test_metrics"] = {"log_rmse": 0.0}
    path.write_text(json.dumps(stored), encoding="utf-8")
    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    with pytest.raises(ValueError, match="outer-test metric"):
        audit_run(workspace["root"], path, dataset, tiny_split(),
                  record["config_hash"])


def test_audit_rejects_altered_validation_labels(workspace):
    record = train_one(workspace)
    path = workspace["root"] / record["artifacts"]["validation"]["path"]
    frame = pd.read_parquet(path)
    frame["y_true"] = frame["y_true"] + 1.0
    frame.to_parquet(path, index=False)
    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    with pytest.raises(ValueError):
        audit_run(workspace["root"], manifest_path(workspace), dataset,
                  tiny_split(), record["config_hash"])


def test_audit_rejects_a_stale_config_hash(workspace):
    record = train_one(workspace)
    path = manifest_path(workspace)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["config"]["seed"] = 99
    path.write_text(json.dumps(stored), encoding="utf-8")
    dataset = torch.load(workspace["dataset_path"], weights_only=False)
    with pytest.raises(ValueError, match="configuration was altered"):
        audit_run(workspace["root"], path, dataset, tiny_split(),
                  record["config_hash"])


def test_smoke_stage_uses_a_station_subset_and_a_separate_identity(workspace):
    smoke = train_one(workspace, stage="smoke")
    assert smoke["config"]["dataset"]["subset_stations"] == min(
        SMOKE_STATIONS, 8
    )
    assert smoke["training"]["epochs"] <= 3
    pilot = train_one(workspace, stage="pilot")
    assert smoke["config_hash"] != pilot["config_hash"]


def test_h3a_record_keeps_its_components_additive(workspace):
    record = train_one(workspace, arm="h3a")
    frame = pd.read_parquet(
        workspace["root"] / record["artifacts"]["validation"]["path"]
    )
    np.testing.assert_array_equal(
        frame.base_log.to_numpy() + frame.correction_log.to_numpy(),
        frame.total_log.to_numpy(),
    )
    np.testing.assert_allclose(
        frame.y_pred.to_numpy(), np.expm1(frame.pred_log_clipped.to_numpy()),
        rtol=1e-12, atol=0,
    )
