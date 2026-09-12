"""R1 regression tests: the save/rebuild cache loop must never fake success.

Everything here uses a counting mock model factory, so no real training is
ever executed. The tests assert both the reported state and the number of
times a model was constructed.
"""

from __future__ import annotations

# Import the runner as a module (scripts/ is not a package).
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.experiments.predictions import (
    CacheState,
    PredictionConflictError,
    cache_state,
    read_meta,
    save_predictions,
)
from river_graph.experiments.provenance import build_meta, config_hash

_SPEC = importlib.util.spec_from_file_location(
    "run_gnn_under_test", Path(__file__).resolve().parents[1] / "scripts" / "run_gnn.py"
)
run_gnn = importlib.util.module_from_spec(_SPEC)
sys.modules["run_gnn_under_test"] = run_gnn
_SPEC.loader.exec_module(run_gnn)


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------

N, T = 5, 8


def _dataset() -> dict:
    ymask = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        for j in range(T):
            if (i + j) % 2 == 0:
                ymask[i, j] = True
    y = torch.zeros(N, T, dtype=torch.float32)
    y[ymask] = torch.arange(int(ymask.sum()), dtype=torch.float32) * 0.5 + 1.0
    return {
        "site_no": [f"S{i}" for i in range(N)],
        "months": [f"2020-{j + 1:02d}-01" for j in range(T)],
        "y": y,
        "y_mask": ymask,
    }


class CountingModel:
    """Records every construction; predicts a constant in log-space terms."""

    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.kwargs = kwargs

    def fit_predict(self, dataset, split):
        y = dataset["y"].numpy()
        return np.where(dataset["y_mask"].numpy(), y + 0.25, np.nan)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolated dataset / mask / results / predictions directories."""
    ds_dir = tmp_path / "data"
    ds_dir.mkdir()
    ds_path = ds_dir / "synthetic.pt"
    torch.save(_dataset(), ds_path)

    masks = tmp_path / "masks"
    masks.mkdir()
    flat = np.flatnonzero(_dataset()["y_mask"].numpy().ravel())
    np.savez(masks / "m1.npz", train=flat[:12], val=flat[12:14], test=flat[14:])
    np.savez(masks / "m2.npz", train=flat[:10], val=flat[10:12], test=flat[12:])

    results = tmp_path / "results"
    results.mkdir()
    preds = tmp_path / "preds"
    preds.mkdir()

    monkeypatch.setattr(run_gnn, "load_dataset", lambda _p: _dataset())
    CountingModel.instances = 0
    return {
        "dataset": str(ds_path), "masks": masks, "results": results,
        "preds": preds, "tmp": tmp_path,
    }


def _args(sandbox, **over):
    class A:
        pass

    a = A()
    a.only = None
    a.variants = ["river"]
    a.lr = 1e-3
    a.arch = "gcn"
    a.dataset = sandbox["dataset"]
    a.tag = "T"
    a.model_name = None
    a.share_weights = False
    a.edge_dropout = 0.0
    a.wd = 0.0
    a.seed = 0
    a.save_predictions = False
    a.rebuild_predictions = False
    a.force = False
    a.repair_metrics = False
    a.env_groups = None
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _run(sandbox, args):
    return run_gnn.run(args, model_factory=CountingModel,
                       results_dir=sandbox["results"], masks_dir=sandbox["masks"],
                       pred_dir=sandbox["preds"])


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_fresh_run_trains_once_per_mask_and_saves(sandbox):
    args = _args(sandbox, save_predictions=True)
    report = _run(sandbox, args)

    assert report.trained == 2, "one training per mask"
    assert CountingModel.instances == 1, "model is built once, not per mask"
    assert report.missing_predictions == []
    assert report.identity_mismatch == []
    assert not report.failed
    for mask in ("m1", "m2"):
        assert (sandbox["preds"] / f"T_river__{mask}.parquet").exists()
        assert (sandbox["preds"] / f"T_river__{mask}.meta.json").exists()


def test_complete_cache_does_not_train(sandbox):
    _run(sandbox, _args(sandbox, save_predictions=True))
    CountingModel.instances = 0

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.trained == 0
    assert CountingModel.instances == 0, "cached run must not touch the model"
    assert report.skipped == 2
    assert report.missing_predictions == []


def test_metrics_only_reports_missing_without_training(sandbox):
    """The R1 trap: metrics cached, predictions absent, --save-predictions on."""
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 1.0, "r2": 0.5, "n": 7}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.missing_predictions == ["T_river__m1"]
    assert not (sandbox["preds"] / "T_river__m1.parquet").exists()
    # m1 (metrics cached, predictions missing) must NOT be retrained; m2 has
    # neither metrics nor predictions, so it is a normal fresh run.
    assert report.trained == 1
    assert CountingModel.instances == 1


def test_metrics_only_with_rebuild_trains_and_stores_both(sandbox):
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 1.0, "r2": 0.5, "n": 7}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox, save_predictions=True,
                                 rebuild_predictions=True))

    assert report.missing_predictions == []
    assert report.trained == 2, "m1 rebuilt, m2 fresh"
    assert (sandbox["preds"] / "T_river__m1.parquet").exists()
    stored = json.loads((sandbox["results"] / "T_river.json").read_text())
    meta = read_meta("T_river", "m1", sandbox["preds"])
    assert stored["m1"]["n"] == meta["split_sizes"]["test"]
    assert stored["m1"]["mae"] != 1.0, "metric now comes from the same run"


def test_metrics_only_without_save_flag_skips_quietly(sandbox):
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 1.0, "r2": 0.5, "n": 7}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox))

    assert report.missing_predictions == []
    assert report.trained == 1  # only m2
    assert CountingModel.instances == 1


def test_predictions_only_rebuilds_metrics_without_training(sandbox):
    """Metrics JSON emptied but predictions intact: rebuild, do not retrain.

    The sidecar must be rebuilt too, because emptying the JSON loses the record
    of the stored metrics and the identity check would otherwise (correctly)
    refuse to treat the pair as trustworthy.
    """
    _run(sandbox, _args(sandbox, save_predictions=True))
    for mask in ("m1", "m2"):
        meta = read_meta("T_river", mask, sandbox["preds"])
        meta["metrics"] = None
        (sandbox["preds"] / f"T_river__{mask}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
    (sandbox["results"] / "T_river.json").write_text("{}", encoding="utf-8")
    CountingModel.instances = 0

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.trained == 0
    assert CountingModel.instances == 0
    assert report.rebuilt_from_predictions == 2
    stored = json.loads((sandbox["results"] / "T_river.json").read_text())
    assert set(stored) == {"m1", "m2"}
    assert all(np.isfinite(stored[k]["mae"]) for k in stored)
    assert all(stored[k]["n"] > 0 for k in stored)


def test_identity_mismatch_is_not_reported_as_success(sandbox):
    """A stored prediction from another configuration must not be reused."""
    ds = _dataset()
    split = {"train": np.array([0, 1]), "val": np.array([2]),
             "test": np.array([3, 4, 5])}
    other = build_meta(model_name="T_river", mask_name="m1",
                       dataset_path=sandbox["dataset"], split=split,
                       params={"seed": 99, "architecture": "gcn"})
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=other)
    (sandbox["results"] / "T_river.json").write_text(
        json.dumps({"m1": {"mae": 9.0, "r2": -1.0, "n": 3}}), encoding="utf-8")

    report = _run(sandbox, _args(sandbox, save_predictions=True))

    assert report.identity_mismatch == ["T_river__m1"]
    assert report.trained == 1, "only m2 runs; m1 is refused"
    assert CountingModel.instances == 1


def test_force_overrides_identity_mismatch(sandbox):
    ds = _dataset()
    split = {"train": np.array([0, 1]), "val": np.array([2]),
             "test": np.array([3, 4, 5])}
    other = build_meta(model_name="T_river", mask_name="m1",
                       dataset_path=sandbox["dataset"], split=split,
                       params={"seed": 99, "architecture": "gcn"})
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=other)

    report = _run(sandbox, _args(sandbox, save_predictions=True, force=True))

    assert report.identity_mismatch == []
    assert report.trained == 2
    meta = read_meta("T_river", "m1", sandbox["preds"])
    assert meta["config"]["seed"] == 0


def test_interrupted_write_is_not_a_complete_cache(sandbox):
    """A leftover .tmp file must not be mistaken for a finished prediction."""
    (sandbox["preds"] / "T_river__m1.parquet.tmp").write_bytes(b"partial")

    assert cache_state("T_river", "m1", sandbox["results"], sandbox["preds"],
                       metrics={"m1": {}}) is CacheState.METRICS_ONLY

    report = _run(sandbox, _args(sandbox, save_predictions=True,
                                 rebuild_predictions=True))
    assert "T_river__m1" not in report.missing_predictions
    assert (sandbox["preds"] / "T_river__m1.parquet").exists()


def test_save_is_atomic_and_leaves_no_temp_files(sandbox):
    ds = _dataset()
    split = {"train": np.array([0, 1]), "val": np.array([2]),
             "test": np.array([3, 4, 5])}
    meta = build_meta(model_name="T_river", mask_name="m1",
                      dataset_path=sandbox["dataset"], split=split,
                      params={"seed": 0}, masks_dir=sandbox["masks"])
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=meta)

    assert not list(sandbox["preds"].glob("*.tmp"))
    stored = read_meta("T_river", "m1", sandbox["preds"])
    # the hash is computed over the STORED config, which now includes the
    # dataset and mask content hashes
    assert stored["config_hash"] == config_hash(stored["config"])
    assert stored["config"]["seed"] == 0
    assert stored["config"]["dataset_sha256"]
    assert stored["config"]["mask_sha256"]
    # and an identity-free dict must NOT reproduce it
    assert stored["config_hash"] != config_hash({"seed": 0,
                                                 "model_name": "T_river"})


def test_save_predictions_refuses_conflicting_config(sandbox):
    ds = _dataset()
    split = {"train": np.array([0, 1]), "val": np.array([2]),
             "test": np.array([3, 4, 5])}
    first = build_meta(model_name="T_river", mask_name="m1",
                       dataset_path=sandbox["dataset"], split=split,
                       params={"seed": 0})
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=first)

    second = build_meta(model_name="T_river", mask_name="m1",
                        dataset_path=sandbox["dataset"], split=split,
                        params={"seed": 1})
    with pytest.raises(PredictionConflictError):
        save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1",
                         "synthetic", out_dir=sandbox["preds"], meta=second)

    # the deliberate override still works
    save_predictions(np.zeros((N, T)), ds, split, "T_river", "m1", "synthetic",
                     out_dir=sandbox["preds"], meta=second, force=True)
    assert read_meta("T_river", "m1", sandbox["preds"])["config_hash"] == \
        second["config_hash"]


def test_cache_state_transitions(sandbox):
    results, preds = sandbox["results"], sandbox["preds"]
    assert cache_state("T_river", "m1", results, preds, metrics={}) is CacheState.ABSENT
    assert cache_state("T_river", "m1", results, preds,
                       metrics={"m1": {}}) is CacheState.METRICS_ONLY
    (preds / "T_river__m1.parquet").write_bytes(b"x")
    assert cache_state("T_river", "m1", results, preds,
                       metrics={"m1": {}}) is CacheState.COMPLETE
    assert cache_state("T_river", "m1", results, preds,
                       metrics={}) is CacheState.PREDICTIONS_ONLY
