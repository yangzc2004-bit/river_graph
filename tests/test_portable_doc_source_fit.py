"""Source-only deployment uses the frozen recipe and explicit station roles."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fit_portable_doc_sources_v1 import validate_source_roles
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_doc_portable_source_fit_v1 import deployment_split


def test_deployment_roles_use_station_membership_without_target_values():
    observed = np.ones((8, 12), bool)
    observed[1, 2] = False
    reference = {"val": np.array([72, 74, 84]), "test": np.arange(24, 36)}
    roles = deployment_split(observed, reference)
    np.testing.assert_array_equal(roles["val"], np.arange(72, 96))
    assert 24 in roles["train"]  # Previous internal test is source for external deployment.
    assert 14 not in roles["train"]
    assert not len(roles["test"]) and not len(roles["context"])
    with pytest.raises(ValueError, match="reference validation"):
        deployment_split(observed, {"val": np.array([14])})


def test_source_fit_rejects_external_test_roles_and_shared_validation_stations():
    dataset = {"y": np.ones((8, 12)), "y_mask": np.ones((8, 12), bool)}
    roles = deployment_split(dataset["y_mask"], {"val": np.array([72])})
    validate_source_roles(dataset, roles)
    contaminated = copy.deepcopy(roles)
    contaminated["test"] = np.array([90])
    with pytest.raises(ValueError, match="test/context"):
        validate_source_roles(dataset, contaminated)
    contaminated = copy.deepcopy(roles)
    contaminated["val"] = np.array([1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="disjoint"):
        validate_source_roles(dataset, contaminated)


def test_frozen_source_fit_exports_replayable_label_free_predictions(tmp_path):
    """Exercise the actual forest/OOF/GRU fit and new facade end to end."""
    torch.set_num_threads(2)
    rng = np.random.default_rng(52)
    n, months = 8, 12
    dataset = {"site_no": np.array([f"source-{i}" for i in range(n)]),
        "months": np.arange("2000-01", "2001-01", dtype="datetime64[M]"),
        "y": rng.uniform(1, 7, (n, months)), "y_mask": np.ones((n, months), bool),
        "x": rng.uniform(1, 20, (n, months, 2)).astype(np.float32),
        "x_mask": np.ones((n, months, 2), bool),
        "static": rng.uniform(0, 2, (n, 2)).astype(np.float32),
        "regime": rng.uniform(0, 1, (n, 13)).astype(np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_attr": np.empty((0, 6), dtype=np.float32)}
    split = deployment_split(dataset["y_mask"], {"val": np.array([72, 84])})
    dataset_path, mask_path, daily_path = (tmp_path/name for name in ("source.pt", "roles.npz", "daily.npz"))
    daily = np.zeros((n, months, 8), dtype=np.float32)
    torch.save(dataset, dataset_path)
    np.savez_compressed(mask_path, **split)
    np.savez_compressed(daily_path, full=daily)
    predictor = PortableDOCReconstructor.fit(dataset_path=dataset_path, mask_path=mask_path,
        daily_path=daily_path, workdir=tmp_path/"fit", seed=42, runtime_snapshot_hash="synthetic-test")
    inputs = {key: np.asarray(dataset[key])[6:] for key in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=dataset["months"], daily_features=daily[6:])
    table = pd.read_parquet(tmp_path/"fit/predictions.parquet")
    expected = table[table.model_name.eq("unmonitored_integrated")].y_pred.to_numpy()
    np.testing.assert_allclose(predictor.predict(inputs).ravel(), expected, atol=1e-6, rtol=1e-6)
    assert not np.intersect1d(predictor.source_bank["station"], inputs["site_no"]).size
    predictor.save(tmp_path/"export")
    loaded = PortableDOCReconstructor.load(tmp_path/"export")
    np.testing.assert_allclose(loaded.predict(inputs), predictor.predict(inputs), atol=1e-12, rtol=1e-12)
    # Resume verifies the completed stage and never refits it.
    completion = (tmp_path/"fit/complete.json").read_bytes()
    PortableDOCReconstructor.fit(dataset_path=dataset_path, mask_path=mask_path,
        daily_path=daily_path, workdir=tmp_path/"fit", seed=42, runtime_snapshot_hash="synthetic-test")
    assert completion == (tmp_path/"fit/complete.json").read_bytes()
    config = json.loads((tmp_path/"fit/config.json").read_text())
    assert config["evaluation_role"] == "source_validation_only"
