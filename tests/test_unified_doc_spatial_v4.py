"""Role isolation for recurrent few-shot adaptation and its fixed readout."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("unified_v4_runner", scripts / "run_unified_doc_spatial_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_labels_cannot_change_recurrent_fit():
    rng = np.random.default_rng(472)
    n, months, hidden = 8, 20, 4
    source_ids, val_ids = np.array([0, 2, 4]), np.array([1, 3, 5])
    cells = np.arange(n * months).reshape(n, months)
    split = {"train": cells[source_ids].ravel(), "val": cells[val_ids].ravel(),
             "test": cells[[6, 7]].ravel(), "context": np.array([], dtype=np.int64)}
    features = {"source_station_ids": source_ids}
    for key, value in {
        "encoded": rng.normal(size=(n, months, hidden)).astype(np.float32),
        "age": rng.uniform(0, 3, size=(n, months)).astype(np.float32),
        "support": rng.uniform(size=(n, months, 3)).astype(np.float32),
    }.items():
        features[f"full_{key}"] = value
        features[f"source_{key}"] = value[source_ids].copy()
    truth = 2 + rng.random((n, months))
    base = np.full(truth.size, 2.)
    oof = np.full_like(truth, np.nan)
    oof[source_ids] = np.log1p(2.)
    torch.manual_seed(2)
    expert = SimpleNamespace(temporal=torch.nn.GRUCell(hidden + 1, hidden), decay=torch.nn.Linear(4, hidden))
    readout = rng.normal(size=(hidden, 2))
    run = _runner()
    first = run.fit_memory(expert, features, oof, truth, split, base, readout,
                           seed=42, epochs=2, patience=2)
    changed = truth.copy()
    changed.ravel()[split["test"]] += 100000
    second = run.fit_memory(expert, features, oof, changed, split, base, readout,
                            seed=42, epochs=2, patience=2)
    assert first.to_dict() == second.to_dict()
    full = {key: features[f"full_{key}"] for key in ("encoded", "age", "support")}
    np.testing.assert_array_equal(first.transform(full), second.transform(full))


def test_fixed_readout_matches_v3_projection_up_to_station_offset():
    from river_graph.models.episodic_station_adapter import EpisodicStationProjector

    rng = np.random.default_rng(326)
    features = rng.normal(size=(3, 16, 18)).astype(np.float32)
    truth = rng.uniform(1, 5, size=(3, 16))
    base = np.full((3, 16), 2.)
    mask = np.ones((3, 16), dtype=bool)
    projector = EpisodicStationProjector(epochs=0).fit(
        features, np.log1p(base), truth, mask, features, base, truth, mask,
        selection_role="source_validation")
    matrix = _runner().readout_matrix(projector.to_dict())
    raw = features.astype(np.float64) @ matrix
    actual = raw - raw.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, projector.transform(features), rtol=1e-12, atol=1e-12)
