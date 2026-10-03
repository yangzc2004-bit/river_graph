"""Source/target separation in the complete episodic projection workflow."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("unified_v3_runner", scripts / "run_unified_doc_spatial_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_labels_cannot_change_projector_training_or_source_whitening():
    rng = np.random.default_rng(212)
    n, months = 8, 18
    train_stations, val_stations = np.array([0, 1, 2]), np.array([3, 4, 5])
    all_cells = np.arange(n * months).reshape(n, months)
    split = {"train": all_cells[train_stations].ravel(), "val": all_cells[val_stations].ravel(),
             "test": all_cells[[6, 7]].ravel(), "context": np.array([], dtype=np.int64)}
    features = {"source_station_ids": train_stations}
    for name, dim in (("gru", 18), ("tree", 20)):
        features[f"full_{name}"] = rng.normal(size=(n, months, dim)).astype(np.float32)
        features[f"source_{name}"] = features[f"full_{name}"][train_stations].copy()
    truth = 2 + rng.random((n, months))
    context = np.full(n * months, 2.)
    oof = np.full((n, months), np.nan)
    oof[train_stations] = np.log1p(2.)
    run = _runner()
    first, shapes = run.fit_projectors(features, oof, truth, split, context,
                                      seed=42, epochs=1, patience=1)
    changed = truth.copy()
    changed.ravel()[split["test"]] += 100000
    second, shapes2 = run.fit_projectors(features, oof, changed, split, context,
                                        seed=42, epochs=1, patience=1)
    for name in first:
        assert first[name].to_dict() == second[name].to_dict()
    for name in shapes:
        np.testing.assert_array_equal(shapes[name], shapes2[name])
    assert set(shapes) == set(run.HEADS)
    assert set(run.MODELS) == {f"{base}_{head}" for base in ("context", "fusion") for head in shapes}
