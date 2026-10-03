"""Fold-fitted source context baselines for support/query episode training."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.models import episodic_station_data
from river_graph.models.episodic_station_data import fit_context_oof
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    fold_split,
    target_values,
)


@pytest.fixture
def source_case():
    rng = np.random.default_rng(87)
    n, months = 12, 14
    x = rng.uniform(0.1, 4, (n, months, 2)).astype(np.float32)
    data = {
        "y": (1 + x[..., 0] + rng.uniform(0, 2, (n, months))).astype(np.float32),
        "y_mask": np.ones((n, months), dtype=bool),
        "x": x, "x_mask": np.ones_like(x),
        "static": rng.uniform(0, 1, (n, 2)).astype(np.float32),
        "regime": rng.uniform(0, 1, (n, 13)).astype(np.float32),
        "months": np.arange("2020-01", "2021-03", dtype="datetime64[M]").astype(str).tolist(),
        "edge_index": np.stack([np.arange(n - 1), np.arange(1, n)]),
    }
    # An unobserved cell within a source station must remain NaN in OOF output.
    data["y_mask"][0, 2] = False
    split = {"train": np.delete(np.arange(10 * months), 2),
             "val": np.arange(10 * months, 11 * months),
             "test": np.arange(11 * months, 12 * months),
             "context": np.array([], dtype=np.int64)}
    x_context = build_rf_features(data, split, FIT_ROLES,
                                  target_transform="log1p", include_network=True)
    selected = ExtraTreesRegressor(
        n_estimators=4, min_samples_leaf=2, max_features=0.6, max_depth=4,
        random_state=29, n_jobs=1,
    ).fit(x_context[split["train"]], target_values(data, "log1p").ravel()[split["train"]])
    model = SimpleNamespace(
        context_forest=selected, context_name="selected_context",
        rf=SimpleNamespace(
            folds=[np.array([i, i + 5]) for i in range(5)],
            # The original RFArtifacts context uses different parameters.
            context=clone(selected).set_params(min_samples_leaf=4, max_features=1.0),
            context_oof_z=np.full((n, months), 999.0),
        ),
    )
    return model, data, split


def test_exactly_five_selected_context_clones_with_source_coverage(source_case, monkeypatch):
    model, data, split = source_case
    cloned = []

    def capture_clone(estimator):
        forest = clone(estimator)
        cloned.append(forest)
        return forest

    monkeypatch.setattr(episodic_station_data, "clone", capture_clone)
    progress = []
    result = fit_context_oof(model, data, split, n_jobs=1, progress=progress.append)
    assert len(cloned) == len(result["fold_records"]) == len(progress) == 5
    train = split["train"]
    pred = result["pred_z"]
    assert pred.shape == data["y"].shape
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(pred.ravel())), train)
    assert np.isnan(pred.ravel()[np.concatenate([split["val"], split["test"]])]).all()
    assert np.isnan(pred[0, 2])
    assert np.ptp(pred.ravel()[train]) > 0
    json.dumps(result["fold_records"], allow_nan=False)
    months = data["y"].shape[1]
    for forest, record, held in zip(cloned, result["fold_records"], model.rf.folds, strict=True):
        assert forest is not model.context_forest
        assert forest.get_params() == model.context_forest.get_params()
        assert record["n_features"] == 39
        assert record["context_name"] == "selected_context"
        assert set(record["held_station_ids"]).isdisjoint(record["training_station_ids"])
        assert set(record["held_station_ids"]) | set(record["training_station_ids"]) == set(range(10))
        assert record["n_train_cells"] + record["n_oof_cells"] == len(train)
        # Independently refit the selected configuration, rather than the
        # RFArtifacts leaf-4 context or its pre-existing OOF array.
        view = fold_split(split, held, months)
        x = build_rf_features(data, view, FIT_ROLES, target_transform="log1p", include_network=True)
        reference = clone(model.context_forest).fit(
            x[view["train"]], target_values(data, "log1p").ravel()[view["train"]],
        )
        held_cells = train[np.isin(train // months, held)]
        np.testing.assert_array_equal(pred.ravel()[held_cells], reference.predict(x[held_cells]))
    assert sum(record["n_oof_cells"] for record in result["fold_records"]) == len(train)


def test_validation_and_test_label_changes_do_not_change_oof(source_case):
    model, data, split = source_case
    expected = fit_context_oof(model, data, split, n_jobs=1)
    changed = copy.deepcopy(data)
    for role in ("val", "test"):
        changed["y"].ravel()[split[role]] += 10000
    actual = fit_context_oof(model, changed, split, n_jobs=1)
    np.testing.assert_array_equal(actual["pred_z"], expected["pred_z"])
    assert actual["fold_records"] == expected["fold_records"]


def test_held_station_labels_do_not_change_its_fold_inputs_or_predictions(source_case):
    model, data, split = source_case
    held = model.rf.folds[0]
    changed = copy.deepcopy(data)
    changed["y"][held] += 10000
    view = fold_split(split, held, data["y"].shape[1])
    before_x = build_rf_features(data, view, FIT_ROLES,
                                target_transform="log1p", include_network=True)
    after_x = build_rf_features(changed, view, FIT_ROLES,
                               target_transform="log1p", include_network=True)
    np.testing.assert_array_equal(after_x, before_x)
    before = fit_context_oof(model, data, split, n_jobs=1)["pred_z"]
    after = fit_context_oof(model, changed, split, n_jobs=1)["pred_z"]
    np.testing.assert_array_equal(after[held], before[held])


@pytest.mark.parametrize("defect", ["four_folds", "overlap", "missing", "non_source"])
def test_malformed_source_folds_rejected_before_fitting(source_case, defect):
    model, data, split = source_case
    if defect == "four_folds":
        model.rf.folds = model.rf.folds[:4]
    elif defect == "overlap":
        model.rf.folds[1][0] = model.rf.folds[0][0]
    elif defect == "missing":
        model.rf.folds[0] = model.rf.folds[0][:1]
    else:
        model.rf.folds[0][0] = 11
    with pytest.raises(ValueError, match="five|partition"):
        fit_context_oof(model, data, split, n_jobs=1)
