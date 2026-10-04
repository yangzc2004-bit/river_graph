"""Fresh-role nested confirmation keeps training and query truth separate."""
from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.models.nested_chemical_adapter import (
    NestedChemicalAdapter,
    NestedChemicalEpisode,
)


def modules():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return (importlib.import_module("run_doc_nested_confirmation_v1"),
            importlib.import_module("verify_doc_nested_confirmation_v1"))


def problem():
    driver, verifier = modules()
    n, months = 12, 10
    data = {"y": np.arange(n * months, dtype=float).reshape(n, months) / 20 + 1,
        "y_mask": np.ones((n, months), bool), "site_no": np.array([f"station{i}" for i in range(n)]),
        "months": np.array([f"2000-{i+1:02d}" for i in range(months)])}
    split, protocol = driver.freeze_split.__globals__["build_unified_spatial_split"](data["y_mask"], seed=342)
    cells = np.arange(n * months)
    full = pd.DataFrame({"cell": cells, "station": data["site_no"][cells // months],
        "month": data["months"][cells % months], "analyte": "doc", "visibility_role": "unobserved",
        "ecological_novelty": 0., "upstream_support": 0., "ph_available": True, "ec_available": True,
        "aux_available": True, "doc_observed": True, "context_pred": 2., "point_pred": 2.,
        "neural_chemistry_pred": 2., "tree_chemistry_pred": 2., "ecological_memory": 0.})

    def references(role):
        _, query = driver.support_query_cells(split, target_role=role, k=0, n_months=months)
        rows = []
        for model in driver.REFERENCES:
            for k in driver.KS:
                row = full.iloc[query].copy()
                row["model_name"], row["k"] = model, k
                row["y_pred"], row["base_pred"], row["candidate_y_pred"] = 2., 2., 2.
                row["adaptation_delta"], row["regional_gamma"] = 0., 0.
                row["support_count"], row["aux_fallback"] = k, False
                row["basis_name"], row["selected_basis"] = "legacy", "legacy"
                rows.append(row)
        return pd.concat(rows, ignore_index=True)

    return driver, verifier, data, split, protocol, full, references


def test_new_partitions_and_smoke_recipes_use_actual_retained_helper_settings():
    driver, verifier = modules()
    assert driver.PARTITIONS == verifier.SPLITS == (342, 343, 344)
    assert driver.SEEDS == verifier.SEEDS == (42, 43, 44)
    assert set(driver.PARTITIONS).isdisjoint((142, 143, 144, 242, 243, 244))
    assert len(driver.MODELS) == 6
    for smoke, trees, epochs in ((True, 20, 1), (False, 300, 20)):
        recipe = driver.retained_initial_recipe(smoke=smoke)
        assert recipe["initial"]["n_estimators"] == trees
        assert recipe["initial"]["max_epochs"] == epochs
        assert recipe["projector"]["epochs"] == (1 if smoke else 100)
        assert recipe["memory"]["epochs"] == (1 if smoke else 30)
        assert driver.retained_chemical_recipe(smoke=smoke)["epochs"] == (1 if smoke else 120)
    nested = driver.retained_nested_recipe(342)
    assert nested["fold_seed"] == 4442 and nested["shared_selection_k"] == [3, 5]
    assert nested["parent_parameters_reselected_by_nested_branch"] is False
    # The new runner calls the retained helpers directly, with its own run
    # directory. It does not load a fitted historical confirmation parent.
    from doc_chemistry_confirmation_chemical import fit_chemistry_and_calibration
    from doc_chemistry_confirmation_initial import fit_initial_and_basis
    from doc_chemistry_confirmation_native import fit_native_and_ecology
    assert driver.fit_initial_and_basis is fit_initial_and_basis
    assert driver.fit_native_and_ecology is fit_native_and_ecology
    assert driver.fit_chemistry_and_calibration is fit_chemistry_and_calibration


def test_label_views_open_only_val_or_reserved_target_support():
    driver, _, data, split, _, _, _ = problem()
    val = driver.label_view(data, split, role="val")
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(val)), split["val"])
    support, query = driver.support_query_cells(split, target_role="test", k=5, n_months=10)
    target = driver.label_view(data, split, role="test")
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(target)), support)
    assert np.isnan(target[query]).all()
    changed = {**data, "y": data["y"].copy()}
    hidden = np.setdiff1d(np.arange(data["y"].size), support)
    changed["y"].ravel()[hidden] = np.nan
    np.testing.assert_array_equal(driver.label_view(changed, split, role="test"), target)
    with pytest.raises(ValueError, match="val or test"):
        driver.label_view(data, split, role="train")


def test_new_split_freeze_is_value_blind_and_keeps_role_identity(tmp_path):
    driver, _, data, _, _, _, _ = problem()
    split, protocol, path = driver.freeze_split(tmp_path, data, 342)
    altered = {**data, "y": np.full_like(data["y"], np.nan)}
    repeated, other_protocol, other_path = driver.freeze_split(tmp_path, altered, 342)
    assert protocol == other_protocol and path == other_path
    for role in split:
        np.testing.assert_array_equal(split[role], repeated[role])
    damaged = {role: value.copy() for role, value in split.items()}
    damaged["test"][0] = damaged["train"][0]
    np.savez_compressed(path, **damaged)
    with pytest.raises(AssertionError):
        driver.freeze_split(tmp_path, data, 342)


def test_six_curve_product_check_precedes_hidden_truth_export():
    driver, _, data, split, _, full, references = problem()
    frame = references("test")
    additional = []
    for name in driver.NESTED.values():
        row = frame[frame.model_name.eq(driver.LEGACY)].copy()
        row["model_name"] = name
        additional.append(row)
    frame = pd.concat([frame, *additional], ignore_index=True)
    _, query = driver.support_query_cells(split, target_role="test", k=0, n_months=10)
    np.testing.assert_array_equal(driver.validate_products(full, frame, data, split), query)
    with pytest.raises(ValueError, match="without query truth"):
        driver.validate_products(full, frame.assign(y_true=1), data, split)
    with pytest.raises(ValueError, match="Incomplete or repeated"):
        driver.validate_products(full, pd.concat([frame, frame.iloc[:1]]), data, split)
    bad = frame.copy()
    bad.loc[0, "cell"] = split["train"][0]
    with pytest.raises(AssertionError):
        driver.validate_products(full, bad, data, split)


def test_target_episode_constructor_never_opens_query_truth(monkeypatch):
    driver, _, data, split, _, full, references = problem()
    cells, months = len(full), 10
    labels = driver.label_view(data, split, role="test")
    calls = []

    def completed(*args, **kwargs):
        supplied_labels, support, query = args[3:6]
        assert np.isnan(supplied_labels[query]).all()
        assert np.isfinite(supplied_labels[support]).all()
        calls.append(kwargs["k"])
        return np.full(len(query), 2.), np.full(len(support), 2.)

    monkeypatch.setattr(driver, "completed_support_predictions", completed)
    coordinates = np.column_stack((np.arange(cells) % months, np.arange(cells) % 3))
    episodes = driver.role_episodes(full, references("test"), labels, split, months,
        np.zeros((cells, 2)), {}, coordinates, np.ones(cells, bool), role="test")
    assert calls == [0, 1, 3, 5]
    for episode in episodes.values():
        assert np.isnan(episode.query_values).all()
        assert np.isfinite(episode.support_values).all()


def test_fresh_nested_stage_fits_source_only_then_returns_truth_free_target(monkeypatch, tmp_path):
    driver, _, data, split, protocol, full, references = problem()
    cells, months = len(full), 10
    source_ids = np.asarray(protocol["station_roles"]["train"])
    chemical = np.column_stack((np.arange(cells) % months, np.arange(cells) % 3)).astype(float)
    shapes = {"legacy": np.zeros((cells, 2)), "masks_aug": np.zeros((cells, 4)), "chemistry_aug": np.zeros((cells, 4))}
    active = np.ones(cells, bool)
    monkeypatch.setattr(driver, "fresh_state", lambda run: (shapes,
        {mode: chemical for mode in driver.NESTED}, active, source_ids, {}))
    monkeypatch.setattr(driver, "reference_validation", lambda *args: references("val"))
    monkeypatch.setattr(driver, "completed_support_predictions", lambda *args, **kwargs:
                        (np.full(len(args[5]), 2.), np.full(len(args[4]), 2.)))
    result = {"full_grid": full, "predictions": references("test")}
    target, validation, cv = driver.fit_nested_stage(tmp_path, data, split, 342, result)
    assert set(target.model_name) == set(driver.MODELS)
    assert set(validation.model_name) == set(cv.model_name) == set(driver.MODELS)
    assert "y_true" not in target and "y_true" not in validation and "y_true" not in cv
    assert (cv[cv.model_name.isin(driver.NESTED.values()) & cv.k.ge(3)].conditional_fold >= 0).all()
    for mode in driver.NESTED:
        state = driver.json.loads((tmp_path / f"nested_{mode}.json").read_text())
        source = chemical.reshape(-1, months, 2)[source_ids]
        np.testing.assert_array_equal(state["coordinate_mean"], source.mean((0, 1)))
        np.testing.assert_array_equal(state["coordinate_std"], source.reshape(-1, 2).std(0))
        assert state["selection_role"] == "source_validation"
    # Changing target query DOC changes neither fitted state nor predictions.
    _, query = driver.support_query_cells(split, target_role="test", k=0, n_months=months)
    alternate = {**data, "y": data["y"].copy()}
    alternate["y"].ravel()[query] = 10000
    changed = driver.fit_nested_stage(tmp_path, alternate, split, 342, result)
    pd.testing.assert_frame_equal(changed[0], target)


def test_independent_saved_legacy_support_calculation_matches_existing_math():
    _, verifier = modules()
    from run_doc_chemical_kernel_v1 import linear_preclip
    full = pd.DataFrame({"context_pred": np.arange(10) / 5 + 2,
        "neural_chemistry_pred": np.arange(10) / 6 + 3, "ecological_memory": np.full(10, .5)})
    legacy = np.column_stack((np.linspace(-1, 1, 10), np.cos(np.arange(10))))
    labels = np.full(10, np.nan)
    support, query = np.arange(5), np.arange(5, 10)
    labels[support] = [1., 3., 5., 3., 6.]
    choice = {"gamma": .5, "alpha": .5, "ridge_strength": 1.}
    state = {"selection_by_k": {"5": choice}}
    predicted = verifier._completed_legacy_support(full, legacy, state, labels, support, query, months=10, k=5)
    base = verifier._manual_base(full.context_pred, full.neural_chemistry_pred, full.ecological_memory, .5)
    qz, sz, _ = linear_preclip(base, legacy, labels, support, query, months=10, k=5, alpha=.5, ridge_strength=1.)
    np.testing.assert_array_equal(predicted[0], np.maximum(0., np.expm1(qz)))
    np.testing.assert_array_equal(predicted[1], np.maximum(0., np.expm1(sz)))
    altered = labels.copy()
    altered[query] = 9999
    changed = verifier._completed_legacy_support(full, legacy, state, altered, support, query, months=10, k=5)
    for expected, actual in zip(predicted, changed, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_independent_nested_saved_prediction_formula_and_zero_fallback():
    _, verifier = modules()
    source = np.array([[-1., -1.], [0., 1.], [1., 0.], [2., 1.]])
    tasks = []
    for k in (3, 5):
        x = np.linspace(-1, 1, k)
        task = NestedChemicalEpisode(k, np.array([7, 8]), np.array([3., 6.]), np.full(2, 3.),
            np.arange(k), np.expm1(np.log(4) + .4 * x), np.full(k, 3.),
            np.array([[.5, 0.], [1.5, 1.]]), np.column_stack((x, x*x)),
            np.array([False, True]), np.ones(k, bool))
        tasks.append(task)
    model = NestedChemicalAdapter(10).fit_coordinates(source, np.ones(4, bool),
        source_role="source_training").fit(tasks, selection_role="source_validation")
    for task in tasks:
        for strength in (0., .5, 1.):
            choice = {"ridge_strength": .1, "strength": strength}
            expected = verifier._manual_nested(model, task, choice)
            actual = model.adapt_components(task.query_prediction, task.query_cells,
                task.support_prediction, task.support_cells, task.support_values,
                query_chemical=task.query_chemical, support_chemical=task.support_chemical,
                query_active=task.query_active, support_active=task.support_active, k=task.k, **choice)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            assert actual["y_pred"][0] == task.query_prediction[0]
    for k in (0, 1):
        task = replace(tasks[0], k=k, support_cells=np.arange(k), support_values=np.ones(k),
            support_prediction=np.ones(k), support_chemical=np.ones((k, 2)), support_active=np.ones(k, bool))
        expected = verifier._manual_nested(model, task, model.selection_)
        np.testing.assert_array_equal(expected["y_pred"], task.query_prediction)
        np.testing.assert_array_equal(expected["chemical_support_count"], [k, k])


def test_invalid_main_worker_budget_is_rejected_before_training(monkeypatch):
    driver, _ = modules()
    monkeypatch.setattr(sys, "argv", ["run_doc_nested_confirmation_v1.py", "--n-jobs", "0"])
    with pytest.raises(ValueError, match="Worker counts"):
        driver.main()
    args = Namespace(root=driver.ROOT, split_seeds=[242], seeds=[42], n_jobs=1, torch_threads=2, smoke=False)
    # CLI validates fresh roles before it reads datasets or starts any fitting.
    monkeypatch.setattr(sys, "argv", ["run_doc_nested_confirmation_v1.py", "--split-seeds", str(args.split_seeds[0])])
    with pytest.raises(ValueError, match="fresh written"):
        driver.main()
