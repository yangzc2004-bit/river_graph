"""Independent examples for metric semantics, matched inputs and frozen scope."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyRegressor

from river_graph.experiments.doc_fusion_benchmark import (
    CANDIDATES,
    concentration_prediction,
    metrics,
    paired_basin_interval,
    seasonal_medians,
    select_recipe,
    structure_metrics,
    tabular_features,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import covariate_inputs


def test_absolute_log_squared_and_variance_metrics_have_different_meaning():
    frame = pd.DataFrame({"y_true": [0., 3.], "y_pred": [1., 1.],
                          "station": ["a", "b"], "high_doc": [False, True]})
    result = metrics(frame)
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["log1p_mae"] == pytest.approx(np.log(2))
    assert result["r2"] == pytest.approx(1 - 5 / 4.5)
    assert result["source_q90_mae"] == 2.
    assert result["bias"] == -.5


def test_zero_variance_empty_tail_and_nonfinite_are_not_silent_zero():
    frame = pd.DataFrame({"y_true": [2., 2.], "y_pred": [1., 3.],
                          "station": ["a", "a"], "high_doc": [False, False]})
    assert metrics(frame)["r2"] is None
    assert metrics(frame)["source_q90_mae"] is None
    frame.loc[0, "y_pred"] = np.nan
    with pytest.raises(ValueError, match="complete finite"):
        metrics(frame)


def test_common_postprocessing_clamps_and_counts_both_transforms():
    predicted, clips = concentration_prediction([-2., 1., 16.], "log1p")
    assert predicted == pytest.approx([0., np.expm1(1), np.expm1(15)])
    assert clips == {"negative_raw_native": 1, "upper_clipped": 1}
    native, counts = concentration_prediction([-1., 2., np.expm1(16)], "native")
    assert native == pytest.approx([0., 2., np.expm1(15)])
    assert counts == {"negative_raw_native": 1, "upper_clipped": 1}


def test_source_seasonal_baseline_has_a_declared_missing_month_fallback():
    result = seasonal_medians([1., 3., 10.], [1, 1, 2])
    assert result["1"] == 2. and result["2"] == 10. and result["3"] == 3.


def test_selection_uses_native_mae_and_declared_exact_tie_break():
    scores = {"a": [{"mae": 2., "log1p_mae": .1}], "b": [{"mae": 1., "log1p_mae": .3}]}
    assert select_recipe(scores) == "b"
    scores["a"][0]["mae"] = 1.
    assert select_recipe(scores) == "a"
    with pytest.raises(ValueError, match="finite candidate"):
        select_recipe({"a": [{"mae": np.nan, "log1p_mae": .1}]})


def test_flattened_history_keeps_order_masks_age_and_source_scaling():
    data = {"x": np.arange(2 * 14 * 2, dtype=float).reshape(2, 14, 2),
        "x_mask": np.ones((2, 14, 2), dtype=bool),
        "feature_channels": ["temperature", "discharge"],
        "months": [f"2020-{m:02d}" for m in range(1, 13)] + ["2021-01", "2021-02"],
        "regime": np.ones((2, 13)), "static": np.ones((2, 2)),
        "y": np.zeros((2, 14)), "y_mask": np.ones((2, 14), dtype=bool)}
    clean = {key: value for key, value in data.items() if key not in ("y", "y_mask")}
    inputs = covariate_inputs(clean, [0])
    frame = tabular_features(inputs, [0, 13, 14, 27], 14)
    assert frame.shape == (4, 113)
    np.testing.assert_array_equal(frame[1, 15:111], inputs["windows"][13, 0].numpy().ravel())
    # Later target observations cannot change an earlier target driver window.
    changed = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in clean.items()}
    changed["x"][1, 10:] = 100000.
    other = covariate_inputs(changed, [0])
    np.testing.assert_array_equal(tabular_features(inputs, [18], 14), tabular_features(other, [18], 14))
    assert inputs["normalization"] == other["normalization"]


def test_seed_repeats_do_not_create_more_bootstrap_basin_clusters():
    records = []
    for basin in range(5):
        for seed in (42, 43, 44):
            for model, prediction in (("base", 5.), ("fixed_fusion", 4.)):
                records.append({"target_huc4": str(basin), "cell": basin,
                    "seed": seed, "model_name": model, "y_true": 3., "y_pred": prediction})
    result = paired_basin_interval(pd.DataFrame(records), "base", "mae")
    assert result["basin_clusters"] == 5
    assert result["difference_interval"] == pytest.approx([-1., -1.])
    assert result["gain_percent_interval"] == pytest.approx([50., 50.])


def test_structure_only_uses_consecutive_observed_months_and_same_month_edges():
    frame = pd.DataFrame({"cell": [0, 1, 3, 6, 7], "station": ["a", "a", "a", "b", "b"],
        "month": ["2020-01", "2020-02", "2020-04", "2020-01", "2020-02"],
        "y_true": [1., 2., 5., 3., 6.], "y_pred": [1., 3., 4., 4., 5.]})
    result = structure_metrics(frame, np.array([[0], [1]]), 6)
    assert result["consecutive_month_pairs"] == 2
    assert result["consecutive_month_delta_mae"] == 1.5
    assert result["same_month_edge_pairs"] == 2
    assert result["same_month_edge_contrast_mae"] == 1.5
    assert result["seasonal_centered_mae"] is None


def runner_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("doc_fusion_benchmark_runner", scripts /
                                                     "run_doc_fusion_benchmark_development_v1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_target_label_poisoning_cannot_change_any_fit_or_source_selection(tmp_path, monkeypatch):
    runner = runner_module()
    months = 12
    split = {"train": np.array([0, 1, 2, 3]), "val": np.array([12, 13]), "test": np.array([24, 25])}
    inputs = {"environment": torch.zeros(3, 15), "windows": torch.zeros(12, 3, 12, 8),
              "season": torch.zeros(12, 2), "normalization": {"source_rows": [0]}}
    ctx = {"months": months, "split": split, "rows": {"train": np.array([0])}, "inputs": inputs}
    data = {"y": np.ones((3, months)), "site_no": ["a", "b", "c"],
            "months": [f"2020-{month:02d}" for month in range(1, 13)]}
    data["y"][0, :4] = [1., 2., 3., 4.]
    protocol = {"runtime_snapshot": {}, "dataset_sha256": "data", "mask_sha256": {"1013": "mask"}, "threads": 1}
    monkeypatch.setattr(runner, "context", lambda *_: ctx)
    monkeypatch.setattr(runner, "make_estimator", lambda *_: DummyRegressor())
    runner.fit_region(tmp_path / "clean", "1013", protocol, data)
    poisoned = {**data, "y": data["y"].copy()}
    poisoned["y"][2] = np.nan
    runner.fit_region(tmp_path / "poisoned", "1013", protocol, poisoned)
    paths = [tmp_path / name / "regions/huc4_1013" for name in ("clean", "poisoned")]
    selections = [json.loads((path / "selection.json").read_text())["selections"] for path in paths]
    assert selections[0] == selections[1]
    assert len(list(paths[0].glob("fits/*/*/seed*/fit_complete.json"))) == 21
    # Changed provenance is rejected even for an already completed selection.
    changed = {**protocol, "dataset_sha256": "different_data"}
    with pytest.raises(ValueError, match="config changed"):
        runner.fit_region(tmp_path / "clean", "1013", changed, data)
    # A changed fitted model cannot silently survive a resume.
    model = next(paths[0].glob("fits/*/*/seed*/model.joblib"))
    model.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed package artifact"):
        runner.fit_region(tmp_path / "clean", "1013", protocol, data)


DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
ROOT = Path("experiments/phase4_transfer/doc_fusion_benchmark_development_v1")


@pytest.mark.skipif(not DATA.exists() or not (ROOT / "protocol.json").exists(),
                    reason="local ST357 data or frozen matched-input benchmark unavailable")
def test_new_freeze_preserves_old_dispatcher_and_whole_basin_masks():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert not protocol["prior_paper_endpoints_modified"]
    assert protocol["joint_primary"] == ["native_mae", "log1p_mae"]
    assert protocol["candidates"] == CANDIDATES
    assert "previously inspected" in protocol["scope"]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_repository_dispatcher_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT / "code_snapshot" / name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
    months = torch.load(DATA, weights_only=False)["y"].shape[1]
    for region, expected in protocol["mask_sha256"].items():
        path = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks") / f"huc4_{region}.npz"
        assert sha256_file(path) == expected
        with np.load(path) as split:
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                assert not np.intersect1d(split[a] // months, split[b] // months).size
