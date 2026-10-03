"""Tests for fixed observation-availability routing of native base predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from river_graph.models.daily_flow_features import build_daily_flow_features
from river_graph.models.daily_hydro_router import DailyHydroRouter


def test_any_valid_descriptor_routes_daily_with_exact_absence_fallback():
    monthly = np.array([[1., 2., 3., 4.]])
    daily = np.array([[11., 12., 13., 14.]])
    features = np.zeros((1, 4, 8), dtype=np.float32)
    features[0, 1, 5] = 1
    features[0, 2, 6:] = 1
    features[0, 3, 5:] = 1
    features[0, 0, :5] = 1  # Numeric values/availability do not substitute for validity.
    result = DailyHydroRouter().predict_components(monthly, daily, features)
    np.testing.assert_array_equal(result["routed_pred"], [[1., 12., 13., 14.]])
    np.testing.assert_array_equal(result["uses_daily"], [[False, True, True, True]])
    np.testing.assert_array_equal(result["daily_numeric_valid_count"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(result["monthly_pred"], monthly)
    np.testing.assert_array_equal(result["daily_pred"], daily)
    assert not np.shares_memory(result["monthly_pred"], monthly)


def test_valid_zero_flow_uses_daily_and_absent_monthly_flow_uses_monthly():
    dataset = {"site_no": ["a"], "months": ["2024-02", "2024-03"],
               "x_mask": np.ones((1, 2, 2), dtype=bool)}
    dataset["x_mask"][0, 1, 1] = False
    daily = pd.DataFrame({"site_no": "a", "date": pd.date_range("2024-02-01", "2024-03-31"),
                         "discharge_cfs": 0.})
    features = build_daily_flow_features(dataset, daily)["full"]
    np.testing.assert_array_equal(features[..., :3], 0)
    result = DailyHydroRouter().predict([[1., 2.]], [[3., 4.]], features)
    np.testing.assert_array_equal(result, [[3., 2.]])


def test_route_ignores_numeric_magnitudes_and_other_cells():
    features = np.zeros((3, 8))
    features[0, 5:] = 1
    router = DailyHydroRouter()
    first = router.predict_components([1, 2, 3], [4, 5, 6], features)
    features[:, :5] = [[.3, .7, .9, 1, 1], [.9, .5, .1, .7, .5], [.1, .2, .3, .4, .5]]
    second = router.predict_components([1, 2, 30], [4, 5, 60], features)
    np.testing.assert_array_equal(first["uses_daily"], second["uses_daily"])
    np.testing.assert_array_equal(first["routed_pred"][:2], second["routed_pred"][:2])


def test_serialization_preserves_fixed_rule_and_rejects_changed_semantics():
    router = DailyHydroRouter()
    payload = json.loads(json.dumps(router.to_dict()))
    restored = DailyHydroRouter.from_dict(payload)
    features = np.zeros((2, 8)); features[1, 5] = 1
    np.testing.assert_array_equal(router.predict([1, 2], [3, 4], features),
                                  restored.predict([1, 2], [3, 4], features))
    payload["rule"] = "all_numeric_descriptors_required"
    with pytest.raises(ValueError, match="fixed supported rule"):
        DailyHydroRouter.from_dict(payload)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(bad):
    router = DailyHydroRouter()
    with pytest.raises(ValueError, match="finite"):
        router.predict([bad], [1.], np.zeros((1, 8)))
    with pytest.raises(ValueError, match="finite"):
        router.predict([1.], [bad], np.zeros((1, 8)))
    features = np.zeros((1, 8)); features[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        router.predict([1.], [2.], features)


def test_shape_and_validity_requirements():
    router = DailyHydroRouter()
    with pytest.raises(ValueError, match="same nonempty shape"):
        router.predict([1], [1, 2], np.zeros((1, 8)))
    with pytest.raises(ValueError, match="same nonempty shape"):
        router.predict([], [], np.zeros((0, 8)))
    with pytest.raises(ValueError, match="eight channels"):
        router.predict([1], [2], np.zeros((1, 7)))
    features = np.zeros((1, 8)); features[0, 5] = .5
    with pytest.raises(ValueError, match="zero or one"):
        router.predict([1], [2], features)
    with pytest.raises(ValueError, match="real numeric"):
        router.predict(["1"], [2], np.zeros((1, 8)))
