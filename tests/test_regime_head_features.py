"""Source-only scaling and input boundaries for the regime-dependent head."""

import copy
import inspect
import json

import numpy as np
import pytest

from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.regime_head_features import build_regime_head_features


def sample():
    regime = np.zeros((4, 13))
    regime[:, 4:] = np.array([2., 6., 10., 14.])[:, None]
    source = np.array([5, 1, 4])
    oof = np.array([7., 3., 1.])
    context = np.full((4, 4), 9.)
    flow = np.arange(4 * 4 * 10, dtype=np.float32).reshape(4, 4, 10) / 200
    return regime, source, oof, context, flow


def build(values):
    return build_regime_head_features(*values, n_months=values[3].shape[1])


def test_hand_calculated_source_scaling_channel_layout_and_oof_alignment():
    values = sample()
    result = build(values)
    full, source = result["full_extra"], result["source_extra"]
    assert full.shape == (4, 4, 30) and source.shape == (2, 4, 30)
    assert full.dtype == source.dtype == np.float32
    np.testing.assert_array_equal(result["source_station_ids"], [0, 1])
    np.testing.assert_array_equal(full[..., :10], values[-1])
    np.testing.assert_array_equal(source[..., :10], values[-1][:2])
    np.testing.assert_allclose(full[:, 0, 10], [-.5, .5, .75, 10 / 12], rtol=1e-7)
    np.testing.assert_array_equal(full[..., 19:28], 1)
    np.testing.assert_array_equal(full[..., 29], 1)
    expected = np.zeros((2, 4))
    expected[1, :2] = [-np.sqrt(2) / (1 + np.sqrt(2)), np.sqrt(2) / (1 + np.sqrt(2))]
    np.testing.assert_allclose(source[..., 28], expected, atol=1e-7)
    np.testing.assert_array_equal(source[..., 29], [[0, 1, 0, 0], [1, 1, 0, 0]])
    norm = result["normalization"]
    assert norm["context"]["log_mean"] == pytest.approx(np.log(4))
    assert norm["context"]["log_sd"] == pytest.approx(np.log(2) / np.sqrt(2))
    assert norm["context"]["source_station_counts"] == [1, 2]
    assert norm["ecology"]["median"] == [4.] * 9
    assert norm["ecology"]["iqr"] == [2.] * 9
    assert len(result["feature_names"]) == len(set(result["feature_names"])) == 30
    assert result["feature_names"][28:] == ["context_log_concentration", "context_valid"]
    assert result["interaction_indices"] == {
        "additive": [0, 2, 4], "concentration": [0, 2, 4, 28],
        "ecological": [0, 2, 4, *range(10, 19), 28],
    }
    json.dumps({key: value for key, value in result.items()
                if key not in ("source_extra", "full_extra", "source_station_ids")}, allow_nan=False)


def test_source_statistics_ignore_target_regime_and_full_context_predictions():
    values = sample()
    first = build(values)
    changed = copy.deepcopy(values)
    changed[0][2:, 4:] = 1e7
    changed[3][:] = 1e8
    second = build(changed)
    assert first["normalization"] == second["normalization"]
    np.testing.assert_array_equal(first["source_extra"], second["source_extra"])
    assert not np.array_equal(first["full_extra"], second["full_extra"])
    assert np.abs(second["full_extra"][..., 10:19]).max() <= 1
    assert np.abs(second["full_extra"][..., 28]).max() <= 1


def test_sentinels_partial_missing_inactive_columns_and_zero_iqr():
    values = sample()
    regime = values[0]
    regime[0, 4] = -1
    regime[1, 5] = np.inf
    regime[:2, 6] = np.nan  # This ecological column has no source observations.
    regime[:2, 7] = 10  # Constant source column: IQR fallback is one.
    regime[3, 4:] = -1  # Entire target station missing.
    result = build(values)
    full, norm = result["full_extra"], result["normalization"]["ecology"]
    assert norm["median"][:4] == [6., 2., 0., 10.]
    assert norm["iqr"][:4] == [1., 1., 1., 1.]
    assert norm["active"][:4] == [True, True, False, True]
    np.testing.assert_array_equal(full[0, :, [10, 19]], 0)
    np.testing.assert_array_equal(full[1, :, [11, 20]], 0)
    np.testing.assert_array_equal(full[..., 12], 0)
    np.testing.assert_array_equal(full[2, :, 21], 1)  # Inactive source column still has observed target flag.
    np.testing.assert_array_equal(full[3, :, 10:28], 0)
    assert np.isfinite(full).all()


def test_permuting_source_cells_and_oof_together_is_bitwise_invariant():
    values = sample()
    first = build(values)
    shuffled = (values[0], values[1][::-1], values[2][::-1], values[3], values[4])
    second = build(shuffled)
    assert first["normalization"] == second["normalization"]
    np.testing.assert_array_equal(first["source_extra"], second["source_extra"])
    np.testing.assert_array_equal(first["full_extra"], second["full_extra"])


def test_constant_oof_context_has_explicit_sd_floor_and_no_nan():
    values = sample()
    constant = (values[0], values[1], np.full(3, 4.), values[3], values[4])
    result = build(constant)
    assert result["normalization"]["context"]["log_sd"] == 1e-6
    np.testing.assert_array_equal(result["source_extra"][..., 28], 0)
    assert np.isfinite(result["full_extra"]).all()


def test_target_labels_do_not_enter_builder_and_future_flow_cannot_change_earlier_features():
    values = sample()
    data = {"x": np.ones((4, 4, 2)), "x_mask": np.ones((4, 4, 2), dtype=bool),
            "y": np.full((4, 4), 2.)}
    data["x"][..., 1] = np.arange(1, 17).reshape(4, 4)
    original_flow = build_causal_flow_features(data)["full"]
    first = build((*values[:4], original_flow))
    data["y"][:] = np.nan
    data["x"][:, 3:, 1] = 5000
    second = build((*values[:4], build_causal_flow_features(data)["full"]))
    np.testing.assert_array_equal(second["full_extra"][:, :3], first["full_extra"][:, :3])
    np.testing.assert_array_equal(second["source_extra"][:, :3], first["source_extra"][:, :3])
    assert not np.array_equal(second["full_extra"][:, 3:, :10], first["full_extra"][:, 3:, :10])
    assert "y" not in inspect.signature(build_regime_head_features).parameters
    assert "query_values" not in inspect.signature(build_regime_head_features).parameters


def test_ignored_hydraulic_regime_columns_do_not_change_features():
    values = sample()
    first = build(values)
    values[0][:, :4] = np.nan
    second = build(values)
    assert first["normalization"] == second["normalization"]
    np.testing.assert_array_equal(first["full_extra"], second["full_extra"])


def test_input_shape_cell_identity_and_observed_oof_guards():
    values = sample()
    with pytest.raises(ValueError, match="unique integer"):
        build((values[0], np.array([1, 1, 4]), *values[2:]))
    with pytest.raises(ValueError, match="align"):
        build((values[0], values[1], np.array([1., np.nan, 3.]), *values[3:]))
    with pytest.raises(ValueError, match="align"):
        build((values[0], values[1], np.array([1., 2.]), *values[3:]))
    with pytest.raises(ValueError, match="unique integer"):
        build((values[0], np.array([16, 1, 4]), *values[2:]))
    with pytest.raises(ValueError, match="full_context_native"):
        build((values[0], values[1], values[2], -values[3], values[4]))
    with pytest.raises(ValueError, match="flow_full"):
        build((*values[:4], values[4][..., :9]))
    with pytest.raises(ValueError, match="positive integer"):
        build_regime_head_features(*values, n_months=0)
