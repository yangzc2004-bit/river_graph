"""Structural participation and covariance summaries on unchanged real paths."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_active_structure import (
    hydro_contrasts,
    mixing_decomposition,
    multibranch_potential,
    observed_mixing,
    participation,
)
from river_graph.analysis.river_flow_state_mechanisms import mixing_potential


def test_effective_participation_is_not_the_physical_branch_count():
    out = participation([1, 3], [[.99, .01], [.5, .5]], reference_mean=2, common_km=.5)
    assert out.effective_sources.iloc[0] < 1.03
    assert out.effective_sources.iloc[1] == 2
    assert out.path_sd_km.iloc[1] == 1
    assert out.effective_fraction.between(.5, 1).all()


def test_fixed_normalization_and_common_translation_have_distinct_meanings():
    a = participation([1, 3], [[.2, .8]], reference_mean=2, common_km=.5)
    b = participation([2, 4], [[.2, .8]], reference_mean=2, common_km=1.5)
    assert a.path_sd_reference.iloc[0] == pytest.approx(b.path_sd_reference.iloc[0])
    assert a.path_mean_km.iloc[0]+1 == pytest.approx(b.path_mean_km.iloc[0])


def test_multibranch_potential_reproduces_the_two_branch_definition():
    assert multibranch_potential([.2, .8], [2., 3.], [[1., .4], [.4, 1.]]) == pytest.approx(
        mixing_potential({"w": .2, "sd_a": 2, "sd_b": 3, "rho": .4}))
    assert multibranch_potential([1/3]*3, [2]*3, np.eye(3)) == pytest.approx(200/3)


def test_analytic_contributions_sum_and_are_invariant_to_source_order():
    low = {"weights": np.array([.1, .3, .6]), "sd": np.array([2., 3., 4.]), "correlation": np.eye(3)}
    high = {"weights": np.array([.3, .3, .4]), "sd": np.array([3., 2., 4.]), "correlation": .4*np.ones((3, 3))+.6*np.eye(3)}
    a = mixing_decomposition(low, high)
    def reverse(p):
        return {"weights": p["weights"][::-1], "sd": p["sd"][::-1], "correlation": p["correlation"][::-1, ::-1]}
    b = mixing_decomposition(reverse(low), reverse(high))
    for key in a:
        assert a[key] == pytest.approx(b[key])


def test_hydro_contrasts_remove_calendar_structure_from_constant_participation():
    dates = pd.date_range("2001-01-01", periods=60, freq="MS")
    frame = participation([1, 3], np.broadcast_to([.2, .8], (60, 2)), reference_mean=2, common_km=.5)
    frame["date"] = dates
    frame["flow_state"] = np.tile(["low", "middle", "high", "high", "middle"], 12)
    frame["source_receiver_flow_share"] = .7
    result = hydro_contrasts(frame)
    assert result["effective_fraction_adjusted_change"] == pytest.approx(0, abs=1e-12)
    assert result["path_sd_reference_adjusted_change"] == pytest.approx(0, abs=1e-12)


def test_invalid_correlations_and_impossible_shared_paths_are_rejected():
    with pytest.raises(ValueError):
        participation([1, 3], [.5, .5], reference_mean=2, common_km=2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        multibranch_potential([.5, .5], [2., 3.], [[1., 2.], [2., 1.]])


def test_dynamic_mixture_and_outlet_share_the_same_calendar_projection():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2001-01-01", periods=60, freq="MS")
    sources = rng.uniform(2, 20, (60, 3))
    weights = rng.uniform(.1, 1, (60, 3))
    weights /= weights.sum(axis=1, keepdims=True)
    outlet = np.sum(sources*weights, axis=1)
    states = np.tile(["low", "middle", "high", "high", "middle"], 12)
    result = observed_mixing(sources, outlet, weights, dates, states)
    assert result["log_ratio_change"] == pytest.approx(0, abs=1e-12)
    assert result["low_ratio"] == pytest.approx(1)
    assert result["high_ratio"] == pytest.approx(1)
