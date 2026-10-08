"""Exact observational identities and distinctions from physical retention."""

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_flow_state_mechanisms import (
    departure_identity,
    mixing_potential,
    no_processing_feasibility,
    shapley_mixing_change,
    state_contrast,
)


def test_synchronous_equal_amplitude_inputs_have_no_fixed_mix_buffer():
    assert mixing_potential({"w": .3, "sd_a": 2., "sd_b": 2., "rho": 1.}) == pytest.approx(0.)
    assert mixing_potential({"w": .5, "sd_a": 2., "sd_b": 2., "rho": -1.}) == pytest.approx(100.)


def test_synchrony_and_water_balance_can_have_opposing_contributions():
    low = {"w": .1, "sd_a": 2., "sd_b": 2., "rho": .1}
    high = {"w": .5, "sd_a": 2., "sd_b": 2., "rho": .5}
    effects, substitutions = shapley_mixing_change(low, high)
    assert effects["correlation_pp"] < 0
    assert effects["mean_flow_share_pp"] > 0
    assert effects["amplitude_balance_pp"] == pytest.approx(0)
    assert len(substitutions) == 8
    assert sum(effects[k] for k in ("correlation_pp", "amplitude_balance_pp", "mean_flow_share_pp")) == pytest.approx(effects["mixing_potential_change_pp"])


def test_shapley_change_is_invariant_to_exchanging_branch_names():
    low = {"w": .2, "sd_a": 2., "sd_b": 3., "rho": .1}
    high = {"w": .4, "sd_a": 4., "sd_b": 2., "rho": .7}
    original, _ = shapley_mixing_change(low, high)
    def swap(x):
        return {"w": 1-x["w"], "sd_a": x["sd_b"], "sd_b": x["sd_a"], "rho": x["rho"]}
    reverse, _ = shapley_mixing_change(swap(low), swap(high))
    for key in original:
        assert original[key] == pytest.approx(reverse[key])


def test_negative_departure_covariance_can_lower_outlet_variance():
    mixture = np.arange(1., 9.)
    terms = departure_identity(mixture, .5*mixture+10)
    assert terms["outlet_mixture_sd_ratio"] == pytest.approx(.5)
    assert terms["normalized_departure_covariance_term"] == pytest.approx(-1)
    assert 1+terms["normalized_departure_variance"]+terms["normalized_departure_covariance_term"] == pytest.approx(.25)


def test_diminishing_input_alone_can_increase_outlet_mixture_ratio():
    parameters = {"w": .5, "sd_a": 1., "sd_b": 1., "rho": .1,
        "dynamic_fixed_mixture_sd_ratio": 1., "normalized_departure_covariance_term": 0.,
        "normalized_departure_variance": 0.}
    low = parameters | {"outlet_sd": 2., "mixture_sd": 4., "outlet_mixture_sd_ratio": .5}
    high = parameters | {"outlet_sd": 2., "mixture_sd": 2., "outlet_mixture_sd_ratio": 1.}
    result = state_contrast(low, high)
    assert result["log_outlet_sd_contribution"] == 0
    assert result["log_mixture_sd_contribution"] == pytest.approx(np.log(2))
    assert result["log_ratio_change"] == pytest.approx(np.log(2))


def test_missing_water_can_reproduce_a_lower_outlet_without_assumed_removal():
    frame = pd.DataFrame({"known_upstream_flow_share": [.5, .5, 1.2, .99, 0.],
        "doc_dynamic_mix": [10., 10., 10., 10., 10.], "doc_receiver": [6., 4., 6., 6., 6.]})
    result = no_processing_feasibility(frame)
    assert result.implied_unmonitored_doc_no_processing.iloc[0] == 2.
    assert result.implied_unmonitored_doc_no_processing.iloc[1] == -2.
    assert result.nonnegative_unmonitored_solution.tolist() == [True, False, False, False, False]
    assert result.budget_exclusion.iloc[2] == "measured_share_exceeds_one"
    assert result.implied_unmonitored_doc_no_processing.iloc[2:].isna().all()


def test_constant_input_does_not_produce_a_fake_identifiable_ratio():
    with pytest.raises(ValueError, match="identifiable"):
        departure_identity(np.ones(8), np.arange(8.))
