import numpy as np
import pytest

from river_graph.analysis.river_routing_mechanisms import (
    contract_paths,
    distribute_reach_inputs,
    normalized_inputs,
    route_pulse,
    two_branch_process,
)


def test_routing_conserves_anomaly_mass_and_keeps_steady_concentration():
    r, _, pulse = route_pulse([0., 1.], [.5, .5])
    np.testing.assert_allclose(r["anomaly_mass_fraction"], 1, atol=1e-12)
    np.testing.assert_allclose(r["pulse_centroid"], .5)
    np.testing.assert_allclose(r["pulse_sd"], np.sqrt(.15**2+.25))
    assert r["steady_doc"] == 5 and r["outlet_flow"] == 1
    assert .5 <= r["pulse_peak"] < .51 and np.isfinite(pulse).all()


def test_pure_common_delay_changes_phase_not_amplitude_or_mean():
    a, _, _ = route_pulse([0., .7], [.3, .7])
    b, _, _ = route_pulse([2., 2.7], [.3, .7])
    for term in ("pulse_peak", "pulse_sd", "anomaly_mass_fraction", "steady_doc", "period_1_gain", "period_4_gain"):
        np.testing.assert_allclose(a[term], b[term], atol=1e-12)
    np.testing.assert_allclose(b["pulse_centroid"]-a["pulse_centroid"], 2)


def test_spread_contraction_preserves_mean_and_collapsed_pulse_is_input():
    delay, weights = np.array([0., .5, 2.]), np.array([.2, .3, .5])
    for factor in (0., .5, 1.):
        changed = contract_paths(delay, weights, factor)
        np.testing.assert_allclose(np.dot(changed, weights), np.dot(delay, weights))
    r, _, _ = route_pulse(contract_paths(delay, weights, 0), weights)
    np.testing.assert_allclose(r["pulse_peak"], 1, atol=1e-12)
    np.testing.assert_allclose(r["travel_delay_sd"], 0)


def test_junction_split_alone_cannot_change_conservative_or_uniform_process():
    for rate in (0., .1):
        a, _, pulse_a = two_branch_process([1., 2.], .2, .3, branch_rate=rate, common_rate=rate)
        b, _, pulse_b = two_branch_process([1., 2.], .8, .3, branch_rate=rate, common_rate=rate)
        np.testing.assert_allclose(pulse_a, pulse_b, atol=1e-12)
        for term in ("pulse_peak", "pulse_centroid", "pulse_sd", "steady_doc"):
            np.testing.assert_allclose(a[term], b[term], atol=1e-12)


def test_junction_can_matter_through_speed_or_processing_environment():
    a, _, _ = two_branch_process([1., 2.], .2, .2, flow_exponent=1/3)
    b, _, _ = two_branch_process([1., 2.], .8, .2, flow_exponent=1/3)
    assert a["pulse_centroid"] != b["pulse_centroid"] and a["travel_delay_sd"] != b["travel_delay_sd"]
    c, _, _ = two_branch_process([1., 2.], .2, .2, branch_rate=.1, common_rate=.8)
    d, _, _ = two_branch_process([1., 2.], .8, .2, branch_rate=.1, common_rate=.8)
    np.testing.assert_allclose(d["retained_fraction"]/c["retained_fraction"], np.exp(-.7*.6))
    assert d["steady_doc"] < c["steady_doc"]


def test_arrival_alignment_restores_peak_but_not_extra_mass():
    apart, _, _ = two_branch_process([1., 2.], .2, .2)
    aligned, _, _ = two_branch_process([1., 2.], .2, .2, offset_b=-1)
    assert aligned["pulse_peak"] > apart["pulse_peak"]
    np.testing.assert_allclose(aligned["pulse_peak"], 1, atol=1e-12)
    np.testing.assert_allclose(aligned["anomaly_mass_fraction"], 1, atol=1e-12)


def test_frequency_gain_depends_on_period_and_not_only_spread():
    r, _, _ = route_pulse([0., .5], [.5, .5])
    np.testing.assert_allclose(r["period_1_gain"], 0, atol=1e-12)
    assert r["period_12_gain"] > .99


def test_area_scale_and_unreachable_catchments_are_explicit():
    weights, paths, coverage = normalized_inputs([1, 3, 4], [2, 8, np.inf], 16)
    np.testing.assert_allclose(weights, [.25, .75])
    np.testing.assert_allclose(paths, [.5, 2])
    assert coverage == .5
    _, changed, _ = normalized_inputs([4, 12, 16], [4, 16, np.inf], 64)
    np.testing.assert_allclose(paths, changed)
    with pytest.raises(ValueError, match="common segment"):
        two_branch_process([1, 2], 1.5, .5)
    with pytest.raises(ValueError, match="causal"):
        route_pulse([-1, 2], [.5, .5])


def test_within_reach_distribution_preserves_flow_mean_and_adds_spread():
    delay, weights, lengths = np.array([1., 2.]), np.array([.3, .7]), np.array([1., 2.])
    distributed, dw = distribute_reach_inputs(delay, weights, lengths)
    np.testing.assert_allclose(dw.sum(), 1)
    np.testing.assert_allclose(dw.reshape(2, 5).sum(1), weights)
    np.testing.assert_allclose(np.dot(distributed, dw), np.dot(delay, weights))
    coarse, _, _ = route_pulse(delay, weights)
    refined, _, _ = route_pulse(distributed, dw)
    expected_extra_variance = .08*np.dot(weights, lengths**2)
    np.testing.assert_allclose(refined["pulse_sd"]**2-coarse["pulse_sd"]**2, expected_extra_variance)
    np.testing.assert_allclose(refined["anomaly_mass_fraction"], coarse["anomaly_mass_fraction"])
    with pytest.raises(ValueError, match="past the receiving outlet"):
        distribute_reach_inputs([.1], [1.], [2.])
