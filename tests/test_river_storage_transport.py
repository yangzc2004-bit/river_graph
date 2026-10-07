"""Analytical controls for a causal conservative transport experiment."""
import numpy as np
import pytest
from scipy.integrate import quad

from river_graph.analysis.river_storage_transport import (
    controlled_responses,
    gaussian_storage_response,
    structural_contrasts,
)


def test_fixed_mean_variance_and_anomaly_area():
    metrics, _ = controlled_responses(5, 20, 15, .4, dt=.00125)
    np.testing.assert_allclose(metrics.pulse_centroid, 1, atol=1e-10)
    np.testing.assert_allclose(metrics.pulse_sd, metrics.analytic_sd, atol=1e-9)
    np.testing.assert_allclose(metrics.anomaly_area_fraction, 1, atol=1e-11)
    np.testing.assert_allclose(metrics.common_translation+metrics.storage_tau,
                               metrics.common_mean_budget, atol=1e-15)
    assert metrics.steady_gain.eq(1).all()


def test_storage_is_convolution_not_an_amplitude_loss():
    t = np.arange(-3, 12, .00125)
    y = gaussian_storage_response(t, .5, .15, .35)
    plain = gaussian_storage_response(t, .85, .15, 0)
    assert y.max() < plain.max()
    np.testing.assert_allclose(np.trapezoid(y, t), np.trapezoid(plain, t), atol=1e-12)
    density = y/np.trapezoid(y, t)
    np.testing.assert_allclose(np.trapezoid(t*density, t), .85, atol=1e-10)


def test_closed_response_matches_independent_causal_quadrature():
    for tau in (.003, .15, 3.):
        time = np.array([-.5, .2, .8, 1.5, 4.])
        # Integrate normalized residence u/tau; only source times <= t-delay.
        reference = np.array([quad(lambda u, t=t, tau=tau: np.exp(-u-.5*((t-.2-tau*u)/.15)**2),
                                   0, np.inf, epsabs=1e-11)[0] for t in time])
        np.testing.assert_allclose(gaussian_storage_response(time, .2, .15, tau),
                                   reference, atol=1e-10, rtol=1e-8)


def test_storage_kernel_causality_and_constant_response():
    # O(t)=integral_{-inf}^t I(s) exp(-(t-s)/tau)/tau ds.
    t = np.arange(-10, 5.001, .001)
    tau = .3
    past = t <= 0
    baseline = np.full(len(t), 5.)
    altered = baseline.copy()
    altered[t > 0] = 1000
    k = np.exp(t[past]/tau)/tau
    np.testing.assert_allclose(np.trapezoid(k*baseline[past], t[past]), 5, atol=5e-6)
    np.testing.assert_array_equal(k*baseline[past], k*altered[past])


def test_small_storage_is_stable_and_converges_to_translation():
    t = np.linspace(-2, 3, 2000)
    plain = gaussian_storage_response(t, 1, .15, 0)
    near = gaussian_storage_response(t, 1-1e-9, .15, 1e-9)
    np.testing.assert_allclose(near, plain, atol=1e-12)
    assert np.isfinite(gaussian_storage_response(t, 0, .15, 1e-12)).all()


def test_equal_branches_remove_only_arrival_dispersion():
    metrics, _ = controlled_responses(5, 20, 15, .4)
    for fraction in (0., .25, .5, 1.):
        a, e = [metrics[(metrics.branch_condition == name) & (metrics.storage_fraction == fraction)].iloc[0]
                for name in ("actual", "equal")]
        np.testing.assert_allclose(a.analytic_sd**2-e.analytic_sd**2, a.branch_variance, atol=1e-15)
        np.testing.assert_allclose(a.storage_variance, e.storage_variance, atol=0)
    metrics["pair_id"] = "toy"
    contrast = structural_contrasts(metrics)
    assert contrast.branch_peak_reduction_pct.ge(0).all()
    assert contrast.storage_peak_reduction_pct.ge(-1e-10).all()


def test_no_common_trunk_cannot_acquire_storage():
    metrics, _ = controlled_responses(5, 20, 0, .4)
    assert metrics.storage_tau.eq(0).all()
    for name in ("actual", "equal"):
        f = metrics[metrics.branch_condition.eq(name)]
        assert f.pulse_peak.nunique() == 1


@pytest.mark.parametrize("kwargs", [{"common": -1}, {"weight": 1}, {"sigma": 0},
                                     {"fractions": (0, 1.5)}, {"fractions": (0, 0)}, {"dt": 0}])
def test_invalid_scenario_rejected(kwargs):
    params = {"branch_a": 5, "branch_b": 20, "common": 15, "weight": .4, **kwargs}
    with pytest.raises(ValueError):
        controlled_responses(**params)


def test_grid_convergence_and_source_symmetry():
    m, _ = controlled_responses(5, 20, 15, .4)
    fine, _ = controlled_responses(5, 20, 15, .4, dt=.00125)
    swapped, _ = controlled_responses(20, 5, 15, .6)
    columns = ["pulse_peak", "pulse_sd", "duration_80", "pulse_centroid"]
    np.testing.assert_allclose(m[columns], swapped[columns], atol=1e-12)
    np.testing.assert_allclose(m[columns], fine[columns], atol=1e-4)
