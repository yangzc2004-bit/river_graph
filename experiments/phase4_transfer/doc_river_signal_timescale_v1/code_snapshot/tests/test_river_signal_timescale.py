import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from river_graph.analysis.river_signal_timescale import (
    filtered_exponential_covariance,
    harmonic_response,
    source_geometry,
    stationary_response,
)


@pytest.mark.parametrize("tau", [0., .4, 1., 1-1e-8, 1+1e-8, 3.])
def test_covariance_against_independent_frequency_integral(tau):
    for separation in (0., .3, 2.):
        def spectrum(omega):
            return 2/(np.pi*(1+omega**2)*(1+(tau*omega)**2))
        if separation:
            value = quad(spectrum, 0, np.inf, weight="cos", wvar=separation, epsabs=1e-9)[0]
        else:
            value = quad(spectrum, 0, np.inf, epsabs=1e-9)[0]
        assert filtered_exponential_covariance(separation, 1., tau) == pytest.approx(value, abs=2e-9)


def test_independent_sources_make_delay_variance_irrelevant():
    actual = stationary_response([.4, 1., 2.], [.2, .3, .5], correlation_time=.7, coherence=0., memory_time=.2)
    aligned = stationary_response([1.38]*3, [.2, .3, .5], correlation_time=.7, coherence=0., memory_time=.2)
    assert actual["geometry_sd_ratio"] == pytest.approx(1.)
    assert actual["sd_ratio"] == pytest.approx(np.sqrt(.7/.9))
    assert actual["sd_ratio"] == pytest.approx(aligned["sd_ratio"])


def test_translation_order_and_units_do_not_change_variance():
    d, w = np.array([.7, 1.4, 2.8]), np.array([.2, .6, .2])
    a = stationary_response(d, w, correlation_time=.5, coherence=.5, memory_time=.2)
    for paths, shares, time, memory in ((d+1, w, .5, .2),
            (d[::-1], w[::-1], .5, .2), (d*100, w, 50., 20.)):
        b = stationary_response(paths, shares, correlation_time=time, coherence=.5, memory_time=memory)
        for key in ("sd_ratio", "geometry_log_change", "shared_memory_log_change", "interaction_log_change"):
            assert a[key] == pytest.approx(b[key])
    assert a["log_sd_ratio"] == pytest.approx(a["geometry_log_change"]+a["shared_memory_log_change"])
    assert a["sd_ratio"] <= a["geometry_sd_ratio"] <= 1


def test_slow_forcing_and_aligned_path_limits():
    fast = stationary_response([.3, 1.5], [.5, .5], correlation_time=.01, coherence=1.)
    slow = stationary_response([.3, 1.5], [.5, .5], correlation_time=1e8, coherence=1.)
    assert fast["sd_ratio"] == pytest.approx(np.sqrt(.5))
    assert slow["sd_ratio"] == pytest.approx(1., abs=1e-8)
    aligned = stationary_response([1., 1.], [.5, .5], correlation_time=.3, coherence=1., memory_time=.3)
    assert aligned["sd_ratio"] == pytest.approx(np.sqrt(.5))
    assert aligned["geometry_log_change"] == 0
    with pytest.raises(ValueError, match="causal translation"):
        stationary_response([.3, 1.5], [.5, .5], correlation_time=.3, coherence=1., memory_time=.4)


def test_passive_delay_can_align_out_of_phase_inputs():
    d, w, phase = np.array([.5, 1.5]), [.5, .5], [0., 3*np.pi/4]
    period = 8/3
    a = harmonic_response(d, w, phase, period=period)
    assert a["outlet_amplitude"] == pytest.approx(1.)
    assert a["amplitude_ratio"] == pytest.approx(1/np.cos(3*np.pi/8))
    t = np.arange(20000)*period/20000
    direct = .5*np.sin(2*np.pi*(t-d[0])/period)+.5*np.sin(2*np.pi*(t-d[1])/period+phase[1])
    assert np.sqrt(2)*direct.std() == pytest.approx(a["outlet_amplitude"])
    b = harmonic_response(d, w, [0., 0.], period=period)
    assert b["amplitude_ratio"] < 1
    stored = harmonic_response(d, w, phase, period=period, memory_time=.3)
    assert stored["outlet_amplitude"] < a["outlet_amplitude"]
    np.testing.assert_array_equal(stored["individual_path_means"], d)
    assert harmonic_response(d, w, [0., np.pi], period=period)["reference_near_cancellation"]


def test_cropped_geometry_counts_common_suffix_once():
    rows = []
    for s, independent, weight in (("a", 1., .3), ("b", 3., .7)):
        rows += [{"target": "z", "source_station": s, "source_order": len(rows)//2,
            "comid": 1 if s == "a" else 2, "sequence": 0, "length_km": independent,
            "area_weight": weight, "shared_by_all_sources": False,
            "mapped_waterbody": False, "waterbody_id": 0},
            {"target": "z", "source_station": s, "source_order": len(rows)//2,
            "comid": 3, "sequence": 1, "length_km": 2., "area_weight": weight,
            "shared_by_all_sources": True, "mapped_waterbody": True, "waterbody_id": 50}]
    f = pd.DataFrame(rows)
    sources, summary = source_geometry(f)
    assert summary["path_mean_km"] == pytest.approx(4.4)
    assert summary["common_km"] == summary["common_tagged_storage_km"] == 2.
    np.testing.assert_allclose(sources.independent_path_km, [1., 3.])
    shuffled, point = source_geometry(f.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(sources, shuffled)
    assert point == summary
    with pytest.raises(AssertionError):
        source_geometry(f.assign(length_km=[1., 2., 3., 3.]))
