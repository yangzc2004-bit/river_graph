"""Separate source clocks, transit and conservative waveform broadening."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_junction_dynamics import (
    field_transit_proxies,
    junction_response,
)


def response(**kwargs):
    return junction_response(2., 1., 1., .4, **kwargs)[0]


def test_source_compensation_removes_arrival_spread_not_river_geometry():
    compensated = response(phase=-1.)
    natural = response(phase=0.)
    reinforced = response(phase=1.)
    assert compensated["pulse_peak"] == pytest.approx(1., abs=1e-12)
    assert compensated["arrival_variance"] == 0
    assert reinforced["arrival_variance"] == pytest.approx(4*natural["arrival_variance"])
    assert compensated["pulse_peak"] > natural["pulse_peak"] > reinforced["pulse_peak"]
    for result in (compensated, natural, reinforced):
        assert result["pulse_centroid"] == pytest.approx(1., abs=1e-11)
        assert result["source_phase_weighted_mean"] == pytest.approx(0., abs=1e-15)
        assert result["anomaly_area_fraction"] == pytest.approx(1., abs=1e-11)


def test_volume_without_dispersion_moves_peak_without_attenuation():
    baseline = response(volume=1.)
    slower = response(volume=2.)
    assert slower["pulse_peak"] == pytest.approx(baseline["pulse_peak"], abs=1e-11)
    assert slower["duration_80"] == pytest.approx(baseline["duration_80"], abs=2e-5)
    assert slower["pulse_centroid"]-baseline["pulse_centroid"] == pytest.approx(baseline["shared_reference_mean"])
    assert slower["peak_time"]-baseline["peak_time"] == pytest.approx(baseline["shared_reference_mean"], abs=1e-7)


def test_flow_can_cancel_volume_change_exactly():
    baseline, curve_a = junction_response(2., 1., 1., .4, mixing=.5, keep_curve=True)
    compensated, curve_b = junction_response(2., 1., 1., .4, volume=2., flow=2., mixing=.5, keep_curve=True)
    pd.testing.assert_frame_equal(curve_a, curve_b)
    for metric in ("pulse_peak", "pulse_centroid", "duration_80", "pulse_sd"):
        assert baseline[metric] == compensated[metric]


def test_distributed_response_preserves_anomaly_and_exact_moments():
    result = response(phase=1., volume=2., flow=1., mixing=.5)
    assert result["anomaly_area_fraction"] == pytest.approx(1., abs=1e-10)
    assert result["pulse_centroid"] == pytest.approx(result["analytic_centroid"], abs=1e-10)
    assert result["pulse_sd"] == pytest.approx(result["analytic_sd"], abs=1e-9)
    translated = response(phase=1., volume=2., flow=1., mixing=0.)
    assert result["pulse_peak"] <= translated["pulse_peak"]


def test_common_zero_has_no_volume_or_mixing_effect():
    a = junction_response(2., 1., 0., .4, volume=1., mixing=0.)[0]
    b = junction_response(2., 1., 0., .4, volume=2., mixing=.5)[0]
    assert a["pulse_peak"] == b["pulse_peak"]
    assert a["duration_80"] == b["duration_80"]


@pytest.mark.parametrize("parameter,value", [("volume", 0.), ("flow", -1.), ("mixing", 1.1),
                                           ("phase", -2.), ("sigma", np.nan)])
def test_invalid_scenarios_are_rejected(parameter, value):
    with pytest.raises(ValueError):
        response(**{parameter: value})


def field_fixture():
    survey = pd.DataFrame({"confluence": ["Con-1"]*4,
        "reach": ["1.upstream"]*2+["2.downstream"]*2, "transect": ["1", "2", "1", "2"],
        "width_m": [1., 3., 2., 6.], "mean_depth_m": [3., 1., 3., 1.],
        "n_missing_depth_points": [0, 0, 0, 0]})
    chemistry = pd.DataFrame({"confluence": ["Con-1"], "season": ["fall"],
        "main_flow_ls": [2.], "receiver_flow_ls": [4.], "receiver_main_residence_ratio": [1.2],
        "receiver_main_depth_ratio": [1.], "receiver_main_width_ratio": [2.],
        "doc_receiver_minus_mix_pct": [5.]})
    return survey, chemistry


def test_area_proxy_averages_transect_products_not_product_of_reach_means():
    survey, chemistry = field_fixture()
    result, _ = field_transit_proxies(survey, chemistry)
    row = result.iloc[0]
    assert row.main_area_proxy_m2 == 3.
    assert row.receiver_area_proxy_m2 == 6.
    assert row.same_length_transit_proxy_ratio == 1.
    assert row.reported_tracer_transit_ratio == 1.2


def test_unobserved_depth_does_not_become_zero_area():
    survey, chemistry = field_fixture()
    survey.loc[0, "mean_depth_m"] = np.nan
    survey.loc[0, "n_missing_depth_points"] = 5
    result, products = field_transit_proxies(survey, chemistry)
    assert pd.isna(products.rectangular_area_proxy_m2.iloc[0])
    assert result.main_usable_proxy_units.iloc[0] == 1
    assert result.n_missing_depth_points.iloc[0] == 5


def test_duplicate_survey_units_are_not_extra_evidence():
    survey, chemistry = field_fixture()
    with pytest.raises(ValueError, match="transect"):
        field_transit_proxies(pd.concat([survey, survey.iloc[[0]]]), chemistry)
