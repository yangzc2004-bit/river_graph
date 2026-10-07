import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_hydrologic_activation import (
    clustered_interaction_fit,
    monthly_source_panel,
    station_flow_responses,
)


def test_only_allowed_doc_and_explicit_hydro_visibility():
    dataset = {"y": np.array([[1., 2., 999.]]), "y_mask": np.ones((1, 3), bool),
               "x": np.array([[[3., 10.], [4., -1.], [999., 999.]]]),
               "x_mask": np.array([[[False, True], [True, True], [True, True]]]),
               "feature_channels": ["temperature", "discharge"],
               "months": pd.date_range("2000-01-01", periods=3, freq="MS"), "site_no": ["001"]}
    metadata = pd.DataFrame({"station": ["001"], "included": [True]})
    original = monthly_source_panel(dataset, [0, 1], metadata)
    dataset["y"][0, 2] = -12345
    dataset["x"][0, 2] = -12345
    pd.testing.assert_frame_equal(original, monthly_source_panel(dataset, [0, 1], metadata))
    assert pd.isna(original.temperature_c.iloc[0])
    assert original.flow_reason.iloc[1] == "negative_or_reverse_flow"
    assert not original.flow_usable.iloc[1]
    with pytest.raises(ValueError, match="unique"):
        monthly_source_panel(dataset, [0, 0], metadata)


def synthetic_station(station="001", slope=.3, offset=0., periods=120):
    dates = pd.date_range("2000-01-01", periods=periods, freq="MS")
    rng = np.random.default_rng(42)
    q = 2+.3*np.sin(2*np.pi*dates.month.to_numpy()/12)+rng.normal(0, .35, len(dates))
    center = q-np.median(q)
    z = 2+offset+slope*center-.1*center**2+.4*np.cos(2*np.pi*dates.month.to_numpy()/12)
    z += .03*np.arange(len(dates))/12
    return pd.DataFrame({"station": station, "month": dates, "doc": np.expm1(z),
                         "discharge_cfs": np.expm1(q), "temperature_c": 10+np.sin(np.arange(len(dates))),
                         "flow_usable": True, "temperature_usable": True, "landscape_included": True})


def test_station_season_trend_removed_and_offset_invariant():
    data = synthetic_station()
    residual, fit = station_flow_responses(data)
    np.testing.assert_allclose(fit.cq_linear, .3, atol=1e-10)
    np.testing.assert_allclose(fit.cq_quadratic, -.1, atol=1e-10)
    shifted, shifted_fit = station_flow_responses(synthetic_station(offset=3.))
    np.testing.assert_allclose(fit.cq_linear, shifted_fit.cq_linear, atol=1e-10)
    np.testing.assert_allclose(residual.log_doc_residual, shifted.log_doc_residual, atol=1e-10)
    assert len(residual) == 120


def test_missing_flow_is_excluded_not_zero_and_small_sample_flagged():
    data = synthetic_station()
    data.loc[:100, "flow_usable"] = False
    residual, fit = station_flow_responses(data)
    assert residual.empty
    assert fit.response_status.iloc[0] == "fewer_than_24_usable_months"
    assert fit.n_doc_flow.iloc[0] == 19


def test_equal_station_weight_recovers_known_source_moderation():
    frames, station_rows = [], []
    # Vary station record size so treating each month equally would change the estimand.
    for i in range(12):
        z = float(i % 3)-1
        frame = synthetic_station(station=f"{i:03}", slope=.25+.12*z,
                                  periods=60 if i % 2 else 120)
        frames.append(frame)
        station_rows.append({"station": f"{i:03}", "source": z, "huc2": "01",
                             "huc4": f"{i//2:04}", "cluster": i % 3+1})
    residual, _ = station_flow_responses(pd.concat(frames))
    stats, bootstrap, _, diag = clustered_interaction_fit(
        residual, pd.DataFrame(station_rows), ["source"], draws=50)
    assert diag["identified_columns"] == 3
    result = stats.set_index("term")
    np.testing.assert_allclose(result.loc["flow_intercept", "coefficient"], .25, atol=1e-10)
    np.testing.assert_allclose(result.loc["source", "coefficient"], .12*np.std([-1, 0, 1]), atol=1e-10)
    np.testing.assert_allclose(result.loc["common_flow_curvature", "coefficient"], -.1, atol=1e-10)
    assert np.isfinite(bootstrap).all()
    repeated = pd.concat([residual, *[residual[residual.station.eq("000")]]*4], ignore_index=True)
    replicated, replicated_boot, _, _ = clustered_interaction_fit(
        repeated, pd.DataFrame(station_rows), ["source"], draws=50)
    np.testing.assert_allclose(replicated.coefficient, stats.coefficient, atol=1e-10)
    np.testing.assert_allclose(replicated_boot, bootstrap, atol=1e-10)


def test_temperature_sensitivity_removes_known_confounder():
    data = synthetic_station()
    q = np.log1p(data.discharge_cfs.to_numpy())
    data["temperature_c"] = 4*q+np.random.default_rng(81).normal(0, 1, len(q))
    data["doc"] = np.expm1(np.log1p(data.doc)+.1*data.temperature_c)
    _, adjusted = station_flow_responses(data, temperature=True)
    _, unadjusted = station_flow_responses(data)
    np.testing.assert_allclose(adjusted.cq_linear, .3, atol=1e-10)
    assert abs(unadjusted.cq_linear.iloc[0]-.3) > .2
