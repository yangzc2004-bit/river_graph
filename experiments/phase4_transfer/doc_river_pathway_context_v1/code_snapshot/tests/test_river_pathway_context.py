from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_junction_layout import partition_mainstem
from river_graph.analysis.river_pathway_context import (
    comparison_opportunities,
    corridor_geometry,
    variance_budget,
)


def geometry():
    p = SimpleNamespace(comids=np.arange(10, 15), successor=np.array([-1, 0, 0, 1, 2]),
        length=np.ones(5), area=np.array([1., 2., 3., 4., 5.]), distance=np.array([.5, 1.5, 1.5, 2.5, 2.5]))
    layout = partition_mainstem(p, np.arange(1, 6), np.arange(5.))
    g = pd.DataFrame({"station": ["a", "b"], "comid": [13, 14], "measure": [50., 50.],
        "source_order": [0, 1], "area_weight": [.4, .6]})
    return p, layout, g


def test_gauge_and_receiver_measures_crop_common_suffix():
    p, layout, g = geometry()
    result, f = corridor_geometry(p, g, 10, 50., ["LakePond", "LakePond", "", "", ""], [30, 30, 0, 0, 0], layout)
    assert result["monitored_path_mean_km"] == pytest.approx(2.)
    assert result["monitored_common_km"] == pytest.approx(.5)
    assert result["monitored_common_fraction"] == pytest.approx(.25)
    assert result["monitored_storage_length_fraction"] == pytest.approx(.45)
    assert result["n_monitored_corridor_waterbodies"] == 1
    assert result["monitored_storage_source_share"] == pytest.approx(1.)
    assert f[f.shared_by_all_sources].comid.unique().tolist() == [10]
    assert result["monitored_path_variance_km2"] == pytest.approx(0.)


def test_storage_outside_cropped_corridor_is_not_counted():
    p, layout, g = geometry()
    # A source exactly at its reach outlet does not traverse that lake reach.
    g.loc[0, "measure"] = 0.
    result, f = corridor_geometry(p, g, 10, 100., ["LakePond", "", "", "LakePond", ""], [30, 0, 0, 40, 0], layout)
    assert result["n_whole_mapped_waterbodies"] == 2
    assert result["n_monitored_corridor_waterbodies"] == 0
    assert result["monitored_storage_length_fraction"] == 0
    assert f[f.mapped_waterbody].length_km.sum() == 0


def test_geometry_source_order_is_invariant():
    p, layout, g = geometry()
    a, _ = corridor_geometry(p, g, 10, 50., [""]*5, [0]*5, layout)
    b, _ = corridor_geometry(p, g.iloc[::-1], 10, 50., [""]*5, [0]*5, layout)
    assert a == pytest.approx(b)


def test_geometry_rejects_invalid_measure_and_disconnected_source():
    p, layout, g = geometry()
    with pytest.raises(ValueError, match="receiver measure"):
        corridor_geometry(p, g, 10, -1., [""]*5, [0]*5, layout)
    p.successor[4] = -1
    with pytest.raises(ValueError, match="does not reach"):
        corridor_geometry(p, g, 10, 50., [""]*5, [0]*5, layout)


def test_observed_amplification_variance_identity():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    a = 10+np.random.default_rng(42).normal(size=(60, 3))
    w = np.array([.2, .3, .5])
    result, _ = variance_budget(a, 2*(a@w), w, dates)
    assert result["mixture_variance_share"] == pytest.approx(.25)
    assert result["mismatch_variance_share"] == pytest.approx(.25)
    assert result["covariance_variance_share"] == pytest.approx(.5)
    assert result["outlet_mix_log_sd_ratio"] == pytest.approx(np.log(2))
    assert result["unexplained_outlet_variance_fraction"] == pytest.approx(0., abs=1e-10)


def test_time_varying_weights_multiply_before_projection():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    a = np.tile([5., 15.], (60, 1))
    w = np.column_stack([1+np.arange(60)%5, np.ones(60)])
    y = 1+(a*w).sum(axis=1)/w.sum(axis=1)
    result, frame = variance_budget(a, y, w, dates)
    np.testing.assert_allclose(frame.doc_mixture, y-1.)
    assert result["mixture_variance"] > 1
    assert result["mixture_variance_share"] == pytest.approx(1.)
    assert result["mismatch_variance"] == pytest.approx(0., abs=1e-10)


def test_constant_signals_are_unidentifiable_and_duplicate_dates_rejected():
    dates = pd.date_range("2000-01-01", periods=36, freq="MS")
    result, _ = variance_budget(np.full((36, 2), 4.), np.full(36, 5.), [.5, .5], dates)
    assert np.isnan(result["outlet_mix_log_sd_ratio"])
    assert np.isnan(result["unexplained_outlet_variance_fraction"])
    with pytest.raises(ValueError, match="unique"):
        variance_budget(np.ones((36, 2)), np.ones(36), [.5, .5], dates.where(dates != dates[1], dates[0]))


def test_same_region_comparison_uses_no_doc_outcomes():
    f = pd.DataFrame({"station": ["a", "b", "c", "d"], "comid": [1, 2, 3, 4],
        "physical_receiver_representative": True, "huc4": ["0503", "0503", "0503", "0708"],
        "cluster": [1, 3, 3, 3], "basin_area_km2": [10., 15., 100., 12.],
        "status": ["included"]*4, "covered_area_fraction": [.9, .85, .95, .95]})
    a = comparison_opportunities(f)
    b = comparison_opportunities(f.assign(doc_response=[-100., 999., 0., 1.]))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2 and a.usable_form_pair.sum() == 1
    assert a.broad_station.tolist() == ["b", "c"]
