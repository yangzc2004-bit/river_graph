"""Scientific units, source visibility and structural classification contracts."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_doc_structure import (
    bootstrap_ols,
    huc_prefix,
    physical_features,
    source_station_response,
)


def test_huc_prefix_handles_huc8_and_huc12_without_station_prefix():
    assert huc_prefix("5010001") == "0501"
    assert huc_prefix("50100010101") == "0501"
    assert huc_prefix("101302010101") == "1013"
    with pytest.raises(ValueError):
        huc_prefix("06438000X")


def toy_dataset():
    months = pd.date_range("2000-01-01", periods=48, freq="MS")
    angle = 2*np.pi*months.month.to_numpy()/12
    flow = np.arange(1, 49.)
    z = 1+.3*np.sin(angle)+.2*np.cos(angle)+.1*np.log1p(flow)
    return {"y": np.stack([np.expm1(z), np.expm1(z)]), "y_mask": np.ones((2, 48), bool),
            "x": np.tile(np.column_stack([np.ones(48)*10, flow]), (2, 1, 1)),
            "x_mask": np.ones((2, 48, 2), bool), "months": months, "site_no": ["a", "b"],
            "feature_channels": ["temperature", "discharge"]}


def test_doc_response_uses_only_unique_permitted_cells_and_harmonic_cq():
    d = toy_dataset()
    cells = np.arange(48)
    a, season, threshold = source_station_response(d, cells)
    assert len(a) == 1 and a.n_doc.iloc[0] == 48 and len(season) == 12
    assert a.cq_slope.iloc[0] == pytest.approx(.1)
    d["y"][1] = 1e9
    b, _, new_threshold = source_station_response(d, cells)
    pd.testing.assert_frame_equal(a, b)
    assert threshold == new_threshold
    with pytest.raises(ValueError, match="unique"):
        source_station_response(d, np.append(cells, cells[0]))


def test_missing_flow_not_read_as_observed_zero():
    d = toy_dataset()
    d["x_mask"][0, :, 1] = False
    d["x"][0, :, 1] = 100000
    a, _, _ = source_station_response(d, np.arange(48))
    assert np.isnan(a.cq_slope.iloc[0]) and np.isnan(a.median_flow.iloc[0])
    assert a.flow_available.iloc[0] == 0


def test_structure_features_ignore_doc_ecology_and_sampled_degree():
    f = {"stream_order": [3, 4], "drainage_area_km2": [100, 200], "slope": [.01, .001],
         "largest_minor_area_share_5km": [.2, 0]}
    for r in (5, 20, 50):
        f[f"upstream_length_{r}km"] = [10, 0]
        f[f"upstream_junction_count_{r}km"] = [2, 0]
        f[f"upstream_major_count_{r}km"] = [1, 0]
        f[f"storage_fraction_{r}km"] = [.1, np.nan]
    a = pd.DataFrame(f)
    x = physical_features(a)
    assert x.junction_density_20.iloc[0] == pytest.approx(np.log1p(20))
    assert x.major_fraction_20.iloc[1] == 0
    a["doc"] = 999
    a["wetland_cover_pct"] = 50
    a["sampled_in_degree"] = 100
    pd.testing.assert_frame_equal(x, physical_features(a))


def test_station_bootstrap_recovers_linear_signal_and_intervals():
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(80), rng.normal(size=80)])
    y = x@np.array([1., 2.])+rng.normal(scale=.1, size=80)
    point, boot, diagnostic = bootstrap_ols(x, y, draws=200)
    assert point[1] == pytest.approx(2, abs=.04)
    assert np.quantile(boot[:, 1], .025) < 2 < np.quantile(boot[:, 1], .975)
    assert diagnostic["rank"] == 2 and diagnostic["n_stations"] == 80


def test_duplicate_design_columns_are_not_separate_effects():
    x = np.column_stack([np.ones(30), np.arange(30.), np.arange(30.)])
    y = 2+3*np.arange(30.)
    point, boot, diag = bootstrap_ols(x, y, draws=20, groups=np.repeat(np.arange(6), 5))
    assert diag["dropped_columns"] == [2] and diag["effective_parameters"] == 2
    np.testing.assert_allclose(x@point, y, atol=1e-10)
    np.testing.assert_allclose(boot[:, 1], 3, atol=1e-9)
