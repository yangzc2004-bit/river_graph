import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from river_graph.analysis.river_morphology_closure import (
    junction_translation,
    pulse_response,
    source_dates,
    station_coordinate_fields,
)


def test_analytic_pulse_gain_moments_and_quantile_duration():
    for memory in (0., .25):
        result, pdf, (lo, hi) = pulse_response([.5, 1.5], [.3, .7], sigma=.1, memory=memory)
        mass = quad(pdf, lo, hi, epsabs=1e-9)[0]
        mean = quad(lambda t, density=pdf: t*density(t), lo, hi, epsabs=1e-9)[0]/mass
        variance = quad(lambda t, mu=mean, density=pdf: (t-mu)**2*density(t), lo, hi, epsabs=1e-9)[0]/mass
        assert mass == pytest.approx(.1*np.sqrt(2*np.pi), abs=1e-9)
        assert mean == pytest.approx(1.2, abs=1e-9)
        assert variance == pytest.approx(result["pulse_sd"]**2, abs=1e-8)
        assert 0 < result["pulse_peak"] <= 1
    aligned, _, _ = pulse_response([1., 1.], [.3, .7], sigma=.1)
    assert aligned["pulse_peak"] == pytest.approx(1.)
    assert aligned["pulse_central80"] == pytest.approx(.2563103131, abs=1e-9)


def test_partition_only_junction_null_and_common_translation():
    paths = np.array([.5, 1.5])
    a, _, _ = pulse_response(paths, [.3, .7], sigma=.1)
    for common in (.1, .4):
        b, _, _ = pulse_response(junction_translation(paths, common), [.3, .7], sigma=.1)
        assert b["pulse_peak"] == pytest.approx(a["pulse_peak"], abs=1e-12)
        assert b["pulse_sd"] == a["pulse_sd"]
    shifted, _, _ = pulse_response(paths+2, [.3, .7], sigma=.1)
    assert shifted["pulse_peak"] == pytest.approx(a["pulse_peak"], abs=1e-12)
    assert shifted["pulse_central80"] == pytest.approx(a["pulse_central80"], abs=1e-10)
    assert shifted["pulse_peak_time"]-a["pulse_peak_time"] == pytest.approx(2., abs=1e-7)


def test_alignment_and_memory_separate_geometry_from_time_spreading():
    d, w = np.array([.5, 1.5]), np.array([.3, .7])
    actual, _, _ = pulse_response(d, w, sigma=.1)
    aligned, _, _ = pulse_response(d, w, sigma=.1, offsets=w@d-d)
    stored, _, _ = pulse_response(d, w, sigma=.1, memory=.2)
    assert aligned["pulse_peak"] == pytest.approx(1.)
    assert aligned["arrival_variance"] == 0
    assert aligned["pulse_peak"] > actual["pulse_peak"] > stored["pulse_peak"]
    assert stored["pulse_centroid"] == actual["pulse_centroid"]
    assert stored["pulse_sd"]**2-actual["pulse_sd"]**2 == pytest.approx(.04)
    with pytest.raises(ValueError, match="causal"):
        pulse_response(d, w, sigma=.1, memory=.6)


def test_activity_selection_uses_dates_without_doc_outcomes():
    events = pd.DataFrame({"site_no": ["a", "b", "a"], "event_id": ["x", "y", "z"],
        "date": pd.to_datetime(["2000-01-01", "2000-01-01", "2000-02-01"]), "doc": [1., 2., 3.]})
    before = source_dates(events)
    after = source_dates(events.sample(frac=1, random_state=4).assign(doc=1000.))
    for site in before:
        pd.testing.assert_index_equal(before[site], after[site])


def test_station_catalogue_actual_coordinate_schema_and_missing_fields():
    standard = ("Location_LatitudeStandardized", "Location_LongitudeStandardized")
    original = ("Location_Latitude", "Location_Longitude")
    assert station_coordinate_fields([*standard, *original]) == standard
    assert station_coordinate_fields(original) == original
    with pytest.raises(ValueError, match="coordinate"):
        station_coordinate_fields(["Location_Identifier"])
