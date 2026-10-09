import numpy as np
import pandas as pd

from river_graph.analysis.river_doc_pulses import (
    flow_pulses,
    half_excess_width,
    measure_doc_responses,
)
from river_graph.analysis.river_kervidy_observations import prepare_flow


def synthetic_record():
    h = np.arange(-96., 96.25, .25)
    t = pd.Timestamp("2021-02-15", tz="UTC")+pd.to_timedelta(h, unit="h")
    q = .1+.3*np.exp(-.5*(h/4)**2)
    flow = prepare_flow(pd.DataFrame({"timestamp_utc": t, "q_dm3_s": q*1000}))
    doc = pd.DataFrame({"timestamp_utc": t, "doc_mg_l": 2+5*np.exp(-.5*((h-3)/6)**2)})
    return flow, doc


def test_known_delayed_broader_doc_pulse_has_three_hour_lag():
    f, d = synthetic_record()
    p = flow_pulses(f)
    assert len(p) == 1
    x = measure_doc_responses(p, f, d).iloc[0]
    assert x.lag_eligible and x.width_pair_eligible
    assert x.doc_peak_lag_hours == 3
    assert 1.35 < x.doc_flow_width_ratio < 1.60


def test_doc_values_do_not_define_flow_candidates():
    f, d = synthetic_record()
    p = flow_pulses(f)
    d.doc_mg_l = d.doc_mg_l.iloc[::-1].to_numpy()*100
    pd.testing.assert_frame_equal(p, flow_pulses(f))
    assert measure_doc_responses(p, f, d).event_id.tolist() == p.event_id.tolist()


def test_missing_hour_prevents_claiming_a_crossing_width():
    t = pd.date_range("2021-01-01", periods=9, freq="h", tz="UTC")
    y = [1., 2., 4., 6., 8., 6., 4., 2., 1.]
    x = half_excess_width(t, y, 4, 1)
    assert x["width_hours"] == 4
    x = half_excess_width(t.delete(3), np.delete(y, 3), 3, 1)
    assert np.isnan(x["width_hours"])
    assert x["crossing_gap_hours"] == 2


def test_no_return_is_censored_and_not_assigned_an_arbitrary_width():
    f, d = synthetic_record()
    d = d[d.timestamp_utc <= pd.Timestamp("2021-02-15T04:00Z")]
    r = measure_doc_responses(flow_pulses(f), f, d).iloc[0]
    assert r.doc_right_crossing_missing
    assert r.doc_recovery_censored
    assert np.isnan(r.doc_width_hours)
    assert not r.lag_eligible


def test_no_peak_in_constant_flow():
    f, _ = synthetic_record()
    f.q_m3_s = .1
    assert flow_pulses(f).empty


def test_later_nested_flow_envelope_is_not_ignored():
    f, d = synthetic_record()
    p = flow_pulses(f)
    later = p.iloc[[0]].copy()
    later["event_id"] = "later_larger_peak"
    later["flow_peak_utc"] += pd.Timedelta(hours=20)
    later["start_utc"] -= pd.Timedelta(hours=1)
    later["flow_return_utc"] += pd.Timedelta(hours=40)
    events = pd.concat([p, later], ignore_index=True)
    r = measure_doc_responses(events, f, d).iloc[0]
    assert r.next_flow_envelope_overlaps_start
    assert r.followup_interrupted_before_flow_return
    assert not r.lag_eligible
