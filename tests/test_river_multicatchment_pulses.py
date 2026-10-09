from io import StringIO

import numpy as np
import pandas as pd

from river_graph.analysis.river_doc_pulses import flow_pulses, measure_doc_responses
from river_graph.analysis.river_multicatchment_pulses import (
    canonical_clock,
    hourly_observations,
    normalize,
    read_bouleau,
    read_rappbode,
)


def test_hourly_protocol_detects_known_delay_without_quarter_hour_count_requirement():
    clock = pd.date_range("2021-01-01", periods=241, freq="h")
    hours = np.arange(241)-120
    frame = pd.DataFrame({"clock": clock, "q_m3_s": .1+.3*np.exp(-.5*(hours/4)**2),
                          "doc_mg_l": 2+5*np.exp(-.5*((hours-3)/6)**2)})
    flow, doc, _, _ = normalize(frame, "synthetic", 60, 2., "native", "none")
    measured = measure_doc_responses(flow_pulses(flow, sample_minutes=60), flow, doc, sample_minutes=60)
    assert len(measured) == 1
    assert measured.iloc[0].width_pair_eligible
    assert measured.iloc[0].doc_peak_lag_hours == 3
    assert measured.iloc[0].antecedent_doc_expected_bins == 6
    assert 1.3 < measured.iloc[0].doc_flow_width_ratio < 1.8


def test_duplicate_conflicts_are_not_arbitrarily_selected():
    clock = pd.to_datetime(["2021-01-01"]*3)
    frame = pd.DataFrame({"clock": clock, "q_m3_s": [1., 1., 1.], "doc_mg_l": [2., 3., 2.]})
    out, conflicts = canonical_clock(frame)
    assert len(out) == 1 and out.q_m3_s.iloc[0] == 1.
    assert np.isnan(out.doc_mg_l.iloc[0])
    assert conflicts.field.tolist() == ["doc_mg_l"]


def test_bouleau_uses_observed_not_predicted_doc_and_converts_hourly_flow(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text("/* metadata */\nSite\tDate/Time\tQ [m**3/h]\tDOC [mg/l]\tDOC [mg/l] (Predicted)\n"
                    "R01\t2018-07-01T00:00:00\t3600\t2\t900\n"
                    "R01\t2018-07-01T01:00:00\t7200\t\t901\n")
    flow, doc, audit, _ = read_bouleau([path])
    assert flow.q_m3_s.tolist() == [1., 2.]
    assert doc.doc_mg_l.tolist() == [2.]
    assert doc.timestamp_utc.dt.tz is None
    assert audit["predicted_doc_rows_not_used"] == 2


def test_rappbode_specific_discharge_conversion_and_negative_doc_exclusion():
    source = StringIO('Date.time,Q.smooth,DOC.smooth\n2018-01-01,86.4,2\n2018-01-01 00:15:00,43.2,-.1\n')
    flow, doc, audit, _ = read_rappbode(source)
    assert np.allclose(flow.q_m3_s, [2.58, 1.29])
    assert len(doc) == 1 and audit["negative_doc_excluded"] == 1


def test_hourly_sampling_does_not_average_or_fill():
    clock = pd.date_range("2021-01-01", periods=9, freq="15min")
    frame = pd.DataFrame({"clock": clock, "q_m3_s": np.arange(9), "doc_mg_l": np.arange(9)*10.})
    flow, doc, _, _ = normalize(frame, "synthetic", 15, 1., "native", "none")
    flow, doc = hourly_observations(flow, doc)
    assert flow.q_m3_s.tolist() == [0, 4, 8]
    assert doc.doc_mg_l.tolist() == [0, 40, 80]
