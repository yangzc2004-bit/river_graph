"""Tests for temporal resolution and real directed connections."""

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from river_graph.analysis.river_event_observations import (
    mapped_network,
    mapped_relations,
    match_campaigns,
    read_sites_csv,
    spring_windows,
    window_coverage,
)


def test_sites_timezone_and_censoring(tmp_path):
    p = tmp_path / "doc.csv"
    p.write_text("TIME ZONE: UTC +1\nDOC (mg C L-1)\nTIMESTAMP,DOC\n"
                 "2020-07-01 12:00,10\n2020-07-02 12:00,LOD\n")
    result = read_sites_csv(p, "DOC")
    assert result.timestamp_utc.iloc[0].hour == 11  # fixed offset, not Sweden DST
    assert result.censored.iloc[1] and pd.isna(result.value.iloc[1])


def test_duplicate_conflict_is_not_averaged(tmp_path):
    p = tmp_path / "doc.csv"
    p.write_text("TIME ZONE: UTC+1\nDOC (mg C L-1)\nTIMESTAMP,DOC\n"
                 "2020-01-01 12:00,10\n2020-01-01 12:00,20\n")
    with pytest.raises(ValueError, match="Conflicting"):
        read_sites_csv(p, "DOC")


def test_selection_uses_flow_calendar_only():
    dates = pd.date_range("2020-03-01", "2020-06-30")
    flow = pd.DataFrame({"date_local": dates, "value": 1.})
    flow.loc[60, "value"] = 10
    window = spring_windows(flow, [2020]).iloc[0]
    assert window.peak_date == dates[60]
    assert window.start_date == dates[46]
    assert window.end_date == dates[74]
    assert spring_windows(flow.iloc[:90], [2020]).empty


def test_sampling_all_on_rise_cannot_resolve_response():
    peak = pd.Timestamp("2020-04-20")
    early = pd.DataFrame({"date_local": pd.date_range("2020-04-06", periods=7), "value": 10.})
    assert not window_coverage(early, peak)["spans_response"]
    both = pd.DataFrame({"date_local": pd.date_range("2020-04-06", periods=15, freq="2D"), "value": 10.})
    assert window_coverage(both, peak)["spans_response"]
    assert window_coverage(both, peak)["max_gap_with_edges_days"] == 2


def test_directed_connections_keep_polygon_context_separate():
    graph = nx.DiGraph()
    graph.add_edge("site:A", "site:B", length_m=200)
    graph.add_edge("site:B", "site:C", length_m=300)
    snapping = pd.DataFrame({"site": ["A", "B", "C"], "mapped": True})
    polygons = {"A": box(0, 0, 1, 1), "B": box(0, 0, 2, 2), "C": box(10, 10, 11, 11)}
    result = mapped_relations(graph, snapping, polygons).set_index(["source", "receiver"])
    assert result.loc[("A", "B"), "usable_mapped_connection"]
    assert result.loc[("A", "C"), "metadata_polygon_overlap_share"] == 0
    assert not result.loc[("A", "C"), "nearest_monitored_upstream"]
    assert ("B", "A") not in result.index


def test_station_crop_does_not_include_tributary_below_receiver():
    def feature(arc, start, stop, coords):
        return {"properties": {"ARCID": arc, "FROM_NODE": start, "TO_NODE": stop},
                "geometry": {"type": "LineString", "coordinates": coords}}
    features = [feature(1, 1, 3, [[19., 64.002], [19., 64.]]),
                feature(2, 2, 3, [[19.002, 64.002], [19., 64.]]),
                feature(3, 3, 4, [[19., 64.], [19., 63.998]])]
    stations = pd.DataFrame({"site": ["A", "R", "B"],
                             "longitude": [19., 19., 19.001],
                             "latitude": [64.0018, 64.001, 64.001]})
    graph, snaps, _ = mapped_network(features, stations)
    assert snaps.mapped.all()
    assert nx.has_path(graph, "site:A", "site:R")
    assert not nx.has_path(graph, "site:B", "site:R")


def test_campaign_span_and_no_reused_upstream_samples():
    def sample(hours):
        times = pd.to_datetime([f"2020-04-20 {h:02d}:00Z" for h in hours])
        return pd.DataFrame({"timestamp_utc": times, "date_local": pd.Timestamp("2020-04-20"), "value": 10.})
    # Individually close to receiver, but the full span exceeds 12 hours.
    assert match_campaigns(sample([1]), sample([23]), sample([12])).empty
    matched = match_campaigns(sample([12]), sample([13]), sample([12, 14]))
    assert len(matched) == 1
    assert matched.sampling_span_hours.iloc[0] == 1
