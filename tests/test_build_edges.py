"""Tests for edge construction logic (no network)."""

import pandas as pd

from river_graph.topology.build_edges import build_edges, first_station_downstream


def _stations():
    return pd.DataFrame(
        {
            "site_no": ["A", "B", "C"],
            "comid": ["100", "200", "300"],
        }
    )


def test_first_station_downstream_skips_own_reach():
    comid_to_sites = {"100": ["A"], "200": ["B"], "300": ["C"]}
    walk = ["100", "200", "300"]  # own reach first, like NLDI returns
    assert first_station_downstream(walk, comid_to_sites, "A") == "B"


def test_first_station_downstream_none_at_outlet():
    comid_to_sites = {"300": ["C"]}
    assert first_station_downstream(["300"], comid_to_sites, "C") is None


def test_build_edges_linear_chain():
    stations = _stations()
    flowlines = {
        "A": ["100", "200", "300"],
        "B": ["200", "300"],
        "C": ["300"],
    }
    edges = build_edges(stations, flowlines)
    assert set(map(tuple, edges[["source", "target"]].values)) == {("A", "B"), ("B", "C")}


def test_build_edges_tributary_confluence():
    # tributary station T flows into the reach hosting B -> edge T -> B
    stations = pd.DataFrame(
        {"site_no": ["A", "T", "B"], "comid": ["100", "150", "200"]}
    )
    flowlines = {
        "A": ["100", "200"],
        "T": ["150", "200"],
        "B": ["200"],
    }
    edges = build_edges(stations, flowlines)
    assert set(map(tuple, edges[["source", "target"]].values)) == {("A", "B"), ("T", "B")}


def test_build_edges_skips_unmatched_stations():
    stations = _stations()
    stations.loc[3] = ["D", None]  # no COMID
    flowlines = {"A": ["100", "200"], "B": ["200", "300"], "C": ["300"], "D": []}
    edges = build_edges(stations, flowlines)
    assert "D" not in set(edges["source"]) | set(edges["target"])


def test_shared_reach_links_to_other_station():
    # two stations snapped to the same reach: walking from A finds B
    comid_to_sites = {"100": ["A", "B"]}
    assert first_station_downstream(["100"], comid_to_sites, "A") == "B"
