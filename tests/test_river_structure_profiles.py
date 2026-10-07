"""Check real storage allocation and station/network scale in the synthesis."""
import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_structure_profiles import (
    POSITION_ORDER,
    assemble_profiles,
    corridor_storage_profile,
    storage_position_counts,
    summarize_profiles,
)


def corridor():
    return pd.DataFrame({"comid": [1, 2, 3, 4], "segment": ["branch_a", "branch_a", "branch_b", "common"],
                         "length_km": [1., 3., 2., 4.]})


def vaa():
    return pd.DataFrame({"comid": [1, 2, 3, 4], "wbareatype": ["LakePond", None, "Reservoir", "LakePond"],
                         "wbareacomi": [100, 0, 200, 100]})


def test_storage_lengths_are_cropped_flow_weighted_and_unique_waterbodies():
    result = corridor_storage_profile(corridor(), vaa(), .75)
    assert result["branch_a_storage_fraction"] == .25
    assert result["weighted_branch_storage_km"] == 1.25
    assert result["weighted_corridor_storage_km"] == 5.25
    assert result["corridor_storage_path_share"] == 5.25/7.5
    assert result["shared_storage_budget_fraction"] == 4/5.25
    assert result["n_unique_corridor_waterbodies"] == 2
    assert result["storage_position"] == "branches_and_common"


@pytest.mark.parametrize("branch,common,expected", [(False, False, 0), (True, False, 1), (False, True, 2), (True, True, 3)])
def test_storage_positions_and_absence_are_explicit(branch, common, expected):
    v = vaa()
    if not branch:
        v.loc[v.comid.ne(4), "wbareatype"] = None
    if not common:
        v.loc[v.comid.eq(4), "wbareatype"] = None
    result = corridor_storage_profile(corridor(), v, .5)
    assert result["storage_position"] == POSITION_ORDER[expected]
    assert np.isnan(result["shared_storage_budget_fraction"]) == (expected == 0)


def test_missing_vaa_and_duplicate_reaches_are_errors():
    with pytest.raises(ValueError, match="every corridor"):
        corridor_storage_profile(corridor(), vaa().iloc[:-1], .5)
    with pytest.raises(ValueError, match="unique"):
        corridor_storage_profile(pd.concat([corridor(), corridor().iloc[:1]]), vaa(), .5)
    v = vaa()
    v.loc[0, "wbareacomi"] = 0
    with pytest.raises(ValueError, match="physical waterbody"):
        corridor_storage_profile(corridor(), v, .5)


def test_three_segments_and_valid_lengths_required():
    c = corridor()
    c.loc[0, "length_km"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        corridor_storage_profile(c, vaa(), .5)
    with pytest.raises(ValueError, match="common trunk"):
        corridor_storage_profile(corridor().iloc[:-1], vaa(), .5)


def inputs():
    panel = pd.DataFrame({"station": ["001", "002", "003"], "comid": [9, 9, 8], "cluster": [1, 1, 2], "huc4": ["0501", "0501", "0708"]})
    whole = panel.assign(mean_delay=2., path_variance=1., storage_mean_share=.1, storage_exposed_flow_share=.4)
    selected = pd.DataFrame({"station": ["001", "002"], "common_fraction": [.6, .6], "relative_arrival_cv": [.2, .2], "selected_pair_area_share": [.8, .8]})
    partition = pd.DataFrame({"station": ["001", "002"], "storage_position": ["common_only", "common_only"]})
    routing = pd.DataFrame({"comid": [9, 8], "scenario": ["actual_spread", "actual_spread"], "pulse_peak": [.3, .4], "pulse_sd": [.5, .6]})
    return panel, whole, selected, partition, routing


def test_station_geography_and_duplicate_networks_retained_with_explicit_absence():
    frame = assemble_profiles(*inputs())
    assert frame.station.tolist() == ["001", "002", "003"]
    assert frame.huc4.tolist() == ["0501", "0501", "0708"]
    assert frame.arrival_dispersion.tolist() == [.5, .5, .5]
    assert frame.relative_time_pulse_peak.tolist() == [.3, .3, .4]
    assert frame.major_confluence_available.tolist() == [True, True, False]
    assert np.isnan(frame.major_common_share.iloc[2])
    counts = storage_position_counts(frame)
    assert counts.loc[counts.cluster.eq(1), "n"].sum() == 2
    assert counts.loc[counts.cluster.eq(2), "fraction"].isna().all()


def test_geography_or_join_identity_cannot_be_silently_replaced():
    p, w, s, c, r = inputs()
    w.loc[0, "huc4"] = "5010"
    with pytest.raises(ValueError, match="canonical"):
        assemble_profiles(p, w, s, c, r)
    p.loc[0, "huc4"] = "501"
    with pytest.raises(ValueError, match="four-digit"):
        assemble_profiles(p, w, s, c, r)


def test_routing_cannot_duplicate_network_votes_or_drop_a_network():
    p, w, s, c, r = inputs()
    with pytest.raises(ValueError, match="per receiving network"):
        assemble_profiles(p, w, s, c, pd.concat([r, r.iloc[:1]]))
    with pytest.raises(ValueError, match="all canonical networks"):
        assemble_profiles(p, w, s, c, r.iloc[:1])


def test_single_huc_interval_unavailable_and_missing_metric_counted():
    frame = assemble_profiles(*inputs())
    result = summarize_profiles(frame, ["arrival_dispersion", "major_common_share"], draws=100)
    assert result.ci_low.isna().all()
    assert not ((result.cluster == 2) & (result.metric == "major_common_share")).any()
    assert result.n_units.tolist() == [2, 2, 1]
