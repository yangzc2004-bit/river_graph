"""Scientific contracts for physical-network and source-only atlas analysis."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.topology.river_structure import (
    ReachNetwork,
    source_doc_summaries,
    source_pair_associations,
)


def toy_network():
    # Two equal tributaries meet above reach 3; reach 4 continues downstream.
    return pd.DataFrame({
        "comid": [1, 2, 3, 4], "fromnode": [10, 11, 20, 30], "tonode": [20, 20, 30, 40],
        "hydroseq": [400, 300, 200, 100], "dnhydroseq": [200, 200, 100, 0],
        "lengthkm": [2., 3., 4., 5.], "streamorde": [1, 1, 2, 2],
        "startflag": [1, 1, 0, 0], "divergence": [0, 0, 0, 0],
        "totdasqkm": [10., 10., 20., 30.], "slope": [.01]*4,
        "wbareatype": ["LakePond", "StreamRiver", None, None], "wbareacomi": [101, 102, 0, 0],
    })


def test_confluence_distance_area_balance_and_partial_lengths():
    net = ReachNetwork(toy_network())
    result = net.neighbourhood(3, 50)
    assert result["nearest_major_confluence_km"] == 2
    assert result["largest_minor_area_share_5km"] == .5
    assert result["upstream_length_5km"] == 7  # 2 + 2 + 3, a branched distance window
    assert result["upstream_storage_length_5km"] == 2
    assert result["waterbody_count_5km"] == 1  # StreamRiver polygon excluded
    assert result["confluence"] and result["storage"] and result["low_order"]
    assert not result["physical_headwater"] and not result["chain"]


def test_small_tributary_does_not_create_major_confluence():
    frame = toy_network()
    frame.loc[1, "totdasqkm"] = .1
    result = ReachNetwork(frame).neighbourhood(3, 50)
    assert result["nearest_confluence_km"] == 2
    assert np.isnan(result["nearest_major_confluence_km"])
    assert result["chain"] and not result["confluence"]


def test_missing_measure_is_explicit_and_headwater_uses_physical_flag():
    net = ReachNetwork(toy_network())
    result = net.neighbourhood(4, np.nan)
    assert result["measure_imputed"] and result["station_measure"] == 50
    assert not result["physical_headwater"]  # no monitored adjacency supplied
    assert net.neighbourhood(1, 50)["physical_headwater"]
    with pytest.raises(ValueError, match="measure"):
        net.neighbourhood(4, 101)


def test_radius_boundary_and_cycle_terminate():
    frame = toy_network()
    frame.loc[0, "fromnode"] = 40  # cycle to downstream reach 4
    net = ReachNetwork(frame)
    result = net.neighbourhood(4, 0)
    assert result["searched_reaches"] == 4
    assert result["upstream_length_5km"] == 5
    assert not result["upstream_search_capped"]
    with pytest.raises(ValueError, match="radii"):
        net.neighbourhood(4, 50, radii=(20, 5))


def test_compressed_station_edge_walks_unmonitored_reaches():
    net = ReachNetwork(toy_network())
    result = net.mainstem_path(1, 50, 4, 50)
    assert result["mainstem_connected"]
    assert result["path_length_km"] == 1+4+2.5
    assert result["path_storage_km"] == 1
    assert result["path_reaches"] == 3
    assert result["path_junctions"] == 1
    assert not net.mainstem_path(4, 50, 1, 50)["mainstem_connected"]
    assert net.mainstem_path(3, 75, 3, 25)["path_length_km"] == 2
    assert not net.mainstem_path(3, 25, 3, 75)["mainstem_connected"]


def test_divergent_reconvergence_counts_a_junction_once():
    frame = toy_network()
    frame.loc[2, "tonode"] = 30
    duplicate = frame.iloc[[2]].copy()
    duplicate["comid"], duplicate["hydroseq"] = 5, 150
    frame.loc[2, "divergence"], duplicate["divergence"] = 1, 2
    frame = pd.concat([frame, duplicate], ignore_index=True)
    result = ReachNetwork(frame).neighbourhood(4, 50)
    assert result["upstream_junction_count_20km"] == 2  # node 30 and shared node 20
    assert result["searched_reaches"] == 5


def toy_data():
    n = 36
    month = pd.date_range("2020-01-01", periods=n, freq="MS").astype(str).tolist()
    q = np.arange(n, dtype=float)+1
    return {"y": np.vstack([2+q*.1, 3+q*.2]),
            "y_mask": np.ones((2, n), bool),
            "x": np.stack([np.column_stack([q*0, q])]*2),
            "x_mask": np.ones((2, n, 2), bool),
            "site_no": ["001", "002"], "months": month,
            "feature_channels": ["temperature", "discharge"]}


def test_hidden_labels_cannot_change_descriptive_statistics():
    dataset = toy_data()
    source_cells = np.arange(0, 24)
    before = source_doc_summaries(dataset, source_cells)
    dataset["y"].ravel()[24:] = np.nan
    after = source_doc_summaries(dataset, source_cells)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    assert before.station.tolist() == ["001"]
    assert before.n_doc_flow.iloc[0] == 24
    assert np.isfinite(before.cq_log1p_slope.iloc[0])


def test_pair_associations_keep_hidden_cells_out_and_all_lags():
    dataset = toy_data()
    cells = np.r_[np.arange(24), np.arange(36, 60)]
    edges = pd.DataFrame({"source": ["001"], "target": ["002"]})
    before = source_pair_associations(dataset, cells, edges)
    dataset["y"][:, 24:] = 1e10
    after = source_pair_associations(dataset, cells, edges)
    pd.testing.assert_frame_equal(before, after)
    assert before.lag_months.tolist() == [0, 1, 3, 6, 12]
    assert before.n_pairs.tolist() == [24, 23, 21, 18, 12]
    assert before.rho_log_doc.iloc[0] == pytest.approx(1)


def test_duplicate_comids_or_source_cells_rejected():
    frame = toy_network()
    frame.loc[1, "comid"] = 1
    with pytest.raises(ValueError, match="unique COMIDs"):
        ReachNetwork(frame)
    with pytest.raises(ValueError, match="training cells"):
        source_doc_summaries(toy_data(), np.array([0, 0]))


def load_builder():
    scripts = str(Path(__file__).resolve().parents[1]/"scripts")
    sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("river_atlas_builder", Path(scripts)/"build_doc_river_structure_atlas_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_losses_are_seed_averaged_before_bootstrap():
    module = load_builder()
    rows = []
    for split in (142, 143):
        for seed in (42, 43, 44):
            for cell, station in enumerate(["001", "002"]):
                for model, pred in [("a", seed-41), ("b", 5)]:
                    rows.append({"split_seed": split, "seed": seed, "cell": cell, "station": station,
                                 "y_true": 0., "y_pred": pred, "q90_train": 1., "model_name": model})
    pair = module.paired_errors(pd.DataFrame(rows), "a", "b")
    result = module.joint_station_bootstrap(pair, draws=50)
    assert result["candidate_mae"] == 2
    assert result["reference_mae"] == 5
    assert result["relative_gain_pct"] == 60
    assert result["n_stations_unique"] == 2
    assert result["n_station_months_unique"] == 2


def test_pairing_rejects_truth_change_or_missing_query():
    module = load_builder()
    rows = [{"split_seed": 142, "seed": 42, "cell": 0, "station": "001", "y_true": y,
             "y_pred": 2., "q90_train": 1., "model_name": model} for model, y in [("a", 1.), ("b", 3.)]]
    with pytest.raises(AssertionError):
        module.paired_errors(pd.DataFrame(rows), "a", "b")
    with pytest.raises(ValueError, match="query population"):
        module.paired_errors(pd.DataFrame(rows[:1]), "a", "b")
