"""Tests for mask generation (no network, no dataset files)."""

import numpy as np
import pandas as pd

from river_graph.experiments.masks import make_e1, make_e2, make_e3, observed_cells


def _mask(n=10, t=24, frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, t)) < frac


def test_observed_cells_flat_c_order():
    m = np.zeros((2, 3), dtype=bool)
    m[0, 1] = m[1, 2] = True
    assert observed_cells(m).tolist() == [1, 5]


def test_e1_partition_and_rate():
    m = _mask()
    cells = observed_cells(m)
    for name, split in make_e1(m).items():
        union = np.concatenate([split["train"], split["val"], split["test"]])
        assert len(np.union1d(union, cells)) == len(cells) == len(union)
        r = int(name.split("_")[1][1:]) / 100
        assert abs(len(split["test"]) / len(cells) - r) < 0.05


def test_e1_deterministic_per_seed():
    m = _mask()
    a = make_e1(m)["e1_r20_seed42"]["test"]
    b = make_e1(m)["e1_r20_seed42"]["test"]
    assert np.array_equal(np.sort(a), np.sort(b))


def test_e2_temporal_order():
    n, t = 5, 48
    m = _mask(n, t, frac=0.8)
    months = pd.date_range("2018-01", periods=t, freq="MS")  # 2018..2021
    split = make_e2(m, months)
    cutoff = pd.Timestamp("2020-12")
    for flat in split["test"]:
        assert months[flat % t] > cutoff
    for flat in np.concatenate([split["train"], split["val"]]):
        assert months[flat % t] <= cutoff


def test_e3_held_out_stations_have_no_train_cells():
    n, t = 8, 12
    m = _mask(n, t, frac=0.6)
    sites = [f"s{i}" for i in range(n)]
    edges = pd.DataFrame({"source": sites[:-1], "target": sites[1:]})  # chain
    split = make_e3(m, edges, sites)
    held = set(split["held_out_sites"])
    assert 0 < len(held) < n
    held_rows = {sites.index(s) for s in held}
    trainval = np.concatenate([split["train"], split["val"]])
    assert all((f // t) not in held_rows for f in trainval)
    assert all((f // t) in held_rows for f in split["test"])
