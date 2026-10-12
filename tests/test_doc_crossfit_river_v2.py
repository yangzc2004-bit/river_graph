"""Contracts for full local-label isolation and capacity-matched river controls."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from river_graph.models.crossfit_river import (
    LocalEnvironmentGRU,
    local_batch,
    new_river,
    relation_banks,
    river_batch,
)
from river_graph.models.kgml_local_transport import station_folds


def tiny():
    n, t = 8, 5
    names = np.asarray([str(i) for i in range(n)])
    edges = pd.DataFrame({"source": ["0", "1", "2"], "target": ["1", "2", "3"],
        "mainstem_connected": True, "path_length_km": 2., "path_major_junctions": 0., "path_storage_km": 0.})
    physical = pd.DataFrame({"station": names, "comid": np.arange(n), "drainage_area_km2": 100.,
                             "stream_order": 2., "storage_fraction_20km": 0.})
    inputs = {"temporal": np.ones((n, t, 14), np.float32), "env": np.ones((n, 15), np.float32),
              "daily": np.ones((n, t, 8), np.float32), "available": np.ones((n, t), bool)}
    anchor = {"states": np.ones((n, t, 83), np.float32), "logpred": np.ones((n, t), np.float32)}
    bank = relation_banks(edges, physical, names, [1], [0, 2, 3, 4, 5, 6, 7], inputs["available"], 42)
    return inputs, anchor, bank


def test_direction_and_nonconnected_exclusion():
    _, _, bank = tiny()
    records = bank["records"]
    assert {r["directed"] for r in records} == {0, 2, 3}
    assert {r["nonconnected"] for r in records} <= {4, 5, 6, 7}
    assert {r["direction"] for r in records} == {1, -1}
    assert bank["valid"][1, 0, :, 1:].sum() == 0


def test_missing_unused_storage_attribute_never_creates_nan():
    names = np.asarray([str(i) for i in range(8)])
    edges = pd.DataFrame({"source": ["0"], "target": ["1"], "mainstem_connected": True,
        "path_length_km": 2., "path_major_junctions": 0., "path_storage_km": 0.})
    physical = pd.DataFrame({"station": names, "comid": np.arange(8), "drainage_area_km2": 100.,
        "stream_order": 2., "storage_fraction_20km": np.nan})
    bank = relation_banks(edges, physical, names, [1], [0, 2, 3, 4, 5, 6, 7], np.ones((8, 5), bool), 42)
    assert np.isfinite(bank["paths"]).all()


def test_common_support_and_generic_feature_contract():
    inputs, anchor, bank = tiny()
    masks = []
    for arm in ("directed", "generic", "nonconnected"):
        arrays, _ = river_batch(inputs, bank, anchor, np.asarray([5, 7, 9]), arm)
        masks.append(arrays["valid"])
        assert not torch.count_nonzero(arrays["states"][~arrays["valid"]])
        if arm == "generic":
            assert not torch.count_nonzero(arrays["features"][..., :8])
    assert torch.equal(masks[0], masks[1]) and torch.equal(masks[1], masks[2])


def test_capacity_initialization_and_zero_message():
    models = [new_river(42) for _ in range(6)]
    for model in models[1:]:
        for a, b in zip(model.state_dict().values(), models[0].state_dict().values()):
            assert torch.equal(a, b)
    inputs, anchor, bank = tiny()
    arrays, _ = river_batch(inputs, bank, anchor, np.asarray([25]), "directed")
    delta, diagnostic = models[0](**arrays)
    assert delta.item() == 0 and not diagnostic["support"].item()


def test_causal_local_window_and_fresh_initialization():
    inputs, _, _ = tiny()
    windows, env = local_batch(inputs, np.asarray([0, 2]))
    assert not torch.count_nonzero(windows[0, :-1])
    assert not torch.count_nonzero(windows[1, :-3])
    a, b = LocalEnvironmentGRU(42), LocalEnvironmentGRU(42)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert a(windows, env)[1].shape == (2, 56)


@pytest.mark.skipif(not Path("data/processed/mississippi_graph_graphfix_st357.pt").exists(),
                    reason="local ST357 dataset absent")
def test_real_fold_excludes_all_receiving_doc():
    masks = np.load("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split142.npz")
    cells = masks["train"]
    folds = station_folds(cells, 654, 42)
    assert len(folds) == 5
    for held in folds:
        fit = cells[~np.isin(cells//654, held)]
        assert not set(fit//654) & set(held)
        assert not set(np.unique(cells//654)) & set(masks["test"]//654)
