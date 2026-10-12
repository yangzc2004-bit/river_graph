"""Covariance semantics, exact grouped GLS, batch isolation and source contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.linalg import solve

from river_graph.experiments.doc_network_work import (
    PHYST_RECIPES,
    PhySTArchitecture,
    SpatialStreamNetwork,
    kernel,
)
from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import graph_view
from river_graph.topology.river_structure import VAA_COLUMNS, ReachNetwork

SOURCE = Path("cache/doc_network_work_v1/sources/PhyST-HC/scripts/Functions_DOC.py")
ROOT = Path("experiments/phase4_transfer/doc_network_work_v1")


def geometry_example():
    distance = np.array([[0., 2., 1.], [2., 0., 1.], [1., 1., 0.]])
    weights = np.array([[1., 0., np.sqrt(.25)], [0., 1., np.sqrt(.75)], [np.sqrt(.25), np.sqrt(.75), 1.]])
    return {"distance": distance, "tailup_weight": weights, "euclidean": distance}


def test_tailup_masks_siblings_and_taildown_retains_shared_junction():
    geometry = geometry_example()
    up, down = kernel(geometry, "tailup", 10.), kernel(geometry, "taildown", 10.)
    assert up[0, 1] == 0 and down[0, 1] > 0
    assert up[0, 2] == pytest.approx(.5*np.exp(-.1))
    for name in ("tailup", "taildown", "euclidean"):
        assert np.linalg.eigvalsh(kernel(geometry, name, 10.)).min() > 0


def test_grouped_reml_coefficients_and_predictions_equal_dense_gls():
    rng = np.random.default_rng(7)
    nodes = np.repeat(np.arange(3), [12, 15, 9])
    x = rng.normal(size=(len(nodes), 2))
    y = 2+x[:, 0]+np.array([-.3, .5, .1])[nodes]+rng.normal(0, .1, len(nodes))
    model = SpatialStreamNetwork(geometry_example(), ("tailup", "taildown"), maxiter=30).fit(x, y, nodes)
    covariance = model.covariance(model.theta)
    v = covariance[np.ix_(nodes, nodes)]+np.eye(len(nodes))*np.exp(model.theta[-1])
    design = np.column_stack([np.ones(len(x)), x])[:, model.columns]
    beta = solve(design.T @ solve(v, design, assume_a="pos"), design.T @ solve(v, y, assume_a="pos"), assume_a="pos")
    np.testing.assert_allclose(model.beta, beta, atol=1e-7)
    predicted = design @ beta+covariance[np.ix_(nodes, nodes)] @ solve(v, y-design @ beta, assume_a="pos")
    np.testing.assert_allclose(model.predict(x, nodes), predicted, atol=1e-7)
    assert model.info["source_cells"] == len(y)


def test_redundant_source_features_are_removed_without_target_fitting():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 1))
    x = np.column_stack([x, 2*x, np.ones(30)])
    nodes = np.tile(np.arange(3), 10)
    model = SpatialStreamNetwork(geometry_example(), ("taildown",), maxiter=15).fit(x, rng.normal(size=30), nodes)
    assert model.info["mean_rank"] == 2
    assert np.isfinite(model.predict(x, nodes)).all()


@pytest.mark.skipif(not SOURCE.exists(), reason="public PhyST-HC architecture source unavailable")
def test_official_physt_monthly_panels_are_disjoint_and_inductive():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = PhySTArchitecture(SOURCE, PHYST_RECIPES["compact"]).eval()
    graph = graph_view(np.array([[0, 1], [2, 2]]), np.array([1, 1, 2]), np.arange(3))
    windows, env, season = torch.randn(2, 3, 12, 8), torch.randn(3, 15), torch.zeros(2, 2)
    with torch.no_grad():
        together = model(windows, env, season, graph)
        separate = torch.cat([model(windows[i:i+1], env, season[i:i+1], graph) for i in range(2)])
        changed = windows.clone()
        changed[1] = 100
        other = model(changed, env, season, graph)
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(together[0], other[0], atol=1e-6, rtol=1e-6)
    assert together.shape == (2, 3) and torch.isfinite(together).all()


def test_nhd_position_junction_distance_and_area_weights():
    spec = importlib.util.spec_from_file_location("prepare_network", Path("scripts/prepare_doc_network_work_v1.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {name: np.zeros(4) for name in VAA_COLUMNS}
    values.update(comid=[10, 20, 30, 40], fromnode=[1, 2, 3, 4], tonode=[3, 3, 4, 5],
        hydroseq=[4, 3, 2, 1], dnhydroseq=[2, 2, 1, 0], lengthkm=[1., 2., 3., 4.],
        totdasqkm=[1., 3., 4., 4.], streamorde=[1, 1, 2, 2])
    network = ReachNetwork(pd.DataFrame(values))
    nodes = pd.DataFrame({"comid": [10, 20, 40], "measure": [50., 50., 50.],
        "dec_lat_va": [40., 40.1, 40.2], "dec_long_va": [-90., -90.1, -90.2]})
    geometry, audit = module.build_geometry(network, nodes)
    assert geometry["distance"][0, 1] == 1.5
    assert geometry["distance"][0, 2] == 5.5
    assert geometry["tailup_weight"][0, 1] == 0
    assert geometry["tailup_weight"][0, 2] == .5
    assert geometry["tailup_weight"][1, 2] == pytest.approx(np.sqrt(.75))
    assert audit["junctions_with_equal_weight_fallback"] == 0


@pytest.mark.skipif(not (ROOT/"protocol.json").exists(), reason="frozen network comparison unavailable")
def test_frozen_protocol_and_data_contract():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    assert protocol["native_reproductions"] is False
    assert protocol["prior_paper_endpoints_modified"] is False
    assert protocol["unique_new_fits"] == 45
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    assert sha256_file("data/processed/mississippi_graph_graphfix_st357.pt") == protocol["dataset_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT/"code_snapshot"/name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
