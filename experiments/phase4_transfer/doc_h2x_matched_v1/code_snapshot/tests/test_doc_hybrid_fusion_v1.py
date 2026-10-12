"""Lag access, local bypass, seeded shared tensors, and frozen input contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.doc_dynamic_reconstruction import (
    DynamicRiverDOC,
    dynamic_inputs,
)
from river_graph.models.hybrid_lag_doc import HybridLagRiverDOC
from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.river_relation_comparison import relation_graph


def inputs():
    n, t = 4, 15
    data = {"x": np.ones((n, t, 2)), "x_mask": np.ones((n, t, 2), bool),
            "feature_channels": ["temperature", "discharge"],
            "months": np.asarray([str(m) for m in __import__('pandas').period_range('2001-01', periods=t, freq='M')]),
            "regime": np.column_stack([np.ones(n), np.zeros((n, 12))]), "static": np.zeros((n, 2))}
    original = covariate_inputs(data, [0, 1])
    return dynamic_inputs(original, np.zeros((n, t, 8), np.float32), "monthly")


def graph(order, reverse=False):
    edges = np.asarray([[0, 1, 2], [1, 2, 3]])
    return relation_graph(edges[::-1] if reverse else edges, order, np.arange(4), "direction_heads")


def test_original_backbone_initial_tensors_are_retained():
    torch.manual_seed(42)
    original = DynamicRiverDOC()
    torch.manual_seed(42)
    hybrid = HybridLagRiverDOC()
    for name in ("environment", "temporal", "combine", "blocks"):
        for a, b in zip(getattr(original, name).parameters(), getattr(hybrid, name).parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert sum(p.numel() for p in original.parameters()) == 12853
    assert sum(p.numel() for p in hybrid.parameters()) == 17617


def test_earliest_month_reaches_explicit_lag_representation():
    model = HybridLagRiverDOC(dropout=0.)
    x = inputs()
    w = x["windows"][12:13].clone().requires_grad_(True)
    local, _ = model.encode_local(w, x["environment"], x["season"][12:13])
    local[..., -model.lag_output:].square().sum().backward()
    assert w.grad[..., 0, :8].abs().sum() > 0
    assert w.grad[..., 8:].eq(0).all()


def test_direct_local_readout_can_be_independent_of_river_messages():
    torch.manual_seed(8)
    model = HybridLagRiverDOC(dropout=0.).eval()
    x = inputs()
    with torch.no_grad():
        model.readout[0].weight[:, model.local_width:] = 0.
        a = model(x["windows"], x["environment"], x["season"], graph(x["order"]))
        b = model(x["windows"], x["environment"], x["season"], graph(x["order"], True))
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_all_three_parts_receive_gradients_and_daily_inputs_are_rejected():
    model = HybridLagRiverDOC(dropout=0.)
    x = inputs()
    result = model(x["windows"], x["environment"], x["season"], graph(x["order"]))
    assert result.shape == (15, 4) and torch.isfinite(result).all()
    result.square().mean().backward()
    for name in ("environment", "temporal", "lag_encoder", "blocks"):
        assert sum(p.grad.abs().sum() for p in getattr(model, name).parameters() if p.grad is not None) > 0
    w = x["windows"].clone()
    w[..., 8] = .1
    with pytest.raises(ValueError, match="monthly covariates only"):
        model(w, x["environment"], x["season"], graph(x["order"]))


@pytest.mark.skipif(not Path('data/processed/mississippi_graph_graphfix_st357.pt').exists(), reason='local ST357 dataset not available')
def test_frozen_hybrid_masks_inputs_and_endpoints_if_present():
    root = Path('experiments/phase4_transfer/doc_hybrid_fusion_v1')
    if not (root / 'protocol.json').exists():
        pytest.skip('hybrid fusion protocol not frozen yet')
    from river_graph.experiments.provenance import sha256_file
    protocol = json.loads((root / 'protocol.json').read_text())
    parent = Path(protocol['parent'])
    old = json.loads((parent / 'protocol.json').read_text())
    assert protocol['settings'] == old['settings']
    assert protocol['joint_primary'] == ['mae', 'log1p_mae']
    assert protocol['dataset_sha256'] == old['dataset_sha256']
    assert protocol['mask_sha256'] == old['mask_sha256']
    assert len(protocol['tasks']) == 96
    assert sum(t['arm'] == 'hybrid_direct' for t in protocol['tasks']) == 24
    assert protocol['uses_doc_inputs'] is False
    for name, h in protocol['mask_sha256'].items():
        assert sha256_file(root / 'masks' / f'{name}.npz') == h
