"""Meaningful data isolation, matched capacity and temporal loss controls."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.models.doc_dynamic_reconstruction import (
    DynamicRiverDOC,
    change_metrics,
    dynamic_inputs,
    label_scales,
    reconstruction_loss,
)
from river_graph.models.river_architecture_comparison import covariate_inputs
from river_graph.models.river_relation_comparison import (
    RiverRelationModel,
    relation_graph,
)


def toy():
    n,t=4,15
    return {"x":np.ones((n,t,2)),"x_mask":np.ones((n,t,2),bool),
            "feature_channels":["temperature","discharge"],
            "months":np.asarray([str(m) for m in __import__('pandas').period_range('2001-01',periods=t,freq='M')]),
            "regime":np.column_stack([np.ones(n),np.zeros((n,12))]),"static":np.zeros((n,2))}


def test_zero_extra_channels_reproduce_original_initial_forward():
    torch.manual_seed(42);base=RiverRelationModel("direction_heads",dropout=0.)
    torch.manual_seed(42);expanded=DynamicRiverDOC(dropout=0.)
    original=covariate_inputs(toy(),[0,1]);inputs=dynamic_inputs(original,np.zeros((4,15,8)),"monthly")
    graph=relation_graph(np.asarray([[0,1,2],[1,2,3]]),original["order"],np.arange(4),"direction_heads")
    base.eval();expanded.eval()
    with torch.no_grad():
        expected=base(original["windows"],original["environment"],original["season"],graph)
        actual=expanded(inputs["windows"],inputs["environment"],inputs["season"],graph)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    assert sum(p.numel() for p in expanded.parameters())==12853


def test_daily_history_is_causal_and_control_hides_extra_channels():
    original=covariate_inputs(toy(),[0,1]);daily=np.ones((4,15,8),np.float32)*.1
    current=dynamic_inputs(original,daily,"daily")
    changed=daily.copy();changed[:,10:]=.9
    other=dynamic_inputs(original,changed,"daily")
    torch.testing.assert_close(current["windows"][:10],other["windows"][:10])
    control=dynamic_inputs(original,daily,"monthly")
    assert not control["windows"][...,8:].any()
    assert current["windows"][0,:,0,8:].eq(0).all()


def test_joint_loss_ignores_hidden_truth_and_only_pairs_visible_training_labels():
    truth=torch.tensor([[1.,float('nan')],[2.,3.],[4.,float('nan')]])
    scales=label_scales(truth)
    assert scales["training_adjacent_pairs"]==2
    z=torch.zeros((2,2),requires_grad=True)
    prev=torch.zeros((2,2),requires_grad=True)
    loss=reconstruction_loss(z,truth[1:],prev,truth[:-1],scales,{"cells":2.,"pairs":1.},"joint")
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and torch.isfinite(prev.grad).all()
    assert z.grad[1,1]==0 and prev.grad[:,1].eq(0).all()


def test_changes_remove_offsets_but_not_temporal_signal():
    import pandas as pd
    frame=pd.DataFrame({"station":[0,0,0,1,1,1],"month":[0,1,2,0,1,2],
                        "y_true":[1.,2.,4.,10.,12.,14.],"y_pred":[6.,7.,9.,5.,7.,9.]})
    result=change_metrics(frame)
    assert result["station_centered_r2"]==pytest.approx(1.)
    assert result["monthly_delta_mae"]==0 and result["monthly_pairs"]==4


@pytest.mark.skipif(not Path("data/processed/mississippi_graph_graphfix_st357.pt").exists(),reason="local ST357 dataset not available")
def test_frozen_protocol_keeps_original_endpoints_and_masks_if_present():
    import json
    p=Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1/protocol.json")
    if not p.exists():pytest.skip("dynamic reconstruction protocol not frozen yet")
    protocol=json.loads(p.read_text())
    assert protocol["joint_primary"]==["mae","log1p_mae"]
    assert protocol["prior_paper_endpoints_modified"] is False
    assert len(protocol["tasks"])==192
