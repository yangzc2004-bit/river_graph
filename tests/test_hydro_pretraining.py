"""Masked hydro adaptation uses the existing causal backbone without DOC labels."""
import copy

import numpy as np
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.hydro_pretraining import masked_hydro_inputs, pretrain_hydro


def fixture():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        spatial = TransportGCNImputer(5, 2, hidden=4, layers=2, env_dim=2, env_emb=3)
        temporal, decay = nn.GRUCell(5, 4), nn.Linear(4, 4)
    model = EncoderNativeResidual(spatial, temporal, decay, encoder_mode="last_self_ecology",
                                  extra_dim=0, interaction_indices=())
    rng = np.random.default_rng(4)
    inputs = {"raw": rng.uniform(.1, 1., (3, 16, 5)).astype(np.float32),
              "env": rng.uniform(0, 1, (3, 2)).astype(np.float32),
              "age": np.ones((3, 16)), "support": np.zeros((3, 16, 3))}
    hydro = rng.uniform(1, 5, (3, 16, 2))
    return model, inputs, hydro, np.ones_like(hydro, dtype=bool)


def test_mask_removes_value_and_visibility_without_mutating_inputs():
    _, inputs, _, valid = fixture()
    original = copy.deepcopy(inputs)
    hidden = np.zeros_like(valid)
    hidden[1, 3] = True
    result = masked_hydro_inputs(inputs, hidden)
    assert np.all(result["raw"][1, 3, :4] == 0)
    np.testing.assert_array_equal(inputs["raw"], original["raw"])
    np.testing.assert_array_equal(result["raw"][0], original["raw"][0])


def test_hydro_forward_backward_preserves_original_and_doc_head():
    model, inputs, hydro, valid = fixture()
    before = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in ("spatial", "temporal", "head")}
    pretrained, report = pretrain_hydro(model, inputs, hydro, valid, epochs=2, batch_size=16)
    assert report["epochs_run"] == 2
    assert np.isfinite(report["validation_mse"])
    assert report["selection_role"] == "masked_source_hydro"
    for name, state in before.items():
        for key, tensor in state.items():
            torch.testing.assert_close(getattr(model, name).state_dict()[key], tensor, rtol=0, atol=0)
    for key, tensor in before["head"].items():
        torch.testing.assert_close(pretrained.head.state_dict()[key], tensor, rtol=0, atol=0)
