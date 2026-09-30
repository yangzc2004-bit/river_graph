"""Contracts for the trainable output-space fusion layer."""

import numpy as np
import pytest
import torch

from river_graph.models.unified_fusion import (
    UnifiedSpatiotemporalFusion,
    fuse_transformed,
)


def test_fusion_endpoints_and_linear_interpolation():
    rf = torch.tensor([1.0, 2.0])
    local = torch.tensor([3.0, 6.0])
    np.testing.assert_allclose(fuse_transformed(rf, local, 0.0), rf)
    np.testing.assert_allclose(fuse_transformed(rf, local, 1.0), local)
    np.testing.assert_allclose(fuse_transformed(rf, local, 0.25), [1.5, 3.0])


def test_gate_starts_at_constant_alpha_and_has_gradients():
    model = UnifiedSpatiotemporalFusion(3, init_alpha=0.25)
    rf = torch.zeros(4)
    local = torch.ones(4)
    features = torch.randn(4, 3)
    alpha, fused = model(rf, local, features)
    np.testing.assert_allclose(alpha.detach().numpy(), 0.25, atol=1e-5)
    np.testing.assert_allclose(fused.detach().numpy(), 0.25, atol=1e-5)
    fused.sum().backward()
    assert model.gate.weight.grad is not None


def test_gate_alpha_floor_avoids_boundary_saturation():
    model = UnifiedSpatiotemporalFusion(2, init_alpha=0.01, alpha_floor=0.01,
                                        alpha_ceiling=0.99)
    alpha, _ = model(torch.zeros(3), torch.ones(3), torch.zeros(3, 2))
    np.testing.assert_allclose(alpha.detach().numpy(), 0.01, atol=1e-5)


def test_fusion_rejects_misaligned_inputs():
    model = UnifiedSpatiotemporalFusion(2)
    with pytest.raises(ValueError):
        model(torch.zeros(2), torch.zeros(3), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        model(torch.zeros(2), torch.zeros(2), torch.zeros(3, 2))
