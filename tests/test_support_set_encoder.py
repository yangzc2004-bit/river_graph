"""Tests for the support-set residual encoder (spec v2)."""

from __future__ import annotations

import torch

from river_graph.models.support_set_encoder import (
    N_DIR,
    QUERY_DIM,
    REGIME_DIM,
    SUPPORT_DIM,
    SupportSetEncoder,
    fit_residual,
)


def _inputs(k: int = 3, q: int = 2, hop_cap: int = 10, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    sup = torch.randn(k, SUPPORT_DIM, generator=g)
    qry = torch.randn(q, QUERY_DIM, generator=g)
    direction = torch.randint(0, N_DIR, (k, q), generator=g)
    hop = torch.randint(0, hop_cap + 2, (k, q), generator=g)
    s_ord = torch.randint(1, 8, (k,), generator=g).float()
    q_ord = torch.randint(1, 8, (q,), generator=g).float()
    s_area = torch.rand(k, generator=g)
    q_area = torch.rand(q, generator=g)
    s_reg = torch.randn(k, REGIME_DIM, generator=g)
    q_reg = torch.randn(q, REGIME_DIM, generator=g)
    return sup, direction, hop, s_ord, s_area, s_reg, qry, q_ord, q_area, q_reg


def test_k0_is_exact_zero():
    model = SupportSetEncoder()
    model.eval()
    args = list(_inputs(k=0, q=4))
    args[0] = torch.zeros(0, SUPPORT_DIM)
    args[1] = torch.zeros(0, 4, dtype=torch.long)
    args[2] = torch.zeros(0, 4, dtype=torch.long)
    args[3] = torch.zeros(0)
    args[4] = torch.zeros(0)
    args[5] = torch.zeros(0, REGIME_DIM)
    out = model(*args)
    assert out.shape == (4,)
    assert torch.equal(out, torch.zeros(4))


def test_zero_init_head_starts_at_zero():
    model = SupportSetEncoder()
    model.eval()
    with torch.no_grad():
        out = model(*_inputs(k=3, q=2))
    assert torch.allclose(out, torch.zeros(2), atol=1e-7)


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    model = SupportSetEncoder(hidden=16)
    out = model(*_inputs(k=5, q=3))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_backward_reaches_all_submodules():
    torch.manual_seed(0)
    model = SupportSetEncoder(hidden=16)
    args = _inputs(k=4, q=2)
    base = torch.zeros(2)
    y = torch.randn(2)
    loss = fit_residual(model, *args, base_log=base, y_log=y)
    loss.backward()
    named = dict(model.named_parameters())
    assert named["mlp_out.3.weight"].abs().sum() == 0  # still zero-init
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    for name in (
        "mlp_sup.0.weight",
        "mlp_qry.0.weight",
        "hop_emb.weight",
        "dir_emb.weight",
        "bias_lin.weight",
        "mlp_out.3.weight",
    ):
        assert name in grads, name
    assert named["mlp_out.3.weight"].grad.abs().sum() > 0
    assert all(torch.isfinite(g).all() for g in grads.values())


def test_support_changes_delta():
    torch.manual_seed(0)
    model = SupportSetEncoder(hidden=16)
    # break zero-init so the head transmits signal
    torch.nn.init.normal_(model.mlp_out[-1].weight, std=0.1)
    model.eval()
    args_a = _inputs(k=3, q=2, seed=1)
    args_b = list(args_a)
    args_b[0] = args_a[0] + 1.0  # different support values
    with torch.no_grad():
        a = model(*args_a)
        b = model(*args_b)
    assert not torch.allclose(a, b, atol=1e-6)


def test_direction_changes_delta():
    torch.manual_seed(0)
    model = SupportSetEncoder(hidden=16)
    torch.nn.init.normal_(model.mlp_out[-1].weight, std=0.1)
    model.eval()
    args_a = _inputs(k=3, q=2, seed=2)
    args_b = list(args_a)
    args_b[1] = (args_a[1] + 1) % N_DIR
    with torch.no_grad():
        a = model(*args_a)
        b = model(*args_b)
    assert not torch.allclose(a, b, atol=1e-6)
