"""H2X smoke tests: clean-commit forward+backward must work.

H2X is ``TransportGCNImputer`` with an ecological context encoder
(``env_dim > 0``), driven through ``GCNDocModel(architecture="transport_enc")``.
These tests lock the ``edge_direction`` gating fix in ``hydro.py`` and prove
that a fresh checkout can train one H2X step without any uncommitted patch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from river_graph.models.gcn import GCNDocModel
from river_graph.models.hydro import GatedDirectedConv, TransportGCNImputer


def _toy_edges(n: int = 6):
    x = torch.randn(n, 10)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # 0->1->2->3->4
    ea = torch.randn(4, 6)
    env = torch.randn(n, 9)  # ecological block (H2X env_raw)
    return x, ei, ea, env


def test_h2x_forward_backward():
    torch.manual_seed(0)
    x, ei, ea, env = _toy_edges()
    model = TransportGCNImputer(
        in_channels=10, edge_dim=6, hidden=16, layers=2, env_dim=9, env_emb=8
    )
    out = model(x, ei, ea, env)
    assert out.shape == (6,)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads)
    # the ecological encoder must receive gradient (it is the H2X increment)
    enc_grads = [p.grad for p in model.env_encoder.parameters()]
    assert enc_grads and all(g is not None for g in enc_grads)
    assert any(g.abs().sum() > 0 for g in enc_grads)


def test_edge_direction_modes_differ_and_validate():
    torch.manual_seed(0)
    x, ei, ea, _env = _toy_edges()
    conv = GatedDirectedConv(10, 16, edge_dim=6)
    conv.eval()
    with torch.no_grad():
        both = conv(x, ei, ea, "both")
        up = conv(x, ei, ea, "upstream")
        down = conv(x, ei, ea, "downstream")
    assert not torch.allclose(both, up, atol=1e-6)
    assert not torch.allclose(both, down, atol=1e-6)
    assert not torch.allclose(up, down, atol=1e-6)
    with pytest.raises(ValueError):
        conv(x, ei, ea, "sideways")


def test_h2x_direction_propagates_through_stack():
    torch.manual_seed(0)
    x, ei, ea, env = _toy_edges()
    outs = {}
    for direction in ("both", "upstream", "downstream"):
        torch.manual_seed(0)
        model = TransportGCNImputer(
            in_channels=10, edge_dim=6, hidden=16, layers=2,
            env_dim=9, env_emb=8, edge_direction=direction,
        )
        model.eval()
        with torch.no_grad():
            outs[direction] = model(x, ei, ea, env)
    assert not torch.allclose(outs["both"], outs["upstream"], atol=1e-6)
    assert not torch.allclose(outs["upstream"], outs["downstream"], atol=1e-6)
    with pytest.raises(ValueError):
        TransportGCNImputer(10, 6, edge_direction="diagonal")


def _tiny_dataset(n: int = 8, t: int = 6, seed: int = 0) -> dict:
    ymask = torch.zeros(n, t, dtype=torch.bool)
    ymask[:, ::2] = True
    y = torch.zeros(n, t)
    y[ymask] = torch.rand(int(ymask.sum()), generator=torch.Generator().manual_seed(seed)) * 5 + 1
    x = torch.randn(n, t, 2, generator=torch.Generator().manual_seed(seed))
    x_mask = torch.ones(n, t, 2, dtype=torch.bool)
    ei = torch.tensor([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]])
    ea = torch.randn(ei.shape[1], 6, generator=torch.Generator().manual_seed(seed))
    regime = torch.randn(n, 13, generator=torch.Generator().manual_seed(seed))
    return {
        "site_no": [f"S{i:03d}" for i in range(n)],
        "months": [f"2020-{j + 1:02d}-01" for j in range(t)],
        "y": y,
        "y_mask": ymask,
        "x": x,
        "x_mask": x_mask,
        "static": torch.stack([torch.linspace(0, 1, n), torch.linspace(1, 0, n)], dim=1),
        "edge_index": ei,
        "edge_attr": ea,
        "regime": regime,
    }


def _tiny_split(dataset: dict, seed: int = 0) -> dict:
    flat = np.flatnonzero(dataset["y_mask"].numpy().ravel())
    rng = np.random.default_rng(seed)
    flat = rng.permutation(flat)
    n_val = max(1, len(flat) // 5)
    return {"train": flat[n_val:], "val": flat[:n_val]}


def test_gcn_doc_model_h2x_fit_predict_smoke():
    dataset = _tiny_dataset()
    split = _tiny_split(dataset)
    model = GCNDocModel(
        architecture="transport_enc",
        variant="river",
        edge_set="river",
        edge_direction="both",
        env_encoder=True,
        hidden=16,
        layers=2,
        max_epochs=3,
        patience=3,
        seed=0,
    )
    model.fit(dataset, split)
    pred = model.predict()
    assert pred.shape == dataset["y"].shape
    assert np.isfinite(pred[dataset["y_mask"].numpy()]).all()


@pytest.mark.parametrize("direction", ["both", "upstream", "downstream"])
def test_gcn_doc_model_accepts_edge_direction(direction):
    dataset = _tiny_dataset()
    split = _tiny_split(dataset)
    model = GCNDocModel(
        architecture="transport_enc",
        edge_direction=direction,
        env_encoder=True,
        hidden=8,
        layers=1,
        max_epochs=1,
        patience=1,
        seed=0,
    )
    model.fit(dataset, split)
    assert model.edge_direction == direction


def test_gcn_doc_model_rejects_bad_edge_direction():
    with pytest.raises(ValueError):
        GCNDocModel(architecture="transport_enc", edge_direction="sideways")
