"""T07: learnability, initialisation and boundary behaviour of the H3-A model."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from river_graph.models.h3 import (
    ENV_FEATURE_NAMES,
    EnvironmentalPredictor,
    H3ResidualImputer,
    build_env_features,
    env_feature_index,
    make_h2x_trunk,
    reset_linear_parameters,
)
from river_graph.models.hydro import TransportGCNImputer

N_REGIME = 13
EDGE_DIM = 6
H2X_IN = 14
H2X_ENV_DIM = 9


def tiny_dataset(n: int = 6, t: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    return {
        "site_no": [f"s{i}" for i in range(n)],
        "months": [f"2000-{m + 1:02d}" for m in range(t)],
        "edge_index": edge_index,
        "edge_attr": torch.rand(edge_index.shape[1], EDGE_DIM, generator=g),
        "y": torch.rand(n, t, generator=g) * 4 + 0.5,
        "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.rand(n, t, 2, generator=g) * 10,
        "x_mask": torch.ones(n, t, 2, dtype=torch.bool),
        "static": torch.rand(n, 2, generator=g),
        "regime": torch.rand(n, N_REGIME, generator=g),
        "feature_channels": ["temperature", "discharge"],
    }


def tiny_split(n: int = 6, t: int = 4) -> dict:
    train = np.arange(n * t - 3, dtype=np.int64)
    return {
        "train": train,
        "val": np.array([n * t - 3, n * t - 2], dtype=np.int64),
        "val_context": np.array([n * t - 1], dtype=np.int64),
        "test": np.array([0], dtype=np.int64),
    }


# --------------------------------------------------------------- ENV view


def test_env_feature_names_and_order_are_frozen():
    assert len(ENV_FEATURE_NAMES) == 21
    assert ENV_FEATURE_NAMES[:8] == (
        "temp_std",
        "temp_missing",
        "flow_std",
        "flow_missing",
        "season_sin",
        "season_cos",
        "lat_std",
        "lon_std",
    )
    assert ENV_FEATURE_NAMES[8:] == tuple(f"regime_{i}" for i in range(N_REGIME))
    # columns 8 and 9 of the graph tensor are the DOC slots and must be absent
    assert env_feature_index() == list(range(8)) + list(range(10, 23))
    assert 8 not in env_feature_index() and 9 not in env_feature_index()


def test_env_view_matches_independent_elementwise_expectation():
    dataset, split = tiny_dataset(n=7, t=5), tiny_split(n=7, t=5)
    n, t = dataset["y"].shape
    env = build_env_features(dataset, split)
    assert env.shape == (t, n, 21)

    train = split["train"]
    ti, tj = train // t, train % t

    def cell_std(v):
        return (v - v[ti, tj].mean()) / (v[ti, tj].std() + 1e-8)

    def node_std(v):
        return (v - v.mean()) / (v.std() + 1e-8)

    x = dataset["x"]
    static = dataset["static"]
    regime = dataset["regime"]
    expected = [
        cell_std(x[:, :, 0]),
        dataset["x_mask"][:, :, 0].float(),
        cell_std(x[:, :, 1]),
        dataset["x_mask"][:, :, 1].float(),
        torch.sin(2 * np.pi * torch.arange(1, t + 1) / 12).unsqueeze(0).expand(n, t),
        torch.cos(2 * np.pi * torch.arange(1, t + 1) / 12).unsqueeze(0).expand(n, t),
        node_std(static[:, 0:1]).expand(n, t),
        node_std(static[:, 1:2]).expand(n, t),
        *[
            ((regime - regime.mean(0)) / (regime.std(0) + 1e-8))[:, c : c + 1].expand(
                n, t
            )
            for c in range(N_REGIME)
        ],
    ]
    for c, want in enumerate(expected):
        got = env[:, :, c].T
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_env_view_ignores_doc_values_and_visibility():
    dataset, split = tiny_dataset(), tiny_split()
    env = build_env_features(dataset, split)
    changed = dict(dataset)
    changed["y"] = torch.rand_like(dataset["y"]) * 100
    changed["y_mask"] = ~dataset["y_mask"]
    torch.testing.assert_close(build_env_features(changed, split), env, rtol=0, atol=0)


def test_env_view_does_not_modify_inputs():
    dataset, split = tiny_dataset(), tiny_split()
    before = {k: v.clone() for k, v in dataset.items() if torch.is_tensor(v)}
    build_env_features(dataset, split)
    for key, value in before.items():
        torch.testing.assert_close(dataset[key], value, rtol=0, atol=0)


# ----------------------------------------------------------- initialisation


def test_explicit_init_reproduces_pytorch_default_init():
    """Reset from a generator must equal building under torch.manual_seed."""
    torch.manual_seed(1234)
    reference = TransportGCNImputer(
        H2X_IN, EDGE_DIM, hidden=64, layers=2, dropout=0.1, env_dim=H2X_ENV_DIM,
        env_emb=32, gate_mode="static",
    )
    redrawn = TransportGCNImputer(
        H2X_IN, EDGE_DIM, hidden=64, layers=2, dropout=0.1, env_dim=H2X_ENV_DIM,
        env_emb=32, gate_mode="static",
    )
    reset_linear_parameters(redrawn, torch.Generator().manual_seed(1234))
    a, b = reference.state_dict(), redrawn.state_dict()
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=0, atol=0)


def test_env_and_h3a_base_start_identical():
    seed = 7
    env_model = EnvironmentalPredictor(21, 64, 2, 0.1)
    env_model.reset_parameters(torch.Generator().manual_seed(seed))
    h3a = H3ResidualImputer(
        EnvironmentalPredictor(21, 64, 2, 0.1),
        make_h2x_trunk(H2X_IN, EDGE_DIM, 64, 2, 0.1, H2X_ENV_DIM, 32, "static"),
    )
    h3a.initialise(seed, seed + 200003)
    a, b = env_model.state_dict(), h3a.base.state_dict()
    assert set(a) == set(b)
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=0, atol=0)


def test_correction_head_is_exactly_zero_and_correction_is_zero():
    model = H3ResidualImputer(
        EnvironmentalPredictor(21, 64, 2, 0.1),
        make_h2x_trunk(H2X_IN, EDGE_DIM, 64, 2, 0.1, H2X_ENV_DIM, 32, "static"),
    )
    info = model.initialise(3, 200006)
    assert info["correction_head_zeroed"] is True
    head = model.correction.head
    assert torch.equal(head.weight, torch.zeros_like(head.weight))
    assert torch.equal(head.bias, torch.zeros_like(head.bias))
    env = torch.randn(6, 21)
    x = torch.randn(6, H2X_IN)
    edge_attr = torch.randn(4, EDGE_DIM)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    base, correction, total = model.forward_components(
        env, x, edges, edge_attr, torch.randn(6, H2X_ENV_DIM)
    )
    assert torch.equal(correction, torch.zeros(6))
    assert torch.equal(total, base)


@pytest.mark.parametrize("dropout", [0.1, 0.0])
def test_initial_total_equals_env_exactly(dropout):
    """The frozen acceptance rule: at init, eval total == same-seed ENV output.

    With dropout switched off the equality also holds in train mode, which
    shows the zero head -- not the absence of dropout -- is what makes the two
    branches agree.
    """
    seed = 11
    env_model = EnvironmentalPredictor(21, 64, 2, dropout)
    env_model.reset_parameters(torch.Generator().manual_seed(seed))
    h3a = H3ResidualImputer(
        EnvironmentalPredictor(21, 64, 2, dropout),
        make_h2x_trunk(H2X_IN, EDGE_DIM, 64, 2, dropout, H2X_ENV_DIM, 32, "static"),
    )
    h3a.initialise(seed, seed + 200003)
    env = torch.randn(6, 21)
    x = torch.randn(6, H2X_IN)
    edge_attr = torch.randn(4, EDGE_DIM)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    modes = ("eval", "train") if dropout == 0.0 else ("eval",)
    for mode in modes:
        getattr(env_model, mode)()
        getattr(h3a, mode)()
        with torch.no_grad():
            want = env_model(env)
            base, correction, total = h3a.forward_components(
                env, x, edges, edge_attr, torch.randn(6, H2X_ENV_DIM)
            )
        torch.testing.assert_close(total, want, rtol=0, atol=0)
        torch.testing.assert_close(base, want, rtol=0, atol=0)
        assert torch.equal(correction, torch.zeros(6))


# ----------------------------------------------------------------- ENV arm


def test_env_forward_shape_and_isolated_node():
    model = EnvironmentalPredictor(21, 32, 2, 0.0)
    model.eval()
    env = torch.randn(5, 21)
    out = model(env)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()
    # an isolated node is just another row: no edge information exists anywhere
    # (batch-of-one kernels differ in the last bits, hence the small tolerance)
    torch.testing.assert_close(model(env[:1]), out[:1], rtol=1e-6, atol=1e-6)


def test_env_output_is_independent_of_other_nodes():
    model = EnvironmentalPredictor(21, 32, 2, 0.0)
    model.eval()
    env = torch.randn(5, 21)
    with torch.no_grad():
        base = model(env)
        moved = env.clone()
        moved[1:] = torch.randn(4, 21) * 100
        after = model(moved)
    torch.testing.assert_close(after[0], base[0], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(after[1:], base[1:])
    # wrong width is refused rather than silently broadcast
    with pytest.raises(ValueError):
        model(torch.randn(5, 20))


# -------------------------------------------------------- H3A learnability


def _h3a(dropout: float = 0.0, edges: str = "river") -> H3ResidualImputer:
    model = H3ResidualImputer(
        EnvironmentalPredictor(21, 32, 2, dropout),
        make_h2x_trunk(H2X_IN, EDGE_DIM, 32, 2, dropout, H2X_ENV_DIM, 16, "static"),
        edge_set=edges,
    )
    model.initialise(5, 200008)
    return model


def _graph():
    return (
        torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        torch.randn(5, EDGE_DIM),
    )


def test_first_backward_gives_head_gradient_but_no_backbone_gradient():
    torch.manual_seed(0)
    model = _h3a()
    model.train()
    env = torch.randn(6, 21)
    x = torch.randn(6, H2X_IN)
    edges, edge_attr = _graph()
    target = torch.randn(6)
    _, _, total = model.forward_components(
        env, x, edges, edge_attr, torch.randn(6, H2X_ENV_DIM)
    )
    F.mse_loss(total, target).backward()

    head = dict(model.correction.named_parameters())
    assert head["head.weight"].grad is not None
    assert torch.any(head["head.weight"].grad != 0)
    backbone = [
        (name, p.grad)
        for name, p in model.correction.named_parameters()
        if not name.startswith("head.")
    ]
    assert backbone
    for name, grad in backbone:
        assert grad is None or torch.all(grad == 0), name
    # the base branch is untouched by the zero head and learns from step one
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.base.parameters()
    )


def test_correction_backbone_gets_gradient_from_the_second_step():
    torch.manual_seed(1)
    model = _h3a()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    env = torch.randn(6, 21)
    x = torch.randn(6, H2X_IN)
    edges, edge_attr = _graph()
    env_raw = torch.randn(6, H2X_ENV_DIM)
    target = torch.randn(6)

    def grads():
        model.train()
        _, _, total = model.forward_components(env, x, edges, edge_attr, env_raw)
        optimizer.zero_grad()
        F.mse_loss(total, target).backward()
        return [
            (name, p.grad)
            for name, p in model.correction.named_parameters()
            if not name.startswith("head.")
        ]

    first = grads()
    assert all(g is None or torch.all(g == 0) for _, g in first)
    optimizer.step()
    second = grads()
    assert any(g is not None and torch.any(g != 0) for _, g in second)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_correction_head_learns_positive_and_negative_corrections(sign):
    torch.manual_seed(2)
    model = _h3a()
    model.eval()
    env = torch.randn(24, 21)
    x = torch.randn(24, H2X_IN)
    edges, edge_attr = _graph()
    env_raw = torch.randn(24, H2X_ENV_DIM)
    with torch.no_grad():
        base = model.base(env)
    target = base + sign * 0.4
    # freeze the base branch so this test isolates the correction head's
    # ability to learn a signed offset
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(model.correction.parameters(), lr=0.05)
    model.train()
    start = None
    for _ in range(120):
        _, correction, total = model.forward_components(
            env, x, edges, edge_attr, env_raw
        )
        if start is None:
            start = float(correction.detach().mean())
        loss = F.mse_loss(total, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, correction, total = model.forward_components(
            env, x, edges, edge_attr, env_raw
        )
    assert start == pytest.approx(0.0, abs=1e-12)
    mean = float(correction.mean())
    assert np.sign(mean) == np.sign(sign)
    assert abs(mean) > 0.1
    assert float(F.mse_loss(total, target)) < 1e-4
    assert mean == pytest.approx(sign * 0.4, abs=0.05)


def test_empty_edges_are_finite_and_keep_the_self_path():
    torch.manual_seed(3)
    model = _h3a(edges="empty")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    env = torch.randn(8, 21)
    x = torch.randn(8, H2X_IN)
    edges, edge_attr = _graph()
    env_raw = torch.randn(8, H2X_ENV_DIM)
    target = torch.randn(8)
    model.train()
    for _ in range(5):
        _, correction, total = model.forward_components(
            env, x, edges, edge_attr, env_raw
        )
        assert torch.isfinite(total).all() and torch.isfinite(correction).all()
        loss = F.mse_loss(total, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, correction, _ = model.forward_components(
            env, x, edges, edge_attr, env_raw
        )
    # the self path is still active, so the no-message correction is not
    # automatically zero: this is why the empty-graph control has to be
    # trained separately rather than read off the full model.
    assert torch.isfinite(correction).all()
    assert torch.any(correction != 0)


def test_neighbour_information_changes_the_graph_branch():
    torch.manual_seed(4)
    model = _h3a()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    env = torch.randn(8, 21)
    x = torch.randn(8, H2X_IN)
    edges, edge_attr = _graph()
    env_raw = torch.randn(8, H2X_ENV_DIM)
    target = torch.randn(8)
    model.train()
    for _ in range(8):
        _, _, total = model.forward_components(env, x, edges, edge_attr, env_raw)
        loss = F.mse_loss(total, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    neighbours = [1, 2, 3, 4, 5]
    with torch.no_grad():
        _, before, _ = model.forward_components(env, x, edges, edge_attr, env_raw)
        moved = x.clone()
        moved[neighbours] += 3.0
        _, after, _ = model.forward_components(env, moved, edges, edge_attr, env_raw)
    assert not torch.allclose(before[0], after[0])

    # the same perturbation cannot move the empty-edge variant's node 0 more
    # than through its own row, which was left untouched
    empty = _h3a(edges="empty")
    empty.load_state_dict(model.state_dict())
    empty.eval()
    with torch.no_grad():
        _, base_before, _ = empty.forward_components(
            env, x, edges, edge_attr, env_raw
        )
        _, base_after, _ = empty.forward_components(
            env, moved, edges, edge_attr, env_raw
        )
    torch.testing.assert_close(base_after[0], base_before[0], rtol=0, atol=0)


def test_components_are_added_in_log_space_only():
    torch.manual_seed(5)
    model = _h3a()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    env = torch.randn(8, 21)
    x = torch.randn(8, H2X_IN)
    edges, edge_attr = _graph()
    env_raw = torch.randn(8, H2X_ENV_DIM)
    model.train()
    for _ in range(5):
        _, _, total = model.forward_components(env, x, edges, edge_attr, env_raw)
        loss = F.mse_loss(total, torch.randn(8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        base, correction, total = model.forward_components(
            env, x, edges, edge_attr, env_raw
        )
    torch.testing.assert_close(total, base + correction, rtol=0, atol=0)
    # expm1 must be applied once to the total, never to each branch separately
    once = torch.expm1(total)
    split = torch.expm1(base) + torch.expm1(correction)
    assert not torch.allclose(once, split)


def test_h3a_parameter_count_is_base_plus_correction():
    model = _h3a()
    total = sum(p.numel() for p in model.parameters())
    base = sum(p.numel() for p in model.base.parameters())
    correction = sum(p.numel() for p in model.correction.parameters())
    assert total == base + correction
    assert base > 0 and correction > base
    # the correction branch is exactly the frozen static H2X trunk, so an
    # identically-shaped TransportGCNImputer must have the same size
    trunk = TransportGCNImputer(H2X_IN, EDGE_DIM, hidden=32, layers=2, dropout=0.1,
                                env_dim=H2X_ENV_DIM, env_emb=16, gate_mode="static")
    assert correction == sum(p.numel() for p in trunk.parameters())
