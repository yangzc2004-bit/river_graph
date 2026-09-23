"""Phase-2A contract tests (docs/paper/phase2_ablation_spec.md).

Freezes what the same-protocol ablation must mean before any of it runs:
arm definitions and information-set parity, no-message behaviour (empty edge
tensor, edge-attribute independence, retained self path, inputs identical to
H2X), the ecological tabular feature set with self-label exclusion, provenance
config-hash distinctness, and a short-training smoke on real masks when the
local ST-core dataset exists. No scientific conclusion is produced here.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.baselines.baselines import (
    EcoMLP,
    EcoRandomForest,
    ecological_tabular_features,
)
from river_graph.experiments.prediction_sources import sha256_file
from river_graph.experiments.predictions import save_predictions
from river_graph.experiments.provenance import build_meta
from river_graph.models.gcn import GCNDocModel
from river_graph.models.hydro import GatedDirectedConv, TransportGCNImputer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs" / "phase2_ablation_stcore_v1.json").read_text(encoding="utf-8")
)
DATASET_PATH = ROOT / CONFIG["dataset"]["path"]
MASKS_DIR = ROOT / CONFIG["masks"]["dir"]
N, T = 6, 4

# arm name -> (architecture, env_groups, env_encoder, edge_set)
EXPECTED_ARMS = {
    "H2": ("transport", ["hydro"], False, "river"),
    "H2E": ("transport", None, False, "river"),
    "H2X": ("transport_enc", None, True, "river"),
    "H2X_nomsg": ("transport_enc", None, True, "empty"),
}


def _toy_dataset() -> dict:
    g = torch.Generator().manual_seed(7)
    y_raw = torch.rand(N, T, generator=g) * 5
    y_mask = torch.zeros(N, T, dtype=torch.bool)
    y_mask[:, :3] = True
    y_mask[0, 3] = True
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return {
        "y": y_raw,
        "y_mask": y_mask,
        "x": torch.rand(N, T, 2, generator=g),
        "x_mask": (torch.rand(N, T, 2, generator=g) > 0.3).float(),
        "months": [f"2020-{m:02d}-01" for m in range(1, T + 1)],
        "static": torch.rand(N, 2, generator=g),
        "regime": torch.randn(N, 13, generator=g),
        "edge_index": ei,
        "edge_attr": torch.randn(ei.shape[1], 6, generator=g),
        "site_no": [f"s{i}" for i in range(N)],
    }


def _toy_split() -> dict:
    obs = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 23]  # flat C-order cells
    return {
        "train": np.array(obs[:8], dtype=np.int64),
        "val": np.array(obs[8:10], dtype=np.int64),
        "test": np.array(obs[10:], dtype=np.int64),
    }


def _model_for(arm: str, **over) -> GCNDocModel:
    arch, env_groups, env_encoder, edge_set = EXPECTED_ARMS[arm]
    kw = {"max_epochs": 1, "patience": 1, **over}
    return GCNDocModel(
        architecture=arch,
        env_groups=env_groups,
        env_encoder=env_encoder,
        edge_set=edge_set,
        **kw,
    )


# ---------------------------------------------------------- config conformance


def test_config_arm_definitions_match_the_frozen_spec():
    arms = CONFIG["arms"]
    assert set(arms) >= set(EXPECTED_ARMS) | {"eco_RF", "eco_MLP"}
    for name, (arch, env_groups, env_encoder, edge_set) in EXPECTED_ARMS.items():
        a = arms[name]
        assert a["architecture"] == arch, name
        assert a["env_groups"] == env_groups, name
        assert a["env_encoder"] is env_encoder, name
        assert a["edge_set"] == edge_set, name
    assert arms["H2X_nomsg"]["paper_name"] == "no-message control"
    assert "removed" in arms["H2X_nomsg"]["semantics"]  # self path retained
    assert arms["H2E"]["node_input_channels"] == 23
    assert arms["H2X"]["node_input_channels"] == 14
    assert arms["H2X"]["env_encoder_input_cols"] == list(range(4, 13))


def test_config_output_rules_keep_historical_dirs_read_only():
    out = CONFIG["output"]
    assert out["dir"] == "experiments/phase2_ablation_stcore_v1/"
    for legacy in (
        "experiments/results/",
        "experiments/predictions/",
        "experiments/frozen_results/",
    ):
        assert legacy in out["never_write"]


def test_config_stage_budget_is_preflight_bounded():
    pre = CONFIG["stages"]["preflight_2a"]
    assert pre["seeds"] == [42]
    assert pre["claims_allowed"] is False
    assert CONFIG["stages"]["pilot_2b"]["claims_allowed"] is False
    assert CONFIG["stages"]["final_2c"]["claims_allowed"] is True
    assert CONFIG["stages"]["final_2c"]["configs"] == 240
    assert len(CONFIG["masks"]["key_masks"]) == 8


def test_config_dataset_hash_matches_the_frozen_file():
    if not DATASET_PATH.is_file():
        pytest.skip(f"local ST-core dataset missing: {DATASET_PATH}")
    assert sha256_file(DATASET_PATH) == CONFIG["dataset"]["sha256"]


# ------------------------------------------------------------- input parity


@pytest.mark.parametrize(
    "arm,channels,env_cols",
    [("H2", 14, 0), ("H2E", 23, 0), ("H2X", 14, 9), ("H2X_nomsg", 14, 9)],
)
def test_arm_input_dimensions_match_the_spec(arm, channels, env_cols):
    xt, _y, _bv, _tc, _st, env_raw = _model_for(arm)._build_inputs(
        _toy_dataset(), _toy_split()
    )
    assert xt.shape[-1] == channels, arm
    assert (env_raw.shape[1] if env_raw is not None else 0) == env_cols, arm


def test_h2x_nomsg_inputs_are_identical_to_h2x():
    """edge_set must not change the information set — only the messages."""
    ds, split = _toy_dataset(), _toy_split()
    a = _model_for("H2X")._build_inputs(ds, split)
    b = _model_for("H2X_nomsg")._build_inputs(ds, split)
    for t1, t2 in zip(a, b, strict=True):
        if torch.is_tensor(t1):
            assert torch.equal(t1, t2)
        elif t1 is None or t2 is None:
            assert t1 is t2
        else:
            assert np.array_equal(np.asarray(t1), np.asarray(t2))


# ------------------------------------------------------- no-message behaviour


def test_empty_edges_zero_messages_and_keep_self_path():
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    conv = GatedDirectedConv(8, 8, edge_dim=6)
    empty = torch.empty((2, 0), dtype=torch.long)
    ea = torch.randn(3, 6)
    out = conv(x, empty, ea)
    assert torch.allclose(out, conv.self_lin(x))
    # output must not depend on edge attributes at all
    out2 = conv(x, empty, torch.randn(3, 6) * 100)
    assert torch.equal(out, out2)
    # self path is effective: output changes with input
    assert not torch.allclose(out, conv(x + 1.0, empty, ea))


def test_transport_model_with_empty_edges_ignores_edge_attr():
    torch.manual_seed(0)
    x = torch.randn(5, 14)
    env = torch.randn(5, 9)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    model = TransportGCNImputer(14, 6, hidden=8, layers=2, dropout=0.0, env_dim=9)
    empty = torch.empty((2, 0), dtype=torch.long)
    base = model(x, empty, torch.randn(3, 6), env)
    for _ in range(3):
        assert torch.equal(base, model(x, empty, torch.randn(3, 6), env))
    # with real edges the edge attributes do matter
    full = model(x, ei, torch.randn(3, 6), env)
    assert not torch.allclose(full, model(x, ei, torch.randn(3, 6), env))


def test_gcn_model_sends_empty_edge_tensor_when_edge_set_empty(monkeypatch):
    from river_graph.models import hydro

    seen: list[int] = []

    class Recording(TransportGCNImputer):
        def forward(self, x, edge_index, edge_attr, env_raw=None):
            seen.append(edge_index.shape[1])
            return super().forward(x, edge_index, edge_attr, env_raw)

    monkeypatch.setattr(hydro, "TransportGCNImputer", Recording)
    ds, split = _toy_dataset(), _toy_split()
    _model_for("H2X_nomsg").fit(ds, split)
    assert seen and all(k == 0 for k in seen)
    seen.clear()
    _model_for("H2X").fit(ds, split)
    assert seen and all(k > 0 for k in seen)


# --------------------------------------------------------- eco feature set


def test_ecological_features_shape_names_and_self_exclusion():
    ds, split = _toy_dataset(), _toy_split()
    feats, names = ecological_tabular_features(ds, split)
    assert feats.shape == (N * T, 23)
    assert names[:2] == ["temp", "temp_avail"]
    assert names[-2:] == ["month_visdoc_mean_excl_self", "month_visdoc_count_excl_self"]

    # a cell's own DOC must never appear in its own feature row
    ds2 = {**ds, "y": ds["y"].clone()}
    ds2["y"][1, 2] = 999.0  # hijack one observed cell (flat idx 6 = row 1, t 2)
    feats2, _ = ecological_tabular_features(ds2, split)
    row = 1 * T + 2
    assert np.allclose(feats[row], feats2[row])
    # ...but other rows of the same month do see it through the aggregate
    other = 0 * T + 2
    assert feats[other, -2] != feats2[other, -2]


def test_synthetic_forward_every_arm():
    ds, split = _toy_dataset(), _toy_split()
    for arm in EXPECTED_ARMS:
        pred = _model_for(arm).fit_predict(ds, split)
        assert pred.shape == (N, T), arm
        assert np.isfinite(pred).all(), arm
    for cls in (EcoRandomForest, EcoMLP):
        kwargs = {"seed": 0}
        if cls is EcoMLP:
            kwargs.update(hidden=(16,), max_epochs=8, patience=3)
        pred = cls(**kwargs).fit_predict(ds, split)
        assert pred.shape == (N, T)
        assert np.isfinite(pred.ravel()[split["test"]]).all()
        outside = np.setdiff1d(np.arange(N * T), split["test"])
        assert np.isnan(pred.ravel()[outside]).all()


# ------------------------------------------------- real-mask short training


@pytest.mark.skipif(
    not DATASET_PATH.is_file() or not (MASKS_DIR / "e2b_partial.npz").is_file(),
    reason="local ST-core dataset or masks_stcore_v1 not built",
)
def test_short_training_on_real_masks_no_nan_and_full_test_coverage():
    import torch as _torch

    ds = _torch.load(DATASET_PATH, weights_only=False)
    for mask_name in CONFIG["stages"]["preflight_2a"]["masks"]:
        split = dict(np.load(MASKS_DIR / f"{mask_name}.npz"))
        test_cells = np.asarray(split["test"], dtype=np.int64)
        for arm in EXPECTED_ARMS:
            model = _model_for(arm, max_epochs=3, patience=2)
            xt, *_ = model._build_inputs(ds, split)
            want = CONFIG["arms"][arm]["node_input_channels"]
            assert xt.shape[-1] == want, (mask_name, arm)
            pred = model.fit_predict(ds, split)
            vals = pred.ravel()[test_cells]
            assert np.isfinite(vals).all(), (mask_name, arm)
            assert len(vals) == len(test_cells)
        for cls in (EcoRandomForest, EcoMLP):
            kwargs = {"seed": 42}
            if cls is EcoMLP:
                kwargs.update(hidden=(32,), max_epochs=15, patience=3)
            pred = cls(**kwargs).fit_predict(ds, split)
            vals = np.asarray(pred).ravel()[test_cells]
            assert np.isfinite(vals).all(), (mask_name, cls.__name__)
            assert len(vals) == len(test_cells)


@pytest.mark.skipif(
    not DATASET_PATH.is_file() or not (MASKS_DIR / "e2b_partial.npz").is_file(),
    reason="local ST-core dataset or masks_stcore_v1 not built",
)
def test_provenance_config_hashes_are_distinct_and_never_conflict(tmp_path):
    ds = _torch_load()
    split = dict(np.load(MASKS_DIR / "e2b_partial.npz"))
    meta_by_arm = {}
    for arm, (arch, env_groups, env_encoder, edge_set) in EXPECTED_ARMS.items():
        params = {
            "script": "tests/test_phase2_controls.py",
            "tag": arm,
            "architecture": arch,
            "variant": "river",
            "seed": 42,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "edge_dropout": 0.0,
            "share_weights": False,
            "env_groups": env_groups,
            "env_encoder": env_encoder,
        }
        meta = build_meta(
            model_name=f"P2X_{arm}",
            mask_name="e2b_partial",
            dataset_path=DATASET_PATH,
            split=split,
            params=params,
            masks_dir=MASKS_DIR,
            caller="tests/test_phase2_controls.py",
        )
        meta_by_arm[arm] = meta
        pred = _model_for(arm, max_epochs=1, patience=1).fit_predict(ds, split)
        save_predictions(
            pred, ds, split,
            model=f"P2X_{arm}", mask_name="e2b_partial",
            dataset_version="graphfix_st357",
            out_dir=tmp_path, meta=meta,
        )
    hashes = [m["config_hash"] for m in meta_by_arm.values()]
    assert len(set(hashes)) == len(hashes), "arms must not share a config hash"


def _torch_load():
    import torch as _torch

    return _torch.load(DATASET_PATH, weights_only=False)
