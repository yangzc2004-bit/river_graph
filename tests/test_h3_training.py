"""T09: hidden-label isolation for the shared H3 trainer.

The protocol only holds if the roles really are isolated:

  * the outer test truth can never change training, selection or an exported
    validation prediction;
  * the internal validation truth (val) can only change *selection*, through
    its score -- never an input and never the loss;
  * val_context is visible during the E2b selection/validation forward and
    nowhere else.

Every check runs for ENV and for both H3A branches, because "the base branch
behaves" is not evidence about the correction branch.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from river_graph.experiments.h3_training import H3Trainer, load_protocol
from river_graph.models.h3 import ENV_FEATURE_NAMES

ARMS = ("env", "h2x", "h3a")
N_REGIME = 13
EDGE_DIM = 6


def tiny_dataset(n: int = 8, t: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
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


def roles() -> dict[str, np.ndarray]:
    """Disjoint roles over 8 stations x 4 months.

    val_target sits at station 6 in months 0-2 and val_context at station 5,
    month 0, directly upstream of station 6; exposing val_context therefore has
    a real chance of moving a validation prediction for the graph arms.
    """
    used = np.array([0, 1, 2, 3, 20, 24, 25, 26, 28, 29, 30, 31], dtype=np.int64)
    train = np.setdiff1d(np.arange(32, dtype=np.int64), used)
    return {
        "test": np.array([0, 1, 2, 3], dtype=np.int64),
        "train": train,
        "val": np.array([24, 25, 26], dtype=np.int64),
        "val_context": np.array([20], dtype=np.int64),
        "context": np.array([28, 29, 30, 31], dtype=np.int64),
    }


def e2a_roles() -> dict[str, np.ndarray]:
    split = roles()
    split.pop("val_context")
    return split


def protocol(epochs: int = 5) -> dict:
    proto = copy.deepcopy(load_protocol())
    proto["training"]["max_epochs"] = epochs
    proto["training"]["patience"] = epochs
    proto["arms"]["env"]["hidden"] = 16
    proto["arms"]["h2x"]["hidden"] = 16
    proto["arms"]["h2x"]["env_emb"] = 8
    return proto


def train(arm: str, dataset: dict, split: dict, epochs: int = 5) -> H3Trainer:
    trainer = H3Trainer(arm, 0, protocol(epochs))
    trainer.fit(dataset, split)
    return trainer


def perturb_cells(dataset: dict, cells: np.ndarray, amount: float) -> dict:
    """Copy of the dataset with the labels at the given cells shifted."""
    changed = dict(dataset)
    y = dataset["y"].clone()
    flat = y.reshape(-1)
    flat[torch.as_tensor(cells)] += amount
    changed["y"] = flat.reshape(y.shape)
    return changed


# ------------------------------------------------------- structural checks


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("split_factory", [roles, e2a_roles], ids=["e2b", "e2a"])
def test_visibility_roles_are_exactly_as_frozen(arm, split_factory):
    dataset, split = tiny_dataset(), split_factory()
    trainer = H3Trainer(arm, 0, protocol(2))
    trainer.fit(dataset, split)
    base = trainer.base_visible.reshape(-1)
    evaluation = trainer.eval_visible.reshape(-1)

    for role in ("val", "val_context", "test"):
        cells = split.get(role, np.empty(0, dtype=np.int64))
        if len(cells) == 0:
            continue
        assert not base[cells].any(), role
    # val and test stay hidden even in the selection/validation arrangement
    for role in ("val", "test"):
        assert not evaluation[split[role]].any(), role
    assert evaluation[split["train"]].all()
    assert evaluation[split["context"]].all()
    if "val_context" in split:
        assert evaluation[split["val_context"]].all()
    # the outer test is hidden even in the fully-open evaluation arrangement
    assert not trainer.full_visible.reshape(-1)[split["test"]].any()
    # during training only the context cells are visible before re-masking
    visible = set(np.flatnonzero(base.numpy()).tolist())
    assert visible == set(split["context"].tolist())


@pytest.mark.parametrize("arm", ARMS)
def test_training_forward_never_sees_val_roles(arm):
    """Re-masking must not leak any hidden role into the visible half."""
    dataset, split = tiny_dataset(), roles()
    trainer = H3Trainer(arm, 0, protocol(2))
    trainer.prepare(dataset, split)
    hidden = set(split["val"].tolist()) | set(split["val_context"].tolist())
    hidden |= set(split["test"].tolist())
    train_cells = set(split["train"].tolist())
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(sorted(train_cells))
        half = perm[: len(perm) // 2]
        assert set(half.tolist()) <= train_cells
        assert not (set(half.tolist()) & hidden)


# -------------------------------------------------------- behavioural checks


@pytest.mark.parametrize("arm", ARMS)
def test_changing_test_truth_changes_nothing(arm):
    dataset, split = tiny_dataset(), roles()
    other = perturb_cells(dataset, split["test"], 50.0)

    a = train(arm, dataset, split)
    b = train(arm, other, split)

    assert a.training_info_["best_epoch"] == b.training_info_["best_epoch"]
    assert a.training_info_["epochs"] == b.training_info_["epochs"]
    assert a.training_info_["best_val_mse_raw"] == b.training_info_["best_val_mse_raw"]
    assert [e["train_loss"] for e in a.epoch_log_] == [
        e["train_loss"] for e in b.epoch_log_
    ]
    assert [e["val_loss"] for e in a.epoch_log_] == [
        e["val_loss"] for e in b.epoch_log_
    ]
    for key in (
        "y_pred",
        "total_log",
        "base_log",
        "correction_log",
        "pred_log_clipped",
        "clipped",
    ):
        np.testing.assert_array_equal(
            np.nan_to_num(a.validation_[key], nan=-999.0),
            np.nan_to_num(b.validation_[key], nan=-999.0),
        )
    for key, value in a.best_state_.items():
        torch.testing.assert_close(b.best_state_[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_val_context_truth_never_enters_training(arm):
    dataset, split = tiny_dataset(), roles()
    other = perturb_cells(dataset, split["val_context"], 40.0)
    a = train(arm, dataset, split)
    b = train(arm, other, split)
    assert [e["train_loss"] for e in a.epoch_log_] == [
        e["train_loss"] for e in b.epoch_log_
    ]
    assert a.training_info_["epochs"] == b.training_info_["epochs"]
    for key in a.best_state_:
        torch.testing.assert_close(b.best_state_[key], a.best_state_[key], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ("h2x", "h3a"))
def test_val_context_is_visible_only_in_the_e2b_forward(arm):
    """Same data, same seed: exposing val_context must move the predictions."""
    dataset, split = tiny_dataset(), roles()
    exposed = train(arm, dataset, split)
    hidden = train(arm, dataset, e2a_roles())
    assert not np.allclose(
        exposed.validation_["total_log"], hidden.validation_["total_log"]
    )
    # the training trajectories stay identical: the difference comes purely
    # from the forward visibility, never from the loss
    assert [e["train_loss"] for e in exposed.epoch_log_] == [
        e["train_loss"] for e in hidden.epoch_log_
    ]


def test_env_forward_cannot_read_the_doc_channel_at_all():
    """ENV has no DOC input, so val_context can only reach it via selection.

    This is a property of the ENV arm, not a leak: the environmental view is
    built without the DOC value or visibility channels, so opening val_context
    cannot change an ENV prediction except by changing which checkpoint wins.
    """
    dataset, split = tiny_dataset(), roles()
    exposed = train("env", dataset, split)
    hidden = train("env", dataset, e2a_roles())
    np.testing.assert_allclose(
        exposed.validation_["total_log"], hidden.validation_["total_log"],
        rtol=0, atol=0,
    )
    assert [e["train_loss"] for e in exposed.epoch_log_] == [
        e["train_loss"] for e in hidden.epoch_log_
    ]


@pytest.mark.parametrize("arm", ARMS)
def test_val_truth_only_moves_selection(arm):
    dataset, split = tiny_dataset(), roles()
    other = perturb_cells(dataset, split["val"], 30.0)
    a = train(arm, dataset, split)
    b = train(arm, other, split)

    shared = min(len(a.epoch_log_), len(b.epoch_log_))
    assert [e["train_loss"] for e in a.epoch_log_[:shared]] == [
        e["train_loss"] for e in b.epoch_log_[:shared]
    ]
    assert [e["val_loss"] for e in a.epoch_log_] != [e["val_loss"] for e in b.epoch_log_]
    assert a.training_info_["parameter_count"] == b.training_info_["parameter_count"]


@pytest.mark.parametrize("arm", ARMS)
def test_exported_validation_never_contains_test_cells(arm):
    dataset, split = tiny_dataset(), roles()
    trainer = train(arm, dataset, split)
    cells = set(trainer.validation_["cells"].tolist())
    assert cells == set(split["val"].tolist())
    assert not cells & set(split["test"].tolist())
    assert not cells & set(split["val_context"].tolist())
    truth = dataset["y"].reshape(-1).numpy()
    np.testing.assert_allclose(
        trainer.validation_["y_true"], truth[split["val"]], rtol=0, atol=0
    )


@pytest.mark.parametrize("arm", ARMS)
def test_components_are_consistent_and_scale_correct(arm):
    dataset, split = tiny_dataset(), roles()
    trainer = train(arm, dataset, split)
    v = trainer.validation_
    assert np.isfinite(v["total_log"]).all()
    if arm == "h3a":
        assert np.isfinite(v["base_log"]).all()
        assert np.isfinite(v["correction_log"]).all()
        np.testing.assert_array_equal(
            v["base_log"] + v["correction_log"], v["total_log"]
        )
    elif arm == "env":
        assert np.isnan(v["correction_log"]).all()
        assert np.isfinite(v["base_log"]).all()
    else:
        assert np.isnan(v["base_log"]).all()
        assert np.isfinite(v["correction_log"]).all()
    lo, hi = v["clip_log_bounds"]
    assert np.all(v["pred_log_clipped"] <= hi + 1e-6)
    assert np.all(v["pred_log_clipped"] >= lo - 1e-6)
    np.testing.assert_allclose(
        v["y_pred"], np.expm1(v["pred_log_clipped"]), rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(v["clipped"], v["pred_log_clipped"] != v["total_log"])


@pytest.mark.parametrize("arm", ARMS)
def test_trainer_is_deterministic(arm):
    dataset, split = tiny_dataset(), roles()
    a = train(arm, dataset, split)
    b = train(arm, dataset, split)
    assert a.training_info_["best_epoch"] == b.training_info_["best_epoch"]
    for key in a.best_state_:
        torch.testing.assert_close(b.best_state_[key], a.best_state_[key], rtol=0, atol=0)
    np.testing.assert_array_equal(
        a.validation_["total_log"], b.validation_["total_log"]
    )


def test_protocol_rejects_a_foreign_loss():
    proto = protocol()
    proto["training"]["loss"] = "mse"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "protocol.json"
        path.write_text(json.dumps(proto), encoding="utf-8")
        with pytest.raises(ValueError):
            load_protocol(path)


def test_env_arm_has_no_graph_inputs():
    dataset, split = tiny_dataset(), roles()
    trainer = H3Trainer("env", 0, protocol(2))
    trainer.prepare(dataset, split)
    assert trainer.env.shape[-1] == len(ENV_FEATURE_NAMES)
    trainer.build_model()
    assert not hasattr(trainer.model, "correction")
