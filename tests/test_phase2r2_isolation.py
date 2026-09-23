"""2B-R2 contracts: validation-label isolation and complete config identity.

The 2B-R2 review found that tabular features were built once from
train ∪ val ∪ context and reused for fitting, early stopping and testing, so
validation labels leaked into fit inputs. These tests pin the staged
visibility fix (fit/early-stop views hide every validation label; only the
final test-time view opens val/context per protocol) and pin that every
result-affecting parameter changes the config hash under schema V2.
"""

import numpy as np
import torch

from river_graph.baselines.baselines import (
    FIT_VISIBILITY,
    TEST_VISIBILITY,
    EcoMLP,
    EcoRandomForest,
    ecological_tabular_features,
)
from river_graph.experiments.provenance import (
    CONFIG_FIELDS_V1,
    CONFIG_FIELDS_V2,
    config_hash,
    identity_problems,
)

N, T = 6, 4


def _toy_dataset() -> dict:
    g = torch.Generator().manual_seed(11)
    y = torch.rand(N, T, generator=g) * 6
    return {
        "y": y,
        "y_mask": torch.ones(N, T, dtype=torch.bool),
        "x": torch.rand(N, T, 2, generator=g),
        "x_mask": (torch.rand(N, T, 2, generator=g) > 0.2).float(),
        "months": [f"2020-{m:02d}-01" for m in range(1, T + 1)],
        "static": torch.rand(N, 2, generator=g),
        "regime": torch.randn(N, 13, generator=g),
        "site_no": [f"s{i}" for i in range(N)],
    }


def _toy_split() -> dict:
    return {
        "train": np.array([0, 1, 2, 4, 5], dtype=np.int64),
        "val": np.array([3, 6], dtype=np.int64),
        "test": np.array([7, 10], dtype=np.int64),
        "context": np.array([8], dtype=np.int64),
    }


def _perturb_val_labels(ds: dict, split: dict) -> dict:
    ds2 = {**ds, "y": ds["y"].clone()}
    for cell in np.asarray(split["val"], dtype=np.int64):
        i, j = divmod(int(cell), T)
        ds2["y"][i, j] += 10.0
    return ds2


def test_fit_view_features_ignore_validation_labels_entirely():
    ds, split = _toy_dataset(), _toy_split()
    ds2 = _perturb_val_labels(ds, split)
    fit1, names = ecological_tabular_features(ds, split, visibility=FIT_VISIBILITY)
    fit2, _ = ecological_tabular_features(ds2, split, visibility=FIT_VISIBILITY)
    assert np.allclose(fit1, fit2), (
        "validation labels must not change fit-view features for ANY row"
    )
    assert "month_visdoc_count_excl_self" in names


def test_test_view_features_do_expose_validation_labels():
    ds, split = _toy_dataset(), _toy_split()
    ds2 = _perturb_val_labels(ds, split)
    t1, _ = ecological_tabular_features(ds, split, visibility=TEST_VISIBILITY)
    t2, _ = ecological_tabular_features(ds2, split, visibility=TEST_VISIBILITY)
    assert not np.allclose(t1, t2), (
        "test-view must open val/context per protocol (this is the staged reveal)"
    )


def test_default_view_is_the_test_view():
    ds, split = _toy_dataset(), _toy_split()
    d, _ = ecological_tabular_features(ds, split)
    t, _ = ecological_tabular_features(ds, split, visibility=TEST_VISIBILITY)
    assert np.allclose(d, t)


def test_unknown_visibility_keys_are_refused():
    try:
        ecological_tabular_features(
            _toy_dataset(), _toy_split(), visibility={"train", "test"}
        )
    except ValueError as exc:
        assert "visibility" in str(exc)
    else:
        raise AssertionError("unknown visibility keys must raise")


def test_mlp_early_stop_trace_is_recorded_with_staged_visibility():
    ds, split = _toy_dataset(), _toy_split()
    clf = EcoMLP(hidden=(8,), max_epochs=6, patience=2, seed=0)
    pred = clf.fit_predict(ds, split)
    tr = clf.early_stop_
    assert tr is not None
    assert set(tr) >= {"epochs_run", "best_epoch", "val_losses",
                       "visibility_fit", "visibility_test"}
    assert tr["visibility_fit"] == sorted(FIT_VISIBILITY)
    assert tr["visibility_test"] == sorted(TEST_VISIBILITY)
    assert "val" not in tr["visibility_fit"]
    assert 1 <= tr["epochs_run"] <= 6
    te = np.asarray(split["test"], dtype=np.int64)
    assert np.isfinite(pred.ravel()[te]).all()
    assert clf.feature_names_ and len(clf.feature_names_) == 23


def test_rf_records_no_early_stop_but_exposes_feature_names():
    ds, split = _toy_dataset(), _toy_split()
    clf = EcoRandomForest(n_estimators=8, seed=0)
    pred = clf.fit_predict(ds, split)
    assert clf.early_stop_ is None
    assert len(clf.feature_names_) == 23
    te = np.asarray(split["test"], dtype=np.int64)
    assert np.isfinite(pred.ravel()[te]).all()


def _base_params() -> dict:
    return {
        "script": "tests/test_phase2r2_isolation.py",
        "model_name": "P2X_H2X", "tag": "H2X", "architecture": "transport_enc",
        "variant": "river", "seed": 42, "lr": 1e-3, "weight_decay": 0.0,
        "edge_dropout": 0.0, "share_weights": False, "env_groups": None,
        "env_encoder": True, "dataset_path": "d.pt", "dataset_sha256": "a" * 64,
        "mask_path": "m.npz", "mask_sha256": "b" * 64,
        "edge_set": "river", "edge_direction": "both",
        "hidden": 64, "layers": 2, "dropout": 0.1,
        "max_epochs": 200, "patience": 20, "env_emb": 32,
    }


def test_config_hash_v2_changes_with_every_result_affecting_param():
    base = _base_params()
    h0 = config_hash(base, version=2)
    for field, value in (
        ("edge_set", "empty"), ("edge_direction", "upstream"),
        ("hidden", 128), ("layers", 3), ("dropout", 0.5),
        ("max_epochs", 5), ("patience", 3), ("env_emb", 16),
    ):
        changed = {**base, field: value}
        assert config_hash(changed, version=2) != h0, field


def test_config_hash_v1_schema_documents_the_old_gap():
    """V1 (historical sidecars) ignored these knobs — that is the defect."""
    base = _base_params()
    assert set(CONFIG_FIELDS_V1) < set(CONFIG_FIELDS_V2)
    h0 = config_hash(base, version=1)
    changed = {**base, "max_epochs": 1, "edge_set": "empty"}
    assert config_hash(changed, version=1) == h0


def test_identity_problems_respects_the_sidecar_schema():
    base = _base_params()
    meta_v2 = {"config": base, "config_hash_version": 2}
    meta_v1 = {"config": {k: base[k] for k in CONFIG_FIELDS_V1}}
    expected_changed = {**base, "max_epochs": 7}
    assert identity_problems(meta_v2, expected_changed)  # v2 detects
    assert identity_problems(meta_v1, expected_changed) == []  # v1 cannot
    assert identity_problems(meta_v2, base) == []
