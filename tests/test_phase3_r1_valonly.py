"""3B-R1 contracts (docs/paper/phase3_uncertainty_spec_v2.md).

Val-only recalibration may change ONLY qhat / pi bounds / uncertainty.
Central prediction, seed spread and covariates stay bit-identical; test and
train label perturbations must not move v2 calibration; val perturbation
must; v1 artifacts are never written.
"""

import json

import numpy as np
import torch

from river_graph.experiments.phase3_uncertainty import (
    calibrate_interval,
    covariates,
    scenario_visibility,
)

N, T = 6, 4


def _toy_dataset(seed: int = 5) -> dict:
    g = torch.Generator().manual_seed(seed)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return {
        "y": torch.rand(N, T, generator=g) * 5,
        "y_mask": torch.ones(N, T, dtype=torch.bool),
        "x": torch.rand(N, T, 2, generator=g),
        "x_mask": (torch.rand(N, T, 2, generator=g) > 0.2).float(),
        "months": [f"2020-{m:02d}-01" for m in range(1, T + 1)],
        "static": torch.rand(N, 2, generator=g),
        "regime": torch.randn(N, 13, generator=g),
        "edge_index": ei,
        "edge_attr": torch.randn(ei.shape[1], 6, generator=g),
        "site_no": [f"s{i}" for i in range(N)],
    }


def _split() -> dict:
    return {
        "train": np.array([0, 1, 2, 4, 5, 8], dtype=np.int64),
        "val": np.array([3, 6], dtype=np.int64),
        "test": np.array([7, 10, 11], dtype=np.int64),
        "context": np.array([9], dtype=np.int64),
    }


def _perturb(ds: dict, cells) -> dict:
    ds2 = {**ds, "y": ds["y"].clone()}
    for cell in np.asarray(cells, dtype=np.int64):
        i, j = divmod(int(cell), T)
        ds2["y"][i, j] += 10.0 + i
    return ds2


def _cal(seed_log, y_log, split):
    val = np.asarray(split["val"], dtype=np.int64)
    return calibrate_interval(seed_log, y_log, val)


def test_valonly_calibration_ignores_train_and_test_labels():
    ds, split = _toy_dataset(), _split()
    rng = np.random.default_rng(1)
    seed_log = rng.normal(size=(5, N, T))
    y0 = np.log1p(ds["y"].numpy())
    base = _cal(seed_log, y0, split)
    for cells, what in ((split["test"], "test"), (split["train"], "train"),
                        (split["context"], "context")):
        y2 = np.log1p(_perturb(ds, cells)["y"].numpy())
        out = _cal(seed_log, y2, split)
        assert out["qhat"] == base["qhat"], what
        assert np.allclose(out["uncertainty"], base["uncertainty"],
                           rtol=0, atol=0), what


def test_valonly_calibration_moves_with_val_labels():
    ds, split = _toy_dataset(), _split()
    rng = np.random.default_rng(2)
    seed_log = rng.normal(size=(5, N, T))
    y0 = np.log1p(ds["y"].numpy())
    base = _cal(seed_log, y0, split)
    y2 = np.log1p(_perturb(ds, split["val"])["y"].numpy())
    out = _cal(seed_log, y2, split)
    assert out["qhat"] != base["qhat"], "val is the calibration set"


def test_center_and_spread_invariant_to_any_label_perturbation():
    ds, split = _toy_dataset(), _split()
    rng = np.random.default_rng(3)
    seed_preds = np.maximum(0.5 + rng.normal(scale=0.2, size=(5, N, T)), 0.0)

    def center_spread(d):
        y_log = np.log1p(d["y"].numpy())
        out = calibrate_interval(np.log1p(seed_preds), y_log,
                                 np.asarray(split["val"], dtype=np.int64))
        return out["prediction_median"], out["uncertainty"]

    for cells in (split["test"], split["train"], split["val"]):
        m1, _ = center_spread(ds)
        m2, _ = center_spread(_perturb(ds, cells))
        assert np.allclose(m1, m2, rtol=0, atol=0)


def test_covariates_invariant_to_any_label_perturbation():
    ds, split = _toy_dataset(), _split()
    vis = scenario_visibility(split, "E2b")
    c1 = covariates(ds, vis)
    c2 = covariates(_perturb(ds, np.arange(N * T)), vis)
    for k in c1:
        assert np.allclose(c1[k], c2[k], rtol=0, atol=0, equal_nan=True), k


def test_cal_def_hash_is_stable_and_roles_recorded():
    import hashlib
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "r1", "scripts/run3b_r1_valonly.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    h = mod.cal_def_hash()
    blob = json.dumps(mod.CAL_DEF, sort_keys=True, separators=(",", ":"))
    assert h == hashlib.sha256(blob.encode()).hexdigest()
    assert mod.CAL_DEF["rule"] == "val_only"
    assert mod.CAL_DEF["keys"] == ["val"]
    assert mod.OUT_V2 != mod.OUT / "full_grid"
    assert "v2" in mod.SPEC_V2.name
