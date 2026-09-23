"""Regression tests for env_groups column selection in GCNDocModel._build_inputs.

The v04 regime matrix is 13 columns: 0:4 hydro, 4:8 landcover, 8:10 climate,
10:11 soil, 11:13 topo. A past defect subset positions were computed from the
kept column list but used to index the full-width matrix, so e.g.
env_groups=["climate"] silently selected the hydro columns. These tests pin
that each group selects exactly its own original columns, on both the encoder
and the non-encoder path.
"""

import numpy as np
import torch

from river_graph.models.gcn import GCNDocModel

GROUPS = {
    "hydro": [0, 1, 2, 3],
    "landcover": [4, 5, 6, 7],
    "climate": [8, 9],
    "soil": [10],
    "topo": [11, 12],
}
N, T = 6, 4
BASE_CHANNELS = 10  # fixed channels before regime columns are appended


def _toy_dataset() -> dict:
    # Distinct per-column shapes: constant or affine-related columns would be
    # indistinguishable after per-column standardization.
    g = torch.Generator().manual_seed(0)
    regime = torch.randn(N, 13, generator=g)
    return {
        "y": torch.rand(N, T),
        "x": torch.rand(N, T, 2),
        "x_mask": torch.ones(N, T, 2),
        "months": [f"2020-{m:02d}-01" for m in range(1, T + 1)],
        "static": torch.rand(N, 2),
        "regime": regime,
    }


def _split() -> dict:
    return {"train": np.arange(N * T)}


def _standardized(regime: torch.Tensor) -> torch.Tensor:
    return (regime - regime.mean(0)) / (regime.std(0) + 1e-8)


def _run(env_groups, env_encoder: bool):
    ds = _toy_dataset()
    model = GCNDocModel(env_groups=env_groups, env_encoder=env_encoder)
    xt, _y, _bv, _tc, _stats, env_raw = model._build_inputs(ds, _split())
    return xt, env_raw, _standardized(ds["regime"])


def _appended(xt: torch.Tensor) -> torch.Tensor:
    """Regime channels appended after the 10 fixed ones; (N, k)."""
    return xt[0, :, BASE_CHANNELS:]


def test_none_selects_all_columns():
    xt, env_raw, std = _run(None, env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 9)
    assert torch.allclose(env_raw, std[:, 4:13])
    assert torch.allclose(_appended(xt), std[:, 0:4])


def test_climate_group_selects_climate_columns():
    # The original defect returned hydro columns 0:2 here.
    xt, env_raw, std = _run(["climate"], env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 2)
    assert torch.allclose(env_raw, std[:, 8:10])
    assert xt.shape[-1] == BASE_CHANNELS  # no hydro kept -> nothing appended


def test_landcover_group_selects_landcover_columns():
    _xt, env_raw, std = _run(["landcover"], env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 4)
    assert torch.allclose(env_raw, std[:, 4:8])


def test_topo_group_selects_topo_columns():
    _xt, env_raw, std = _run(["topo"], env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 2)
    assert torch.allclose(env_raw, std[:, 11:13])


def test_soil_group_selects_soil_column():
    _xt, env_raw, std = _run(["soil"], env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 1)
    assert torch.allclose(env_raw, std[:, 10:11])


def test_hydro_only_goes_to_x_without_encoder_input():
    xt, env_raw, std = _run(["hydro"], env_encoder=True)
    assert env_raw is None
    assert torch.allclose(_appended(xt), std[:, 0:4])


def test_hydro_plus_climate_splits_correctly():
    xt, env_raw, std = _run(["hydro", "climate"], env_encoder=True)
    assert env_raw is not None and env_raw.shape == (N, 2)
    assert torch.allclose(env_raw, std[:, 8:10])
    assert torch.allclose(_appended(xt), std[:, 0:4])


def test_non_encoder_path_respects_subset():
    xt, env_raw, std = _run(["climate"], env_encoder=False)
    assert env_raw is None
    assert torch.allclose(_appended(xt), std[:, 8:10])


def test_non_encoder_none_keeps_all_13():
    xt, env_raw, std = _run(None, env_encoder=False)
    assert env_raw is None
    assert xt.shape[-1] == BASE_CHANNELS + 13
    assert torch.allclose(_appended(xt), std)
