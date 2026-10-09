"""Expanded donor access, double-held exclusion and variable-width attention."""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
    nested_current_available_candidates,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_wide_availability import nested_wide_available_candidates


def test_larger_pool_preserves_nearest_prefix_scale_and_calendar(tmp_path):
    rng = np.random.default_rng(73)
    names = np.array([f"source-{i}" for i in range(80)])
    months = pd.period_range("2000-01", periods=24, freq="M").astype(str).to_numpy()
    eco, residual = rng.normal(size=(80, 9)), rng.normal(size=(80, 24))
    residual[rng.uniform(size=residual.shape) < .4] = np.nan
    hydro = rng.normal(size=(80, 24, 8))
    libraries = [SourceDOCInnovationLibrary(candidate_count=k).fit(names, months, eco, residual)
                 for k in (20, 60)]
    assert libraries[0].sigma_ == libraries[1].sigma_
    arrays = [current_available_candidates(lib, np.array(["new", names[0]]), eco[:2],
        months, names, hydro) for lib in libraries]
    for key in ("donor_owner", "donor_ecology", "donor_log_prior"):
        np.testing.assert_array_equal(arrays[0][key][:, :20], arrays[1][key][:, :20])
    for key in ("donor_values_real", "donor_values_seasonal", "donor_valid", "donor_previous_valid"):
        np.testing.assert_array_equal(arrays[0][key][..., :20], arrays[1][key][..., :20])
    assert arrays[1]["donor_valid"].shape == (2, 24, 61)
    assert arrays[1]["donor_valid"][..., :-1].sum() > arrays[0]["donor_valid"][..., :-1].sum()
    assert 0 not in arrays[1]["donor_owner"][1]
    assert not arrays[1]["donor_values_real"][~arrays[1]["donor_valid"]].any()
    libraries[1].save(tmp_path/"library.npz")
    restored = SourceDOCInnovationLibrary.load(tmp_path/"library.npz")
    replay = current_available_candidates(restored, np.array(["new", names[0]]), eco[:2],
                                         months, names, hydro)
    for key, values in arrays[1].items():
        np.testing.assert_array_equal(values, replay[key])
    # Fixed preprocessing: later source observations cannot enter earlier inputs.
    altered = deepcopy(restored)
    altered.innovations_[:, 16:] += 1000.
    future = current_available_candidates(altered, np.array(["new", names[0]]), eco[:2],
                                         months, names, hydro)
    np.testing.assert_array_equal(replay["donor_values_real"][:, :16], future["donor_values_real"][:, :16])


def test_nested_pool_exclusion_and_narrow_reproduction():
    args = _nested_fixture()
    hydro = np.zeros((6, 36, 8))
    narrow, records = nested_current_available_candidates(*args, hydro)
    reproduction, wide_records = nested_wide_available_candidates(*args, hydro, candidate_count=20)
    for key in narrow:
        np.testing.assert_array_equal(narrow[key], reproduction[key])
    assert records == [{k: v for k, v in record.items() if k != "candidate_count"}
                       for record in wide_records]
    wide, records = nested_wide_available_candidates(*args, hydro)
    changed = deepcopy(args)
    changed[0][:2] += 1000.
    changed[0][6] = np.nan
    altered, _ = nested_wide_available_candidates(*changed, hydro)
    for key in wide:
        if key != "donor_hydro_bank":
            np.testing.assert_array_equal(wide[key][:2], altered[key][:2])
    assert wide["donor_owner"].shape == (6, 61)
    assert not np.isin(wide["donor_owner"][:2], [0, 1]).any()
    assert all(not set(r["query_station_ids"]) & set(r["library_station_ids"]) for r in records)


def test_wider_attention_keeps_capacity_and_accepts_new_nodes():
    original, _, args = _model()
    model = AvailableSourceAttentionResidual(original.spatial, original.temporal, original.decay,
                                            **original._config(), **original.attention_config)
    assert model.trainable_parameter_count_ == original.trainable_parameter_count_
    rng = np.random.default_rng(87)
    for position in (0, 4):
        view = args[position]
        n, t = view["age"].shape
        view["attention_reference"] = np.full((n, t), 3.)
        view["donor_owner"] = np.tile(np.r_[np.arange(60), -1], (n, 1))
        view["donor_ecology"] = rng.normal(size=(n, 61, 2))
        view["donor_ecology"][:, -1] = 0.
        view["donor_log_prior"] = np.tile(np.r_[np.linspace(-1., -5., 60), 0.], (n, 1))
        view["donor_values"] = rng.normal(size=(n, t, 61))
        view["donor_values"][..., -1] = 0.
        view["donor_valid"] = np.ones((n, t, 61), bool)
        view["donor_hydro_bank"] = rng.normal(size=(60, t, 8))
    model.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = AvailableSourceAttentionResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(model.predict(args[4], args[5]), restored.predict(args[4], args[5]))
    arrays = restored._prepare_inputs(args[4])
    cells = torch.tensor([0, 8, 15])
    _, weights = restored._attention_cells(arrays, cells, restored._hidden_cells(arrays, cells))
    torch.testing.assert_close(weights.sum(-1), torch.ones((3, 2), dtype=model.dtype), rtol=0, atol=1e-15)
    future = deepcopy(args[4])
    future["donor_values"][:, 8:, :-1] += 100.
    future["donor_hydro_bank"][:, 8:] += 100.
    np.testing.assert_array_equal(restored.predict_delta(args[4])[:, :8], restored.predict_delta(future)[:, :8])
    more = deepcopy(args[4])
    for key, values in more.items():
        if key != "donor_hydro_bank":
            more[key] = np.concatenate([values, values[:1]], axis=0)
    assert restored.predict_delta(more).shape == (3, 14)
