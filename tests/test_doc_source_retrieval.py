"""Transfer-bank exclusion, portable prediction and sparse attention contracts."""
import copy
import io
import json

import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.unmonitored_doc import (
    development_labels,
    geographical_split,
    strip_auxiliary_water,
    validate_zero_observation_view,
)
from river_graph.models.doc_source_data import nested_source_episodes
from river_graph.models.doc_source_retrieval import (
    SourceResidualBank,
    SourceRetrievalAttention,
)


def bank_case():
    rng = np.random.default_rng(3)
    n, t = 10, 18
    ecology = rng.uniform(1, 10, (n, 9))
    hydro = rng.uniform(1, 10, (n, t, 2))
    valid = np.ones_like(hydro, dtype=bool)
    months = np.arange("2000-01", "2001-07", dtype="datetime64[M]")
    names = np.asarray([f"source{i}" for i in range(n)])
    cells = np.arange(8*t)
    p = np.full(len(cells), 5.)
    y = p+np.repeat(np.linspace(-2, 2, 8), t)
    bank = SourceResidualBank(max_donors=4).fit(ecology, hydro, valid, months, cells, p, y, station_names=names)
    return bank, ecology, hydro, valid, months, names


def query(bank, case, rows=(8, 9)):
    _, ecology, hydro, valid, months, names = case
    return bank.query(ecology[list(rows)], hydro[list(rows), 0], valid[list(rows), 0],
                      np.repeat(months[0], len(rows)), np.full(len(rows), 5.),
                      np.ones((len(rows), 4)), station_names=names[list(rows)])


def tensors(prepared):
    return [torch.as_tensor(prepared[key], dtype=torch.bool if key == "valid" else torch.float32)
            for key in ("query", "keys", "values", "valid")]


def test_bank_excludes_entire_pseudo_target_station():
    case = bank_case()
    prepared = query(case[0], case, (0, 8))
    assert "source0" not in prepared["donor_names"][0]
    bank, ecology, hydro, valid, months, names = case
    with pytest.raises(ValueError, match="excluded"):
        SourceResidualBank().fit(ecology, hydro, valid, months, np.arange(8*18),
                                 np.full(8*18, 5.), np.full(8*18, 6.), station_names=names,
                                 excluded_station_names=["source0"])
    assert set(bank.station_names_) == set(names[:8])


def test_normalization_and_profiles_never_use_target_covariates_or_labels():
    case = bank_case()
    bank, ecology, hydro, valid, months, names = case
    changed_eco, changed_hydro = ecology.copy(), hydro.copy()
    changed_eco[8:] = 1e9
    changed_hydro[8:] = 1e9
    restored = SourceResidualBank(max_donors=4).fit(changed_eco, changed_hydro, valid, months,
        np.arange(8*18), np.full(8*18, 5.), np.full(8*18, 5.)+np.repeat(np.linspace(-2, 2, 8), 18),
        station_names=names)
    assert bank.to_dict() == restored.to_dict()


def test_new_station_count_months_and_names_are_not_cohort_bound():
    case = bank_case()
    bank = case[0]
    q = query(bank, case)
    out = bank.query(case[1][8:9], case[2][8:9, 3], case[3][8:9, 3],
                     np.array(["2030-12"], dtype="datetime64[M]"), np.array([7.]),
                     np.zeros((1, 4)), station_names=["entirely-new-station"])
    assert q["keys"].shape == (2, 4, 28)
    assert out["query"].shape[0] == 1
    assert out["donor_names"].shape == (1, 4)


def test_zero_projection_preserves_static_memory_and_sparse_weights_normalize():
    case = bank_case()
    q = query(case[0], case)
    model = SourceRetrievalAttention(q["query"].shape[1]).eval()
    delta, diagnostics = model(*tensors(q), diagnostics=True)
    torch.testing.assert_close(delta, torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(diagnostics["weights"].sum(-1), torch.ones((2, 2)))
    assert torch.all(diagnostics["effective_donors"] >= 1)
    delta.sum().backward()
    assert torch.count_nonzero(model.projection.weight.grad) > 0


def test_padding_donors_have_zero_attention_and_no_valid_donors_are_explicit():
    case = bank_case()
    q = query(case[0], case)
    q["valid"][:, -1] = False
    model = SourceRetrievalAttention(q["query"].shape[1]).eval()
    _, diagnostics = model(*tensors(q), diagnostics=True)
    assert torch.count_nonzero(diagnostics["weights"][..., -1]) == 0
    q["valid"][:] = False
    with pytest.raises(ValueError, match="valid donor"):
        model(*tensors(q))


def test_bank_and_attention_save_load_replay_nonzero_branch():
    case = bank_case()
    bank = case[0]
    q = query(bank, case)
    model = SourceRetrievalAttention(q["query"].shape[1]).eval()
    with torch.no_grad():
        model.projection.weight.fill_(.1)
    expected = model(*tensors(q))
    buffer = io.BytesIO()
    torch.save(model.to_payload(), buffer)
    buffer.seek(0)
    replay = SourceRetrievalAttention.from_payload(torch.load(buffer, weights_only=False))
    bank_replay = SourceResidualBank.from_dict(json.loads(json.dumps(bank.to_dict())))
    torch.testing.assert_close(replay(*tensors(query(bank_replay, case))), expected, rtol=0, atol=0)


def test_geographical_roles_use_canonical_huc_column_and_keep_sparse_stations():
    mask = np.ones((6, 8), dtype=bool)
    mask[0, 1:] = False
    split, metadata = geographical_split(mask, [10130200, 10190000, 7080200, 10300100, 11010000, 5010000], "1013")
    assert split["test"].tolist() == [0]
    assert set(split["val"]//8) == {1}
    assert 0 not in metadata["curve_eligible"]
    assert len(np.concatenate(list(split.values()))) == mask.sum()


def test_development_labels_and_auxiliary_stripping_keep_target_truth_out():
    data = {"y": np.arange(12).reshape(3, 4), "ph": np.ones((3, 4)), "ph_mask": np.ones((3, 4))}
    split = {"train": np.arange(4), "val": np.arange(4, 8), "test": np.arange(8, 12)}
    expected = development_labels(data, split)
    changed = copy.deepcopy(data)
    changed["y"][2] = 999999
    np.testing.assert_array_equal(expected, development_labels(changed, split))
    assert np.isnan(expected[2]).all()
    assert "ph" not in strip_auxiliary_water(data)


def test_zero_observation_view_rejects_last_value_and_visibility():
    inputs = {"raw": np.zeros((2, 3, 23)), "support": np.zeros((2, 3, 3))}
    validate_zero_observation_view(inputs, [1])
    inputs["raw"][1, 0, -9] = 4
    with pytest.raises(ValueError, match="local DOC"):
        validate_zero_observation_view(inputs, [1])


def test_nested_forests_and_donor_bank_are_invariant_to_outer_station_labels():
    rng = np.random.default_rng(19)
    n, t = 8, 8
    dataset = {"y": rng.uniform(1, 5, (n, t)), "y_mask": np.ones((n, t), dtype=bool),
        "x": rng.uniform(1, 5, (n, t, 2)), "x_mask": np.ones((n, t, 2), dtype=bool),
        "static": rng.uniform(0, 1, (n, 2)), "regime": rng.uniform(1, 3, (n, 13)),
        "edge_index": np.empty((2, 0), dtype=np.int64), "site_no": np.array([f"site{i}" for i in range(n)]),
        "months": np.arange("2000-01", "2000-09", dtype="datetime64[M]")}
    split = {"train": np.arange(6*t), "val": np.arange(6*t, 7*t), "test": np.arange(7*t, 8*t),
             "context": np.empty(0, dtype=np.int64)}
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    forest = ExtraTreesRegressor(n_estimators=3, random_state=42, n_jobs=1)
    before, records = nested_source_episodes(forest, dataset, split, folds, n_jobs=1)
    changed = copy.deepcopy(dataset)
    changed["y"][:2] += 1000
    changed["y"][6:] += 5000
    after, _ = nested_source_episodes(forest, changed, split, folds, n_jobs=1)
    np.testing.assert_array_equal(before[0]["context"], after[0]["context"])
    assert before[0]["bank"].to_dict() == after[0]["bank"].to_dict()
    for inner in records[0]["inner_records"]:
        assert not set(inner["fitted_stations"]) & {0, 1}
        assert not set(inner["held_stations"]) & {0, 1}
