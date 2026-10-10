"""Dynamic allocation, protected base, physical banks and causal inputs."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_river_structure_residual import geography
from test_source_innovation_training import _nested_fixture

from river_graph.models.dynamic_river_residual import (
    DynamicRiverResidual,
    river_cell_features,
)
from river_graph.models.river_structure_residual import path_candidates


def example():
    rng = np.random.default_rng(42)
    n, t, c = 2, 8, 3
    hydro = rng.uniform(0, 1, (4, t, 8))
    messages = {'river_owner': np.tile([0, 1, 2], (n, 1)),
        'river_path': rng.uniform(0, 1, (n, c, 8)),
        'river_values': np.zeros((n, t, c, 3)),
        'river_valid': np.zeros((n, t, c, 3), bool),
        'river_age': np.zeros((n, t, c, 3))}
    for j, lag in enumerate((0, 1, 3)):
        messages['river_values'][:, lag:, :, j] = .3+j*.1
        messages['river_valid'][:, lag:, :, j] = True
        messages['river_age'][:, lag:, :, j] = j
    cells = np.arange(n*t)
    arrays = river_cell_features(messages, cells, hydro[:n], hydro)
    del arrays['query_hydro']
    arrays['query'] = rng.normal(size=(n*t, 10)).astype(np.float32)
    model = DynamicRiverResidual(10, arrays['features'].shape[-1], epochs=3, patience=3, batch_size=8)
    return model, arrays, messages, hydro, cells


def test_zero_initialization_and_no_support_preserve_full_base_bits():
    model, arrays, _, _, _ = example()
    base = np.arange(16, dtype=float)/7+.03
    np.testing.assert_array_equal(model.predict(arrays, base), base)
    with torch.no_grad():
        model.output.weight.fill_(.1)
    changed = deepcopy(arrays)
    changed['valid'][5] = False
    changed['values'][5] = 0.
    pred, diag = model.predict(changed, base, diagnostics=True)
    assert pred[5] == base[5]
    assert diag['prior_mass'][5] == 1.
    assert diag['delta_log'][5] == 0.
    assert np.any(pred != base)
    np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., atol=1e-6)


def test_future_hydro_and_observations_leave_earlier_features_unchanged():
    _, _, messages, hydro, cells = example()
    before = river_cell_features(messages, cells, hydro[:2], hydro)
    changed = hydro.copy()
    changed[:, 5:] += 10.
    altered = deepcopy(messages)
    altered['river_values'][:, 5:] += 7.
    altered['river_values'][~altered['river_valid']] = 0.
    after = river_cell_features(altered, cells, changed[:2], changed)
    for key in before:
        np.testing.assert_array_equal(before[key][cells % 8 < 5], after[key][cells % 8 < 5])


def test_source_hydro_indices_follow_candidate_lag_and_previous_month():
    _, _, messages, _, _ = example()
    hydro = np.broadcast_to(np.arange(8)[None, :, None]/10., (4, 8, 8)).copy()
    features = river_cell_features(messages, np.array([5]), hydro[:2], hydro)['features']
    for j, lag in enumerate((0, 1, 3)):
        np.testing.assert_allclose(features[0, 0, j, 8:16], (5-lag)/10.)
        np.testing.assert_allclose(features[0, 0, j, 16:24], .1)


@pytest.mark.parametrize('mode', ['static_same_month', 'dynamic_same_month', 'dynamic_lagged'])
def test_training_checkpoint_and_temporal_ablation(mode):
    _, arrays, _, _, _ = example()
    model = DynamicRiverResidual(10, arrays['features'].shape[-1], mode=mode,
                                  epochs=3, patience=3, batch_size=8)
    base = np.full(16, 3.)
    truth = np.full(16, 3.5)
    model.fit(arrays, base, truth, arrays, base, truth, tail_threshold=3.4)
    assert model.epochs_run_ > 0
    assert np.linalg.norm(model.output.weight.detach().numpy()) > 0
    loaded = DynamicRiverResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(model.predict(arrays, base), loaded.predict(arrays, base))
    assert model.predict(arrays, base).mean() > base.mean()
    diag = loaded.predict(arrays, base, diagnostics=True)[1]
    if mode != 'dynamic_lagged':
        assert np.all(diag['lag_mass'][:, 1:] == 0.)


def test_observation_bank_excludes_receiver_labels_and_direction():
    names = np.array(['a', 'b', 'c', 'd'])
    edges, physical = geography(names)
    residual = np.full((2, 8), np.nan)
    residual[0, 2], residual[1, 3] = .4, -.2
    upstream = path_candidates(edges, physical, ['c'], ['a', 'd'], residual)
    assert upstream['river_owner'][0, 0] == 0
    assert 1 not in upstream['river_owner']
    changed = residual.copy()
    changed[1] = 1000.
    repeated = path_candidates(edges, physical, ['c'], ['a', 'd'], changed)
    for key in upstream:
        np.testing.assert_array_equal(upstream[key], repeated[key])


def test_double_held_training_bank_and_matched_control(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_doc_dynamic_river_v1 import nested_banks

    truth, names, _, _, cells, folds, references = _nested_fixture()
    edges, physical = geography(names)
    # Disjoint short links leave eligible non-ancestors for every real donor.
    edges = edges.iloc[[0, 2, 4]].copy()
    physical['comid'] = np.arange(len(names))
    banks, roles, _ = nested_banks(truth, names, cells, folds, references, edges, physical)
    for key in ('river_valid', 'river_age', 'river_path'):
        np.testing.assert_array_equal(banks['matched'][key], banks['control'][key])
    altered = truth.copy()
    altered[:2] += 1000
    altered[6] = np.nan
    changed, _, _ = nested_banks(altered, names, cells, folds, references, edges, physical)
    for kind in banks:
        for key in banks[kind]:
            np.testing.assert_array_equal(banks[kind][key][:2], changed[kind][key][:2])
    assert all(not set(row['query_station_ids']) & set(row['library_station_ids']) for row in roles)


def test_invalid_hidden_source_and_pre_calendar_lag_fail():
    model, arrays, messages, hydro, cells = example()
    invalid = deepcopy(messages)
    invalid['river_valid'][0, 0, 0, 2] = True
    with pytest.raises(ValueError, match='pre-calendar'):
        river_cell_features(invalid, cells, hydro[:2], hydro)
    invalid = deepcopy(arrays)
    invalid['valid'][0, 0, 0] = False
    with pytest.raises(ValueError, match='hidden source'):
        model.predict(invalid, np.full(16, 3.))


def test_static_ignores_dynamic_inputs_while_dynamic_uses_them():
    original, arrays, _, _, _ = example()
    for mode in ('static_same_month', 'dynamic_lagged'):
        model = DynamicRiverResidual(**{**original.config, 'mode': mode})
        with torch.no_grad():
            model.output.weight.fill_(.2)
        changed = deepcopy(arrays)
        changed['query'] += 1.
        changed['features'] += .2
        equal = np.array_equal(model.predict(arrays, np.full(16, 3.)),
                               model.predict(changed, np.full(16, 3.)))
        assert equal == (mode == 'static_same_month')
