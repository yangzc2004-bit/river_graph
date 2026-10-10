"""Label-free environmental messages and protected observed-state fusion."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_river_structure_residual import geography

from river_graph.models.dynamic_river_state import (
    DynamicRiverState,
    environmental_states,
    environmental_view,
    hydro_history_valid,
    state_cell_features,
    state_paths,
)
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)


def inputs():
    rng = np.random.default_rng(42)
    return {'raw': rng.normal(size=(3, 16, 23)).astype(np.float32),
        'extra': rng.normal(size=(3, 16, 41)).astype(np.float32),
        'age': rng.normal(size=(3, 16)).astype(np.float32),
        'support': rng.uniform(size=(3, 16, 3)).astype(np.float32),
        'env': rng.normal(size=(3, 9)).astype(np.float32)}


def example():
    rng = np.random.default_rng(42)
    n, t, c = 2, 8, 3
    paths = {'river_owner': np.tile([0, 1, -1], (n, 1)),
        'river_path': rng.uniform(size=(n, c, 8))}
    states = rng.normal(size=(3, t, 9)).astype(np.float32)
    hydro = rng.uniform(size=(3, t, 8))
    availability = np.ones((3, t), bool)
    arrays = state_cell_features(paths, np.arange(n*t), hydro[:n], hydro, states, availability)
    arrays['query'] = rng.normal(size=(n*t, 10)).astype(np.float32)
    model = DynamicRiverState(10, 36, 9, epochs=3, patience=3, batch_size=8)
    return model, arrays, paths, hydro, states, availability


def test_all_doc_channels_and_derived_support_are_removed():
    raw = inputs()
    before = environmental_view(raw)
    altered = deepcopy(raw)
    altered['raw'][..., 8:10] += 10000
    altered['raw'][..., 14:] += 10000
    for key in ('age', 'support', 'extra'):
        altered[key] += 10000
    after = environmental_view(altered)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    for index in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13):
        np.testing.assert_array_equal(before['raw'][..., index], raw['raw'][..., index])


def test_hydro_availability_uses_only_causal_twelve_months():
    raw, daily = np.zeros((1, 16, 23)), np.zeros((1, 16, 8))
    raw[0, 1, 1] = 1
    valid = hydro_history_valid(raw, daily)
    assert not valid[0, 0] and valid[0, 1:13].all() and not valid[0, 13:].any()
    daily[0, 14, 5] = 1
    changed = hydro_history_valid(raw, daily)
    np.testing.assert_array_equal(changed[:, :14], valid[:, :14])
    assert changed[0, 14:].all()


def test_state_indices_follow_real_lags_and_do_not_use_future():
    _, _, paths, hydro, states, available = example()
    before = state_cell_features(paths, np.arange(16), hydro[:2], hydro, states, available)
    altered, changed_hydro = states.copy(), hydro.copy()
    altered[:, 5:] += 10000
    changed_hydro[:, 5:] += 10000
    after = state_cell_features(paths, np.arange(16), changed_hydro[:2], changed_hydro, altered, available)
    for key in before:
        np.testing.assert_array_equal(before[key][np.arange(16) % 8 < 5], after[key][np.arange(16) % 8 < 5])
    for j, lag in enumerate((0, 1, 3)):
        np.testing.assert_array_equal(before['states'][5, 0, j], states[0, 5-lag])
    assert not before['valid'][0, :, 1:].any()


@pytest.mark.parametrize('allocation', ['dynamic', 'uniform'])
def test_zero_start_no_hydro_and_checkpoint_replay(allocation):
    _, arrays, *_ = example()
    model = DynamicRiverState(10, 36, 9, allocation=allocation, epochs=3, patience=3, batch_size=8)
    base = np.linspace(.3, 3., 16)
    np.testing.assert_array_equal(model.predict(arrays, base), base)
    model.fit(arrays, base, base+.2, arrays, base, base+.2, tail_threshold=2.)
    prediction, diag = model.predict(arrays, base, diagnostics=True)
    saved = DynamicRiverState.from_payload(model.to_payload())
    np.testing.assert_array_equal(saved.predict(arrays, base), prediction)
    np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., atol=1e-6)
    silent = {**arrays, 'valid': np.zeros_like(arrays['valid']), 'states': arrays['states']*0,
              'features': arrays['features']*0}
    np.testing.assert_array_equal(saved.predict(silent, base), base)


def test_hydro_controls_exclude_all_ancestors_and_same_reach_aliases():
    names = np.array(list('abcdefgh'))
    edges, physical = geography(names)
    # Two separate components provide downstream and disconnected controls.
    edges = edges.iloc[[0, 1, 3, 4, 5, 6]].copy()
    physical['comid'] = np.arange(len(names))
    real, control, records = state_paths(edges, physical, ['c'], names, np.ones((8, 16), bool), candidates=2)
    assert set(names[real['river_owner'][0]]) == {'a', 'b'}
    assert all(record['control_source'] not in set('abc') for record in records)
    np.testing.assert_array_equal(real['river_path'], control['river_path'])
    # Altering receiver/downstream state cannot affect actual upstream messages.
    states = np.ones((8, 16, 9), np.float32)
    hydro = np.ones((8, 16, 8))
    availability = np.ones((8, 16), bool)
    before = state_cell_features(real, np.arange(16), hydro[2:3], hydro, states, availability)
    states[2:] += 1000
    after = state_cell_features(real, np.arange(16), hydro[2:3], hydro, states, availability)
    np.testing.assert_array_equal(before['states'], after['states'])


def test_hidden_environmental_slots_must_be_zero():
    model, arrays, *_ = example()
    changed = deepcopy(arrays)
    changed['states'][0, 2, 0, 0] = 1.
    with pytest.raises(ValueError, match='must be zero'):
        model.predict(changed, np.ones(16))


CHECKPOINT = Path('experiments/phase4_transfer/doc_current_availability_attention_v1/runs/split142_seed42/available_real.pt')


@pytest.mark.skipif(not CHECKPOINT.exists(), reason='retained source encoder checkpoint absent')
def test_actual_frozen_encoder_is_label_free_and_prefix_causal():
    model = AvailableSourceAttentionResidual.from_payload(torch.load(CHECKPOINT, weights_only=False, map_location='cpu'))
    before = inputs()
    values, available = environmental_states(model, before, batch=20)
    changed = deepcopy(before)
    changed['raw'][..., 8:10] += 10000
    changed['raw'][..., 14:] += 10000
    changed['support'] += 10000
    changed['age'] += 10000
    changed['extra'][..., :30] += 10000
    changed['extra'][..., 38:] += 10000
    repeated, repeated_available = environmental_states(model, changed, batch=20)
    np.testing.assert_array_equal(values, repeated)
    np.testing.assert_array_equal(available, repeated_available)
    changed['raw'][:, 9:, :4] += 10000
    changed['extra'][:, 9:, 30:38] += 10000
    future, future_available = environmental_states(model, changed, batch=20)
    np.testing.assert_array_equal(values[:, :9], future[:, :9])
    np.testing.assert_array_equal(available[:, :9], future_available[:, :9])
