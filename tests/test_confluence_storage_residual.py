"""Physical mixing, causal source priors and compatibility with retained GNN."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_dynamic_river_residual import example

from river_graph.models.confluence_storage_residual import (
    ConfluenceStorageResidual,
    confluence_inputs,
)
from river_graph.models.dynamic_river_residual import DynamicRiverResidual
from river_graph.models.river_structure_residual import path_candidates


def branches():
    names = np.array(['a', 'b', 'c', 'r'])
    # a -> b -> r and c -> r: a and b are nested, c is a separate tributary.
    edges = pd.DataFrame({'source': ['a', 'b', 'c'], 'target': ['b', 'r', 'r'],
        'mainstem_connected': True, 'path_length_km': [10., 10., 12.],
        'path_major_junctions': [0., 1., 1.], 'path_storage_km': [0., 1., 2.]})
    physical = pd.DataFrame({'station': names, 'drainage_area_km2': [10., 20., 30., 60.],
        'largest_minor_area_share_5km': .2, 'stream_order': 3., 'storage_fraction_20km': .1})
    values = np.full((3, 8), .2)
    messages = path_candidates(edges, physical, ['r'], names[:3], values, candidates=3)
    return names, edges, physical, messages


def test_nested_catchments_not_double_counted_and_flow_prior():
    names, edges, physical, messages = branches()
    flow = np.broadcast_to(np.array([1., 2., 6.])[:, None], (3, 8)).copy()
    priors = confluence_inputs(messages, np.array([5]), names[:3], edges, physical, flow, np.ones_like(flow, bool))
    slots = messages['river_owner'][0]
    a, b, c = (int(np.where(slots == i)[0][0]) for i in range(3))
    assert priors['nested_suppressed'][0, a].all()
    assert not priors['frontier'][0, a].any()
    np.testing.assert_allclose(priors['mix_prior'][0, b], .25)
    np.testing.assert_allclose(priors['mix_prior'][0, c], .75)
    np.testing.assert_allclose(priors['mix_prior'].sum(1), 1.)
    assert priors['measured_flow'][0, b].all()
    missing = deepcopy(messages)
    missing['river_valid'][0, 5, b, 0] = False
    missing['river_values'][0, 5, b, 0] = 0.
    updated = confluence_inputs(missing, np.array([5]), names[:3], edges, physical, flow, np.ones_like(flow, bool))
    assert updated['frontier'][0, a, 0]  # upstream backup when closer observation absent
    assert not updated['frontier'][0, a, 1:].any()


@pytest.mark.parametrize('kind', ['missing', 'reverse', 'all_zero'])
def test_whole_group_area_fallback_is_not_a_fake_flow_measurement(kind):
    names, edges, physical, messages = branches()
    flow, mask = np.ones((3, 8)), np.ones((3, 8), bool)
    if kind == 'missing':
        mask[1] = False
        flow[1] = np.nan
    elif kind == 'reverse':
        flow[1] = -1.
    else:
        flow[:] = 0.
    result = confluence_inputs(messages, np.array([5]), names[:3], edges, physical, flow, mask)
    b = int(np.where(messages['river_owner'][0] == 1)[0][0])
    np.testing.assert_allclose(result['mix_prior'][0, b], .4, rtol=1e-6)
    assert not result['measured_flow'].any()


def test_future_flow_and_source_labels_cannot_change_earlier_priors():
    names, edges, physical, messages = branches()
    flow, mask = np.ones((3, 8)), np.ones((3, 8), bool)
    cells = np.arange(5)
    original = confluence_inputs(messages, cells, names[:3], edges, physical, flow, mask)
    flow[:, 5:] = 1000.
    messages['river_values'][:, 5:] = 5.
    repeated = confluence_inputs(messages, cells, names[:3], edges, physical, flow, mask)
    for key in original:
        np.testing.assert_array_equal(original[key], repeated[key])


def inputs():
    _, arrays, _, _, _ = example()
    arrays.update({'frontier': arrays['valid'].copy(),
        'mix_prior': arrays['valid'].astype(np.float32)/3,
        'measured_flow': arrays['valid'].copy(), 'nested_suppressed': np.zeros_like(arrays['valid'])})
    return arrays


@pytest.mark.parametrize('operator', ['plain', 'mixing', 'storage'])
def test_zero_start_normalization_empty_messages_and_roundtrip(operator):
    arrays = inputs()
    model = ConfluenceStorageResidual(10, 36, operator=operator, epochs=3, patience=3, batch_size=8)
    base = np.linspace(.01, 3, 16)
    np.testing.assert_array_equal(model.predict(arrays, base), base)
    model.fit(arrays, base, base+.5, arrays, base, base+.5, tail_threshold=2.)
    predicted, diag = model.predict(arrays, base, diagnostics=True)
    assert np.any(predicted != base)
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., atol=2e-6)
    loaded = ConfluenceStorageResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(loaded.predict(arrays, base), predicted)
    silent = deepcopy(arrays)
    for key in ('values', 'valid', 'frontier', 'mix_prior', 'measured_flow', 'nested_suppressed'):
        silent[key][:] = 0
    np.testing.assert_array_equal(loaded.predict(silent, base), base)


def test_plain_operator_is_matched_to_existing_dynamic_attention():
    arrays = inputs()
    old = DynamicRiverResidual(10, 36)
    new = ConfluenceStorageResidual(10, 36, operator='plain')
    with torch.no_grad():
        old.output.weight.fill_(.1)
        new.output.weight.copy_(old.output.weight)
    np.testing.assert_allclose(old.predict(arrays, np.ones(16)), new.predict(arrays, np.ones(16)), rtol=1e-7)


def test_storage_modulates_attenuation_and_lag_mass_before_readout():
    arrays = inputs()
    model = ConfluenceStorageResidual(10, 36, operator='storage')
    with torch.no_grad():
        model.output.weight.fill_(.1)
    _, low = model.predict(arrays, np.ones(16), diagnostics=True)
    high_storage = deepcopy(arrays)
    high_storage['features'][..., 2] += high_storage['valid'].astype(float)*2.
    # Freeze attention encoder's geometry weights to isolate explicit operator.
    with torch.no_grad():
        model.encoder[0].weight[:, 2].zero_()
    _, low = model.predict(arrays, np.ones(16), diagnostics=True)
    _, high = model.predict(high_storage, np.ones(16), diagnostics=True)
    assert high['attenuation_mean'].mean() < low['attenuation_mean'].mean()
    assert high['lag_mass'][:, -1].mean() > low['lag_mass'][:, -1].mean()


def test_invalid_frontier_cannot_open_hidden_target_observations():
    arrays = inputs()
    arrays['frontier'][0, 0, 2] = True
    with pytest.raises(ValueError, match='frontier'):
        ConfluenceStorageResidual(10, 36).predict(arrays, np.ones(16))
