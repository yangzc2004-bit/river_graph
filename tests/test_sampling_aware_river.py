"""Actual-date support, causal flow correspondence, and retained operator identity."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_dynamic_river_residual import example

from river_graph.models.sampling_aware_river import (
    SamplingAwareRiverResidual,
    monthly_sampling_metadata,
    sampling_cell_features,
    shuffle_hydro_timing,
)


def metadata_example():
    months = pd.date_range('2000-01-01', periods=8, freq='MS')
    sites = ['a', 'b', 'c', 'r']
    rows = pd.DataFrame([(s, m+pd.Timedelta(days=d)) for s in sites for m in months for d in (2, 2, 19)],
        columns=['site_no', 'date'])
    dates = pd.date_range('1999-12-20', '2000-08-31')
    daily = pd.DataFrame([(s, d, float(10+j % 19)) for s in sites for j, d in enumerate(dates)],
        columns=['site_no', 'date', 'discharge_cfs'])
    flow, mask = np.full((4, 8), 20.), np.ones((4, 8), bool)
    meta, phase, ends = monthly_sampling_metadata(rows, sites, months, flow, mask, daily)
    return rows, daily, flow, mask, meta, phase, ends, months


def test_result_weights_and_fixed_month_end_no_concentration_or_receiver_sample_dates():
    rows, daily, flow, mask, meta, phase, ends, months = metadata_example()
    first = pd.Timestamp('2000-01-01').to_numpy().astype('datetime64[D]').astype(int)
    assert meta[0, 0, 0] == pytest.approx(first+23/3)
    np.testing.assert_array_equal(meta[0, 0, 2:5], [17, 2, 3])
    assert ends[0] == pd.Timestamp('2000-01-31').to_numpy().astype('datetime64[D]').astype(int)
    rows['doc'] = 123456.
    changed = rows.copy()
    changed.loc[changed.site_no.eq('r'), 'date'] += pd.Timedelta(days=1)
    altered, same_phase, _ = monthly_sampling_metadata(changed, ['a', 'b', 'c', 'r'], months, flow, mask, daily)
    np.testing.assert_array_equal(meta[:3], altered[:3])
    np.testing.assert_array_equal(phase, same_phase)


def test_flow_gaps_reverse_and_duplicates_never_become_observed_phase():
    rows, daily, flow, mask, _, _, _, months = metadata_example()
    daily.loc[daily.site_no.eq('a'), 'discharge_cfs'] = -2.
    daily = daily[~(daily.site_no.eq('b') & daily.date.dt.day.eq(24))]
    mask[2] = False
    meta, phase, _ = monthly_sampling_metadata(rows, ['a', 'b', 'c', 'r'], months, flow, mask, daily)
    assert not meta[0, :, [5, 6, 7, 8]].any()
    assert not meta[2, :, [5, 6, 7, 8]].any()
    assert not phase[0].any() and not phase[2].any()
    with pytest.raises(ValueError, match='unique'):
        monthly_sampling_metadata(rows, ['a', 'b', 'c', 'r'], months, flow, mask,
            pd.concat([daily, daily.iloc[:1]]))


def test_hidden_and_future_metadata_do_not_enter_earlier_messages():
    _, _, messages, _, _ = example()
    _, _, _, _, meta, phase, ends, _ = metadata_example()
    visible = np.ones((4, 8), bool)
    visible[:, 1:3] = False
    before = sampling_cell_features(messages, np.array([3, 4]), meta, phase[:2], ends, visible)
    changed = meta.copy()
    changed[:, 1:3] = 1e9  # hidden sampling cells
    changed[:, 5:] = 1e9  # future cells
    after = sampling_cell_features(messages, np.array([3, 4]), changed, phase[:2], ends, visible)
    np.testing.assert_array_equal(before, after)
    illegal = meta.copy()
    illegal[:, 3, 1] = ends[3]+1
    with pytest.raises(ValueError, match='future'):
        sampling_cell_features(messages, np.array([3]), illegal, phase[:2], ends, visible)


def test_true_donor_date_not_matched_common_month_age():
    _, _, messages, _, _ = example()
    _, _, _, _, meta, phase, ends, _ = metadata_example()
    visible = np.ones((4, 8), bool)
    before = sampling_cell_features(messages, np.array([5]), meta, phase[:2], ends, visible)
    messages['river_age'][:] = 12
    after = sampling_cell_features(messages, np.array([5]), meta, phase[:2], ends, visible)
    np.testing.assert_array_equal(before, after)
    assert after[0, 0, 0, 0]*396 < 31


def test_shuffle_preserves_dates_values_receivers_and_causal_pool_marginals():
    _, arrays, _, _, _ = example()
    rng = np.random.default_rng(7)
    timing = rng.uniform(size=(*arrays['valid'].shape, 12)).astype(np.float32)
    timing[~arrays['valid']] = 0
    months, folds = np.tile(np.arange(8), 2), np.zeros(16, int)
    shuffled = shuffle_hydro_timing(timing, arrays['valid'], months, folds, 42)
    np.testing.assert_array_equal(shuffled[..., :5], timing[..., :5])
    np.testing.assert_array_equal(shuffled[..., [8, 11]], timing[..., [8, 11]])
    assert np.any(shuffled[..., 7] != timing[..., 7])
    for month in range(8):
        for lag in range(3):
            selected = (months == month)[:, None] & arrays['valid'][..., lag]
            np.testing.assert_array_equal(np.sort(shuffled[..., lag, 7][selected]), np.sort(timing[..., lag, 7][selected]))


@pytest.mark.parametrize('mode', ['none', 'age', 'hydro'])
def test_training_empty_bank_zero_start_and_checkpoint(mode):
    _, arrays, _, _, _ = example()
    arrays['timing'] = np.where(arrays['valid'][..., None], .2, 0.)*np.ones((*arrays['valid'].shape, 12), np.float32)
    model = SamplingAwareRiverResidual(10, 36, sampling_mode=mode, epochs=3, patience=3, batch_size=8)
    base = np.full(16, 3.)
    np.testing.assert_array_equal(model.predict(arrays, base), base)
    model.fit(arrays, base, base+.5, arrays, base, base+.5, tail_threshold=2.)
    pred, diag = model.predict(arrays, base, diagnostics=True)
    assert np.any(pred != base)
    np.testing.assert_allclose(diag['lag_mass'].sum(-1)+diag['prior_mass'], 1., atol=2e-6)
    loaded = SamplingAwareRiverResidual.from_payload(model.to_payload())
    np.testing.assert_array_equal(loaded.predict(arrays, base), pred)
    empty = deepcopy(arrays)
    for name in ('valid', 'values', 'timing'):
        empty[name][:] = 0
    np.testing.assert_array_equal(loaded.predict(empty, base), base)
    if mode == 'age':
        changed = deepcopy(arrays)
        changed['timing'][..., 5:] *= 100
        np.testing.assert_array_equal(model.predict(changed, base), pred)


def test_none_path_and_zero_sampling_weights_reproduce_original_operator():
    old, arrays, _, _, _ = example()
    arrays['timing'] = np.where(arrays['valid'][..., None], .3, 0.)*np.ones((*arrays['valid'].shape, 12), np.float32)
    new = SamplingAwareRiverResidual(10, 36, sampling_mode='hydro', epochs=3, patience=3, batch_size=8)
    with torch.no_grad():
        old.output.weight.fill_(.1)
        new.output.weight.copy_(old.output.weight)
    np.testing.assert_array_equal(old.predict(arrays, np.ones(16)), new.predict(arrays, np.ones(16)))
    old, _, _, _, _ = example()
    none = SamplingAwareRiverResidual(10, 36, sampling_mode='none', epochs=3, patience=3, batch_size=8)
    for model in (old, none):
        model.fit(arrays, np.ones(16), np.ones(16)+.3, arrays, np.ones(16), np.ones(16)+.3, tail_threshold=2.)
    np.testing.assert_array_equal(old.predict(arrays, np.ones(16)), none.predict(arrays, np.ones(16)))
