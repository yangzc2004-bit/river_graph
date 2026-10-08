"""Source matching, causal messages and frozen-base correction contracts."""
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from river_graph.models.frozen_river_correction import (
    FrozenRiverCorrection,
    history_bank,
    matched_messages,
    message_features,
)
from river_graph.models.relative_source_attention import relative_source_residual_grid


def fixture():
    names = np.array(['a', 'alias', 'b', 'c', 'd', 'r'])
    physical = pd.DataFrame({'station': names, 'comid': [1, 1, 2, 3, 4, 5],
        'drainage_area_km2': [100., 100., 200., 110., 220., 500.],
        'largest_minor_area_share_5km': .3, 'stream_order': 4., 'storage_fraction_20km': .2})
    edges = pd.DataFrame({'source': ['a', 'b'], 'target': ['b', 'r'],
        'mainstem_connected': True, 'path_length_km': [3100., 10.],
        'path_major_junctions': 1., 'path_storage_km': 2.})
    residual = np.full((5, 24), np.nan)
    residual[:, ::3] = np.arange(1., 6.)[:, None]
    residual[3, :3] = np.nan
    return edges, physical, names, residual


def test_history_never_reads_future_and_expired_values_are_invalid():
    residual = np.full((1, 22), np.nan)
    residual[0, 2] = .7
    value, age, valid = history_bank(residual)
    assert valid[0, 2, 0] and age[0, 4, 0] == 2
    assert not valid[0, 15, 0]
    assert not valid[0, :3, 2].any()
    assert (value[~valid] == 0).all()
    changed = residual.copy()
    changed[:, 10:] = 88.
    for old, new in zip((value, age, valid), history_bank(changed), strict=True):
        np.testing.assert_array_equal(old[:, :10], new[:, :10])


def test_matching_excludes_uncapped_ancestors_and_shared_reach_aliases():
    edges, physical, names, residual = fixture()
    real, fake, records, raw = matched_messages(edges, physical, ['r'], names[:5], residual, candidates=2)
    # a is >3000 km away but remains forbidden; alias shares a's COMID.
    assert {r['control_source'] for r in records} <= {'c', 'd'}
    assert {r['real_source'] for r in records} == {'b'}
    np.testing.assert_array_equal(real['river_valid'], fake['river_valid'])
    np.testing.assert_array_equal(real['river_age'], fake['river_age'])
    np.testing.assert_array_equal(real['river_path'], fake['river_path'])
    assert np.all(real['river_valid'] <= raw)
    assert not np.array_equal(real['river_values'], fake['river_values'])


def test_bank_ignores_receiver_labels_and_nonpermitted_source_cells():
    truth = np.arange(40., dtype=float).reshape(5, 8)
    oof = np.ones_like(truth)
    cells = np.array([0, 2, 8, 11])
    ids, original = relative_source_residual_grid(truth, oof, cells)
    altered = truth.copy()
    altered.ravel()[np.setdiff1d(np.arange(40), cells)] = 100000
    ids2, changed = relative_source_residual_grid(altered, oof, cells)
    np.testing.assert_array_equal(ids, ids2)
    np.testing.assert_array_equal(original, changed)
    assert set(ids) == {0, 1}


def test_features_zero_without_support_and_simple_ignores_structure():
    edges, physical, names, residual = fixture()
    real, _, _, _ = matched_messages(edges, physical, ['r'], names[:5], residual, candidates=2)
    cells = np.arange(24)
    base, daily, morphology = np.ones((1, 24))*3, np.ones((1, 24, 8)), np.ones((1, 7))
    simple, support = message_features(real, cells, base, daily, morphology, structured=False)
    changed = deepcopy(real)
    changed['river_path'] += .2
    simple2, _ = message_features(changed, cells, base, daily, morphology*2, structured=False)
    np.testing.assert_array_equal(simple, simple2)
    structured, _ = message_features(real, cells, base, daily, morphology, structured=True)
    structured2, _ = message_features(changed, cells, base, daily, morphology*2, structured=True)
    assert not np.array_equal(structured, structured2)
    assert (simple[~support] == 0).all()
    hidden = deepcopy(real)
    hidden['river_valid'][:] = False
    x, supported = message_features(hidden, cells, base, daily, morphology, structured=True)
    assert not supported.any() and not x.any()


def test_regularization_uses_station_cv_and_preserves_fixed_base(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 4))
    stations = np.repeat(np.arange(6), 20)
    base = np.full(120, 5.)
    truth = base+x@np.array([.3, -.4, .1, .5])
    model = FrozenRiverCorrection().fit(x, truth, base, stations)
    assert model.alpha_ is not None
    assert np.abs(model.predict(x, base)-truth).mean() < .03
    assert model.to_dict()['n_calibration_stations'] == 6
    joblib.dump(model, tmp_path/'model.joblib')
    loaded = joblib.load(tmp_path/'model.joblib')
    np.testing.assert_array_equal(model.predict(x, base), loaded.predict(x, base))
    np.testing.assert_array_equal(model.predict(np.zeros_like(x), base), base)
    np.testing.assert_array_equal(base, np.full(120, 5.))


def test_zero_option_retained_when_no_useful_calibration_information():
    x, base = np.ones((18, 4)), np.arange(18.)+1
    model = FrozenRiverCorrection().fit(x, base, base, np.repeat(np.arange(3), 6))
    assert model.alpha_ is None and not model.coef_.any()
    np.testing.assert_array_equal(model.predict(x, base), base)
