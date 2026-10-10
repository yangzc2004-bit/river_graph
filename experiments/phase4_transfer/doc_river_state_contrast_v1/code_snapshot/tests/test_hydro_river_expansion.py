"""Hydro-only node inputs, fixed preprocessing and whole-receiver exclusion."""
from pathlib import Path

import numpy as np
import torch
from test_river_structure_residual import geography

from river_graph.models.hydro_river_expansion import (
    FrozenHydroPreprocessing,
    eligible_hydro_nodes,
)


def data():
    rng = np.random.default_rng(42)
    return {'x': rng.normal(size=(9, 8, 2)).astype(np.float32),
        'x_mask': np.ones((9, 8, 2), np.float32),
        'static': rng.normal(size=(9, 2)).astype(np.float32),
        'regime': rng.normal(size=(9, 13)).astype(np.float32),
        'months': [f'2020-{m:02d}-01' for m in range(1, 9)],
        'y': rng.uniform(size=(9, 8)), 'y_mask': np.ones((9, 8), bool)}


def test_input_never_reads_water_quality_or_its_masks():
    dataset = data()
    cells, daily = np.arange(6*8), np.ones((9, 8, 8), np.float32)
    processing = FrozenHydroPreprocessing.fit(dataset, cells)
    baseline = processing.transform(dataset, daily)
    for key in ('y', 'y_mask', 'ph', 'conductance'):
        dataset[key] = object()  # Any attempt to interpret chemistry must fail.
    changed = FrozenHydroPreprocessing.fit(dataset, cells)
    assert changed.statistics == processing.statistics
    for name, array in changed.transform(dataset, daily).items():
        np.testing.assert_array_equal(array, baseline[name])
    assert not baseline['raw'][..., 8:10].any()
    assert not baseline['raw'][..., 14].any()
    assert not baseline['raw'][..., 16:].any()
    assert not baseline['support'].any()


def test_saved_preprocessing_preserves_prefix_when_future_hydro_changes():
    dataset = data()
    daily = np.ones((9, 8, 8), np.float32)
    processing = FrozenHydroPreprocessing.fit(dataset, np.arange(6*8))
    baseline = processing.transform(dataset, daily)
    dataset['x'][:, 5:] += 123
    daily[:, 5:] = 0
    restored = FrozenHydroPreprocessing(processing.statistics)
    after = restored.transform(dataset, daily)
    for key in ('raw', 'age', 'support', 'extra'):
        np.testing.assert_array_equal(after[key][:, :5], baseline[key][:, :5])
    np.testing.assert_array_equal(after['env'], baseline['env'])


def test_environmental_input_matches_existing_torch_preprocessing():
    from river_graph.experiments.temporal_h2x import build_temporal_inputs

    dataset = data()
    dataset['y'], dataset['y_mask'] = torch.ones(9, 8), torch.ones(9, 8, dtype=torch.bool)
    split = {'train': np.arange(6*8), 'val': np.arange(6*8, 7*8), 'test': np.arange(7*8, 9*8)}
    inputs = build_temporal_inputs(dataset, split, target_transform='log1p')
    processor = FrozenHydroPreprocessing.fit(dataset, split['train'])
    full = processor.transform(dataset, np.zeros((9, 8, 8)))
    np.testing.assert_array_equal(full['raw'][..., :14], inputs.xt_static.permute(1, 0, 2).numpy())
    np.testing.assert_array_equal(full['env'], inputs.env_raw.numpy())


def test_expanded_bank_excludes_entire_fold_and_retains_unlabelled_nodes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_doc_hydro_river_expansion_v1 import state_matrices

    names = np.array(list('abcdefghi'))
    edges, physical = geography(names)
    physical['comid'] = np.arange(9)
    edges = edges.iloc[:4].copy()
    edges['source'], edges['target'] = list('agbd'), list('cdii')
    ids, validation = np.arange(6), np.array([8])
    t = 8
    cells = np.arange(len(ids)*t)
    folds = [np.array([2, 1]), np.array([3, 0]), np.array([5, 4])]
    states = np.broadcast_to(np.arange(9)[:, None, None]+1., (9, t, 9)).copy()
    matrices, roles, _ = state_matrices(names, ids, validation, cells, np.arange(8*t, 9*t),
        folds, edges, physical, states, np.ones((9, t), bool), np.ones((9, t, 8)))
    for row in roles:
        assert set(row['library_station_ids']) == set(eligible_hydro_nodes(9, row['query_station_ids']))
        assert not set(row['query_station_ids']) & set(row['library_station_ids'])
        assert 6 in row['library_station_ids']  # No source-training DOC role.
    np.testing.assert_array_equal(matrices['source_raw']['states'][3*t+5, 0, 0], np.ones(9)*7)
    np.testing.assert_array_equal(matrices['source_raw']['states'][2*t+5, 0, 0], np.ones(9))
    for role in ('source', 'validation'):
        np.testing.assert_array_equal(matrices[f'{role}_matched']['valid'], matrices[f'{role}_control']['valid'])
        np.testing.assert_array_equal(matrices[f'{role}_matched']['features'][..., :8],
                                     matrices[f'{role}_control']['features'][..., :8])
