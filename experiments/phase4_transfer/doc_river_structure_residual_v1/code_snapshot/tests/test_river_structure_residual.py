"""Physical direction, donor isolation and joint river model contracts."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from test_current_source_attention import _model
from test_source_innovation_training import _nested_fixture

from river_graph.models.river_structure_residual import (
    RiverStructureResidual,
    nested_path_candidates,
    path_candidates,
    upstream_paths,
)
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)


def geography(names):
    structures = pd.DataFrame({"station": names, "drainage_area_km2": np.arange(1, len(names)+1)*100.,
        "largest_minor_area_share_5km": .3, "stream_order": 4., "storage_fraction_20km": .2})
    edges = pd.DataFrame({"source": names[:-1], "target": names[1:], "mainstem_connected": True,
        "path_length_km": 10., "path_major_junctions": 1., "path_storage_km": 2.})
    return edges, structures


def fixture(mode="structure", epochs=2):
    original, _, args = _model(epochs=epochs)
    old = AvailableSourceAttentionResidual(original.spatial, original.temporal, original.decay, **original._config())
    new = RiverStructureResidual(original.spatial, original.temporal, original.decay,
        river_mode=mode, **original._config())
    for i in (0, 4):
        inputs = args[i]
        n, t = inputs["age"].shape
        inputs["attention_reference"] = np.ones((n, t))*3
        inputs.update({"river_owner": np.tile([0, 1], (n, 1)), "river_path": np.full((n, 2, 8), .2),
            "river_values": np.zeros((n, t, 2, 3)), "river_valid": np.zeros((n, t, 2, 3), bool),
            "river_age": np.ones((n, t, 2, 3))*2})
        for j, lag in enumerate((0, 1, 3)):
            inputs["river_values"][:, lag:, :, j] = .5+j*.1
            inputs["river_valid"][:, lag:, :, j] = True
    return new, old, args


def test_paths_traverse_hidden_nodes_preserve_upstream_direction():
    edges, physical = geography(['a', 'b', 'c', 'd'])
    paths = upstream_paths(edges, physical, ['c', 'a'], ['a', 'd'])
    assert paths == [[('a', 20., 2., 4.)], []]
    reversed_edges = edges.rename(columns={'source': 'target', 'target': 'source'})
    assert upstream_paths(reversed_edges, physical, ['c'], ['a', 'd'])[0][0][0] == 'd'


def test_source_age_and_causality_missing_observations():
    edges, physical = geography(['a', 'b', 'c'])
    residual = np.full((3, 20), np.nan)
    residual[0, 2] = .7
    arrays = path_candidates(edges, physical, ['c'], ['a', 'b', 'c'], residual, eligible=['a'], candidates=2)
    assert arrays['river_valid'][0, 2, 0, 0]
    assert arrays['river_age'][0, 4, 0, 0] == 2
    assert not arrays['river_valid'][0, 15, 0, 0]
    assert not arrays['river_valid'][0, :3, :, 2].any()
    altered = residual.copy()
    altered[:, 8:] = 10.
    changed = path_candidates(edges, physical, ['c'], ['a', 'b', 'c'], altered, eligible=['a'], candidates=2)
    for name in ('river_values', 'river_valid', 'river_age'):
        np.testing.assert_array_equal(arrays[name][:, :8], changed[name][:, :8])


def test_nested_bank_excludes_receiver_fold_and_nonpermitted_truth():
    args = _nested_fixture()
    truth, names, _, _, cells, folds, references = args
    edges, physical = geography(names)
    arrays, records = nested_path_candidates(truth, names, cells, folds, references, edges, physical, candidates=2)
    changed = deepcopy(truth)
    changed[:2] += 1000
    changed[6] = np.nan
    altered, _ = nested_path_candidates(changed, names, cells, folds, references, edges, physical, candidates=2)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name][:2], altered[name][:2])
    assert not np.isin(arrays['river_owner'][:2], [0, 1]).any()
    assert all(not set(row['query_station_ids']) & set(row['library_station_ids']) for row in records)
    bad = deepcopy(references)
    bad[(0, 1)]['fitted_stations'].append(0)
    with pytest.raises(ValueError, match='query AND donor'):
        nested_path_candidates(truth, names, cells, folds, bad, edges, physical, candidates=2)


def test_rewiring_matches_slots_and_excludes_true_ancestors():
    names = np.array(['a', 'b', 'c', 'd', 'e'])
    edges, physical = geography(names)
    edges = edges.iloc[:1].copy()  # a -> b only
    residual = np.arange(5*6.).reshape(5, 6)
    real = path_candidates(edges, physical, ['b'], names, residual, candidates=2)
    rewire = path_candidates(edges, physical, ['b'], names, residual, candidates=2, rewired=True)
    np.testing.assert_array_equal(real['river_path'], rewire['river_path'])
    assert (real['river_owner'] >= 0).sum() == (rewire['river_owner'] >= 0).sum() == 1
    assert rewire['river_owner'][0, 0] in [2, 3, 4]


def test_zero_initialization_matches_existing_fitted_local_and_retrieval_head():
    new, old, args = fixture()
    with torch.no_grad():
        old.head.linear.weight.fill_(.05)
        old.head.output.weight.fill_(.1)
        for name, value in old.head.state_dict().items():
            new.head.state_dict()[name].copy_(value)
    np.testing.assert_array_equal(new.predict_delta(args[4]), old.predict_delta(args[4]))
    diagnostic = new.river_diagnostics(args[4], np.array([0, 4, 15]))
    np.testing.assert_allclose(diagnostic['river_prior_mass']+diagnostic['river_lag_mass'].sum(-1), 1.)


@pytest.mark.parametrize('mode', ['none', 'simple', 'structure'])
def test_joint_training_checkpoint_new_nodes_and_future_information(mode):
    new, old, args = fixture(mode)
    new.fit(*args, tail_threshold=4., selection_role='source_validation')
    assert new.optimizer_steps_ > 0
    assert (new.to_dict()['river_output_norm'] > 0) == (mode != 'none')
    loaded = RiverStructureResidual.from_payload(new.to_payload())
    np.testing.assert_array_equal(new.predict(args[4], args[5]), loaded.predict(args[4], args[5]))
    changed = deepcopy(args[4])
    for name in ('raw', 'extra', 'river_values'):
        changed[name][:, 8:] += 30
    changed['river_values'][~changed['river_valid']] = 0
    changed['donor_hydro_bank'][:, 8:] += 100
    np.testing.assert_array_equal(new.predict_delta(args[4])[:, :8], new.predict_delta(changed)[:, :8])
    extended = deepcopy(args[4])
    for key in extended:
        if key != 'donor_hydro_bank':
            extended[key] = np.concatenate([extended[key], extended[key][:1]])
    assert loaded.predict_delta(extended).shape == (3, 14)
    if mode == 'none':
        old.fit(*args, tail_threshold=4., selection_role='source_validation')
        np.testing.assert_allclose(old.predict(args[4], args[5]), new.predict(args[4], args[5]), rtol=0, atol=1e-12)


def test_absent_river_message_exact_zero_even_after_fitting():
    new, _, args = fixture()
    with torch.no_grad():
        new.head.river.output.weight.fill_(1)
    hidden = deepcopy(args[4])
    hidden['river_valid'][:] = False
    hidden['river_values'][:] = 0
    diag = new.river_diagnostics(hidden, np.array([0, 4, 15]))
    np.testing.assert_array_equal(diag['river_delta_unscaled'], np.zeros(3))
    np.testing.assert_array_equal(diag['river_prior_mass'], np.ones(3))
    assert np.isfinite(diag['river_entropy']).all()


def test_path_conditioning_changes_scores_and_simple_ignores_path():
    for mode in ('simple', 'structure'):
        new, _, args = fixture(mode)
        with torch.no_grad():
            new.head.river.output.weight.fill_(.2)
        changed = deepcopy(args[4])
        changed['river_path'][:, :, :4] += .5
        equal = np.array_equal(new.predict_delta(args[4]), new.predict_delta(changed))
        assert equal == (mode == 'simple')
