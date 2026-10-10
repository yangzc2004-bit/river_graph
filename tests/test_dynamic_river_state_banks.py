"""Additional source-fold and matching check after the frozen execution."""
from pathlib import Path

import numpy as np
from test_river_structure_residual import geography


def test_episode_libraries_exclude_unsorted_receiving_folds(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_doc_dynamic_river_state_v1 import state_matrices

    names = np.array(list('abcdefghi'))
    edges, physical = geography(names)
    physical['comid'] = np.arange(9)
    edges = edges.iloc[:4].copy()
    edges['source'], edges['target'] = list('abcd'), list('cdii')
    ids, val_ids = np.arange(8), np.array([8])
    months = 8
    cells = np.arange(8*months)
    folds = [np.array([2, 1]), np.array([3, 0]), np.array([7, 5]), np.array([6, 4])]
    # Each donor carries a unique identity, exposing accidental fold/order mixups.
    states = np.broadcast_to(np.arange(8)[:, None, None]+1., (8, months, 9)).copy()
    available, hydro = np.ones((8, months), bool), np.ones((8, months, 8))
    matrices, roles, _ = state_matrices(names, ids, val_ids, cells, np.arange(8*months, 9*months),
        folds, edges, physical, states, available, hydro, np.ones((1, months, 8)))
    for row in roles:
        assert not set(row['query_station_ids']) & set(row['library_station_ids'])
        selected = np.isin(cells//months, row['query_station_ids'])
        values = matrices['source_raw']['states'][selected][matrices['source_raw']['valid'][selected]]
        assert not np.isin(values[:, 0]-1, row['query_station_ids']).any()
    np.testing.assert_array_equal(matrices['validation_matched']['valid'], matrices['validation_control']['valid'])
    np.testing.assert_array_equal(matrices['source_raw']['states'][2*months+5, 0, 0], np.ones(9))
    np.testing.assert_array_equal(matrices['source_raw']['states'][3*months+5, 0, 0], np.ones(9)*2)
