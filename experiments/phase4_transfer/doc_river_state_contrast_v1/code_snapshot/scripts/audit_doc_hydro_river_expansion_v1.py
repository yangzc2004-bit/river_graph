"""Audit genuine upstream hydro coverage before fitting the expanded operator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_dynamic_river_v1 import ATLAS, REFERENCE
from run_unified_doc_spatial import write_json

from river_graph.models.dynamic_river_state import hydro_history_valid
from river_graph.models.hydro_river_expansion import (
    FrozenHydroPreprocessing,
    eligible_hydro_nodes,
)
from river_graph.models.river_structure_residual import LAGS, upstream_paths

ROOT = Path('experiments/phase4_transfer/doc_hydro_river_expansion_v1')
DAILY = Path('experiments/phase4_transfer/doc_daily_hydro_residual_v1')


def path_support(names, receivers, donors, available, edges, physical, cells):
    """Coverage of allowed dates, including receivers with no mapped ancestor."""
    paths = upstream_paths(edges, physical, names[receivers], names[donors])
    t = available.shape[1]
    lookup = {name: i for i, name in enumerate(names)}
    owner = np.full((len(receivers), 20), -1, int)
    nearest = np.full(len(receivers), np.nan)
    for row, candidates in enumerate(paths):
        for slot, (name, km, _, _) in enumerate(candidates):
            owner[row, slot] = lookup[name]
            if slot == 0:
                nearest[row] = km
    row = np.searchsorted(receivers, cells//t)
    dates = cells[:, None, None] % t-np.asarray(LAGS)[None, None]
    candidates = owner[row, :, None]
    valid = (candidates >= 0) & (dates >= 0) & available[candidates.clip(min=0), dates.clip(min=0)]
    return valid.any((1, 2)), nearest[row], owner


def audit(root=ROOT):
    physical = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    rows, nodes = [], []
    for partition in (142, 143, 144):
        config = json.loads((REFERENCE/'runs'/f'split{partition}_seed42/config.json').read_text())
        data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
        names = np.asarray(data['site_no'], str)
        with np.load(config['mask_path']) as saved:
            source, val = saved['train'], saved['val']
        t = np.shape(data['x'])[1]
        ids, receivers = np.unique(source//t), np.unique(val//t)
        daily, _, _ = load_daily_pack(DAILY, config['dataset_hash'], (len(names), t))
        preprocessing = FrozenHydroPreprocessing.fit(data, source)
        inputs = preprocessing.transform(data, daily)
        # Verify the retained preprocessing using saved environmental channels.
        with np.load(Path(config['source_run'])/'joint_source_inputs.npz') as saved:
            np.testing.assert_array_equal(inputs['raw'][ids, :, :14], saved['raw'][..., :14] * np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.float32))
            np.testing.assert_array_equal(inputs['env'][ids], saved['env'])
            np.testing.assert_array_equal(inputs['extra'][ids, :, 30:38], saved['extra'][..., 30:38])
        available = hydro_history_valid(inputs['raw'], daily)
        expanded = eligible_hydro_nodes(len(names), receivers)
        prior, distance, _ = path_support(names, receivers, ids, available, edges, physical, val)
        full, full_distance, owner = path_support(names, receivers, expanded, available, edges, physical, val)
        new_donors = np.setdiff1d(expanded, ids)
        rows.append({'partition': partition, 'query_cells': len(val), 'receiving_stations': len(receivers),
            'source_nodes': len(ids), 'expanded_nodes': len(expanded), 'new_nodes': len(new_donors),
            'prior_fraction': float(prior.mean()), 'expanded_fraction': float(full.mean()),
            'new_support_cells': int((full & ~prior).sum()), 'lost_support_cells': int((prior & ~full).sum()),
            'prior_path_fraction': float(np.isfinite(distance).mean()),
            'expanded_path_fraction': float(np.isfinite(full_distance).mean())})
        for node in new_donors:
            nodes.append({'partition': partition, 'station': names[node],
                'hydro_months': int(available[node].sum()), 'used_by_receivers': int((owner == node).any(1).sum())})
    output = root/'audit'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'coverage.csv', index=False)
    pd.DataFrame(nodes).to_csv(output/'added_nodes.csv', index=False)
    write_json(output/'definition.json', {'uses_DOC_values': False, 'uses_DOC_masks_as_features': False,
        'query_identities': 'frozen source-validation cells; labels not read for coverage',
        'pool': 'all ST357 covariate-known nodes except receiving role/fold',
        'scope': 'known-atlas transductive environmental covariates, not external inductive validation',
        'lags': list(LAGS), 'candidates': 20, 'max_path_km': 3000,
        'validity': 'measured hydro in causal preceding 12 months', 'preprocessing_replay': 'exact'})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    audit()
