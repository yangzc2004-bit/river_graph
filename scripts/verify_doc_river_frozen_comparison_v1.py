"""Replay all four fixed river procedures and their source/receiver roles."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_river_frozen_comparison_v1 import ARMS, BASE_NAME, ROOT, features, prepare
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS


def verify(run, runtime):
    config = json.loads((run/'config.json').read_text())
    if config['runtime_snapshot_hash'] != runtime:
        raise ValueError('execution snapshot differs across packages')
    for record in ('point_complete.json', 'complete.json'):
        verify_files(run, record, config)
    p = prepare(config['target_huc4'], config['seed'])
    parent = p['prior'].parent.parent  # The source base is separately bound below.
    if not parent.exists():
        raise ValueError('saved complete model root absent')
    old_parent = type(run)(p['old']['parent_run'])
    verify_files(old_parent, 'complete.json', json.loads((old_parent/'config.json').read_text()))
    for physical, expected in config['physical_hashes'].items():
        if sha256_file(physical) != expected:
            raise ValueError('physical river inputs changed')
    with np.load(run/'message_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_ids'], p['source_ids'])
        np.testing.assert_array_equal(saved['receivers'], p['receivers'])
        np.testing.assert_array_equal(saved['raw_valid'], p['raw_valid'])
        for kind in ('true', 'fake'):
            for key, value in p[kind].items():
                np.testing.assert_array_equal(saved[f'{kind}_{key}'], value)
    np.testing.assert_array_equal(p['true']['river_valid'], p['fake']['river_valid'])
    np.testing.assert_array_equal(p['true']['river_age'], p['fake']['river_age'])
    target = np.unique(p['split']['test']//p['base'].shape[1])
    if np.intersect1d(p['source_ids'], target).size:
        raise ValueError('target station entered source bank')
    grids = {'current_complete': p['base']}
    with np.load(run/'components.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['current_complete'], p['base'])
        for arm in ARMS[1:]:
            model = joblib.load(run/f'{arm}.joblib')
            grid = p['base'].copy()
            for role in ('val', 'test'):
                x, support = features(p, role, arm)
                cells = p['split'][role]
                pred = model.predict(x, p['base'].ravel()[cells])
                np.testing.assert_array_equal(pred[~support], p['base'].ravel()[cells][~support])
                grid.ravel()[cells] = pred
            np.testing.assert_array_equal(grid, saved[arm])
            grids[arm] = grid
    data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
    names, dates, truth = np.asarray(data['site_no'], str), np.asarray(data['months'], str), np.asarray(data['y'])
    t = truth.shape[1]
    counts = {}
    for name in ('predictions.parquet', 'support_curves.parquet'):
        frame = pd.read_parquet(run/name)
        meta = json.loads((run/name.replace('.parquet', '.meta.json')).read_text())
        if (meta['config_hash'] != digest(config) or meta['prediction_sha256'] != sha256_file(run/name)
                or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
            raise ValueError('prediction sidecar cannot be reproduced')
        if set(frame.model_name) != set(ARMS) or not frame.visibility_role.eq('test').all():
            raise ValueError('four-arm test panel incomplete')
        if frame.duplicated(['model_name', 'k', 'cell']).any():
            raise ValueError('duplicate receiving cell')
        reference = None
        for (_, _), group in frame.groupby(['model_name', 'k']):
            cells = group.cell.to_numpy()
            if reference is None:
                reference = cells
            np.testing.assert_array_equal(cells, reference)
            np.testing.assert_array_equal(group.y_true, truth.ravel()[cells])
            np.testing.assert_array_equal(group.station, names[cells//t])
            np.testing.assert_array_equal(group.month, dates[cells % t])
            if not np.isfinite(group.y_pred).all() or (group.y_pred < 0).any():
                raise ValueError('invalid DOC prediction')
        if name == 'predictions.parquet':
            np.testing.assert_array_equal(reference, p['split']['test'])
            for arm in ARMS:
                group = frame[frame.model_name.eq(arm)]
                np.testing.assert_array_equal(group.y_pred, grids[arm].ravel()[group.cell])
            prior = pd.read_parquet(p['prior']/name)
            prior = prior[prior.model_name.eq(BASE_NAME)].sort_values('cell')
            now = frame[frame.model_name.eq('current_complete')].sort_values('cell')
            np.testing.assert_array_equal(now.y_pred, prior.y_pred)
        counts[name] = len(reference)
    return {'run': run.name, 'prediction_replay': 'bitwise', 'source_target_station_overlap': 0,
        'support_and_age_matched': True, 'baseline_unchanged': True, 'query_counts': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    snapshot = json.loads((args.root/'runtime_snapshot.json').read_text())
    for name, expected in snapshot.items():
        if sha256_file(args.root/'code_snapshot'/name) != expected:
            raise ValueError('saved execution source changed')
    rows = [verify(args.root/'runs'/f'huc4_{region}_seed{seed}', digest(snapshot))
        for region in HUC4_BLOCKS for seed in (42, 43, 44)]
    write_json(args.root/'verification'/'replay.json', {'packages': rows,
        'n_procedure_results': 4*len(rows), 'verifier_sha256': sha256_file(__file__)})
    print(f'Verified {len(rows)} packages / {4*len(rows)} procedure results', flush=True)


if __name__ == '__main__':
    main()
