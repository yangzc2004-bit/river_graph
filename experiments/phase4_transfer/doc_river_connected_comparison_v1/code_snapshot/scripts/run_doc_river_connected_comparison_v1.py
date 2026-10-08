"""Nested station-held comparison where upstream support can be learned."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT
from run_doc_river_frozen_comparison_v1 import (
    ARMS,
    ATLAS,
    BASE_NAME,
    GEOMETRY,
    morphology_view,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.frozen_river_correction import (
    FrozenRiverCorrection,
    matched_messages,
    message_features,
)
from river_graph.models.relative_source_attention import relative_source_residual_grid

ROOT = Path('experiments/phase4_transfer/doc_river_connected_comparison_v1')
REFERENCE = Path('experiments/phase4_transfer/doc_current_availability_attention_v1')
CODE = ('scripts/run_ladder.py', 'scripts/run_doc_river_connected_comparison_v1.py',
        'scripts/run_doc_river_frozen_comparison_v1.py', 'src/river_graph/models/frozen_river_correction.py',
        'src/river_graph/models/river_structure_residual.py', 'src/river_graph/models/relative_source_attention.py')


def receiver_folds(ids, cells, support, t, seed):
    """Balance observed source support without reading any receiving DOC."""
    rng = np.random.default_rng(seed)
    supported = np.unique(cells[support]//t)
    folds = [[] for _ in range(3)]
    offset = 0
    for group in (supported, np.setdiff1d(ids, supported)):
        for number, station in enumerate(rng.permutation(group)):
            folds[(number+offset) % 3].append(int(station))
        offset = (offset+len(group)) % 3
    return [np.sort(fold) for fold in folds]


def prepare(partition, seed):
    prior = REFERENCE/'runs'/f'split{partition}_seed{seed}'
    old = json.loads((prior/'config.json').read_text())
    verify_files(prior, 'complete.json', old)
    for kind in ('dataset', 'mask'):
        if sha256_file(old[f'{kind}_path']) != old[f'{kind}_hash']:
            raise ValueError('saved station task inputs changed')
    dataset = torch.load(old['dataset_path'], weights_only=False, map_location='cpu')
    names, t = np.asarray(dataset['site_no'], str), dataset['y'].shape[1]
    with np.load(old['mask_path'], allow_pickle=False) as saved:
        source, query = saved['train'].copy(), saved['val'].copy()
    source_truth = np.full_like(np.asarray(dataset['y']), np.nan)
    source_truth.ravel()[source] = np.asarray(dataset['y']).ravel()[source]
    parent = Path(old['parent_run'])
    verify_files(parent, 'complete.json', json.loads((parent/'config.json').read_text()))
    with np.load(parent/'source_oof.npz', allow_pickle=False) as saved:
        ids, residual = relative_source_residual_grid(source_truth, saved['pred_z'], source)
    receiver_ids = np.unique(query//t)
    if np.intersect1d(receiver_ids, ids).size:
        raise ValueError('held receiving station in donor bank')
    prior_frame = pd.read_parquet(prior/'predictions.parquet')
    prior_frame = prior_frame[prior_frame.model_name.eq(BASE_NAME)]
    np.testing.assert_array_equal(prior_frame.cell, query)
    base = np.zeros_like(source_truth, dtype=float)
    base.ravel()[query] = prior_frame.y_pred.to_numpy()
    true, fake, records, raw_valid = matched_messages(
        pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str}),
        pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str}),
        names[receiver_ids], names[ids], residual)
    daily, _, _ = load_daily_pack(DAILY_ROOT, old['dataset_hash'], base.shape)
    morphology, normal = morphology_view(names, ids)
    local_cells = np.searchsorted(receiver_ids, query//t)*t+query % t
    matrices, support = {}, None
    for arm in ARMS[1:]:
        matrices[arm], current = message_features(fake if arm == 'matched_nonupstream' else true,
            local_cells, base[receiver_ids], daily[receiver_ids], morphology[receiver_ids],
            structured=arm != 'simple_upstream')
        if support is not None:
            np.testing.assert_array_equal(current, support)
        support = current
    return {'old': old, 'prior': prior, 'dataset': dataset, 'names': names, 'source_ids': ids,
        'receiver_ids': receiver_ids, 'cells': query, 'base': base.ravel()[query],
        'matrices': matrices, 'support': support, 'true': true, 'fake': fake, 'raw_valid': raw_valid,
        'matching': records, 'normal': normal,
        'folds': receiver_folds(receiver_ids, query, support, t, partition)}


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('saved nested river execution changed')
        verify_files(run, 'complete.json', config)
        return
    p = prepare(partition, seed)
    old, data, cells = p['old'], p['dataset'], p['cells']
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'q90_threshold_train', 'split_seed', 'seed')}, 'started_at': datetime.now(timezone.utc).isoformat(),
        'experiment': ROOT.name, 'runtime_snapshot_hash': runtime, 'models': ARMS,
        'source_run': str(p['prior']), 'source_completion_hash': sha256_file(p['prior']/'complete.json'),
        'evaluation_role': 'nested station-held source-validation development; base validation previously seen',
        'selection_role': 'inner station CV excludes outer held stations',
        'reason_for_extension': 'whole-region study had zero/one supported calibration station and zero-selecting ties',
        'base_frozen': True, 'base_previously_used_all_receiving_labels_for_selection': True,
        'independent_confirmation_claimed': False, 'support_balancing_uses_receiving_DOC': False,
        'lags': [0, 1, 3], 'alphas': FrozenRiverCorrection.ALPHAS, 'zero_option': True,
        'physical_hashes': {str(folder/name): sha256_file(folder/name) for folder, name in (
            (ATLAS, 'station_edge_paths.csv'), (ATLAS, 'station_structure.csv'),
            (GEOMETRY, 'station_planform.csv'), (GEOMETRY, 'station_classes.csv'))}}
    if (run/'config.json').exists():
        previous = json.loads((run/'config.json').read_text())
        config['started_at'] = previous['started_at']
        if previous != json.loads(json.dumps(config)):
            raise ValueError('incomplete nested package belongs to another design')
    write_json(run/'config.json', config)
    write_json(run/'receiver_roles.json', {'source_station_ids': p['source_ids'].tolist(),
        'outer_folds': [fold.tolist() for fold in p['folds']], 'labels_used_for_folds': False})
    write_json(run/'morphology_normalization.json', p['normal'])
    pd.DataFrame(p['matching']).to_csv(run/'matched_donors.csv', index=False)
    np.savez_compressed(run/'message_inputs.npz', receivers=p['receiver_ids'], source_ids=p['source_ids'],
        raw_valid=p['raw_valid'], **{f'{kind}_{key}': value for kind in ('true', 'fake') for key, value in p[kind].items()})
    np.savez_compressed(run/'features.npz', cells=cells, base=p['base'], support=p['support'], **p['matrices'])
    files = ['config.json', 'receiver_roles.json', 'morphology_normalization.json', 'matched_donors.csv',
             'message_inputs.npz', 'features.npz']
    t = data['y'].shape[1]
    prediction = {'current_complete': p['base'].copy()}
    states = []
    for arm in ARMS[1:]:
        result = np.empty(len(cells))
        for fold, held in enumerate(p['folds']):
            query = np.isin(cells//t, held)
            fit = ~query
            # No outer-query labels are passed to the correction fit.
            calibration_y = np.asarray(data['y']).ravel()[cells[fit]]
            model = FrozenRiverCorrection().fit(p['matrices'][arm][fit], calibration_y,
                p['base'][fit], p['names'][cells[fit]//t])
            result[query] = model.predict(p['matrices'][arm][query], p['base'][query])
            np.testing.assert_array_equal(result[query & ~p['support']], p['base'][query & ~p['support']])
            joblib.dump(model, run/f'{arm}_fold{fold}.joblib')
            files.append(f'{arm}_fold{fold}.joblib')
            states.append({'model_name': arm, 'outer_fold': fold, **model.to_dict(),
                'held_station_ids': held.tolist(), 'calibration_station_ids': np.unique(cells[fit]//t).tolist(),
                'supported_calibration_stations': len(np.unique(cells[fit & p['support']]//t)),
                'supported_query_stations': len(np.unique(cells[query & p['support']]//t))})
        prediction[arm] = result
    write_json(run/'readout_states.json', states)
    files.append('readout_states.json')
    np.savez_compressed(run/'point_predictions.npz', **prediction)
    files.append('point_predictions.npz')
    bind_files(run, 'point_complete.json', [run/f for f in files], config)
    files.append('point_complete.json')
    frames = []
    truth = np.asarray(data['y']).ravel()[cells]
    for arm in ARMS:
        frames.append(pd.DataFrame({'model_name': arm, 'cell': cells, 'station': p['names'][cells//t],
            'month': np.asarray(data['months'], str)[cells % t], 'y_true': truth, 'y_pred': prediction[arm],
            'visibility_role': 'outer_query', 'split_seed': partition, 'seed': seed, 'k': 0,
            'river_supported': p['support'], 'graph_delta': prediction[arm]-p['base']}))
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/f for f in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    print(json.dumps({'run': run.name, 'n_query_stations': len(p['receiver_ids']),
        'n_supported_query_stations': len(np.unique(cells[p['support']]//t)),
        'nonzero_readouts_selected': sum(row['selected_alpha'] is not None for row in states),
        'total_readouts': len(states)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_only:
        for partition in (142, 143, 144):
            p = prepare(partition, 42)
            t = p['dataset']['y'].shape[1]
            print(json.dumps({'split': partition, 'stations': len(p['receiver_ids']),
                'supported_stations': len(np.unique(p['cells'][p['support']]//t)),
                'supported_cells': int(p['support'].sum())}), flush=True)
        return
    snapshot = {name: sha256_file(name) for name in (*CODE, str(ROOT/'study_plan.md'))}
    if (args.root/'runtime_snapshot.json').exists():
        if json.loads((args.root/'runtime_snapshot.json').read_text()) != snapshot:
            raise ValueError('nested study source changed; retain this saved version')
    else:
        write_json(args.root/'runtime_snapshot.json', snapshot)
        for name in snapshot:
            target = args.root/'code_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for partition in (142, 143, 144):
        for seed in (42, 43, 44):
            run_one(args.root, partition, seed, digest(snapshot))
    write_json(args.root/'batch_complete.json', {'packages': 9, 'outer_folds_per_package': 3,
        'procedure_evaluations': 108, 'evaluation_role': 'retrospective source-development, not confirmation'})


if __name__ == '__main__':
    main()
