"""Fit actual sampling-age/hydrology message operators on fixed DOC source roles."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from prepare_doc_sampling_river_v1 import prepare_metadata
from run_doc_dynamic_river_v1 import (
    ATLAS,
    BASE,
    REFERENCE,
    consumed,
    nested_banks,
)
from run_doc_dynamic_river_v1 import (
    ROOT as OBSERVED_ROOT,
)
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.frozen_river_correction import matched_messages
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.river_structure_residual import path_candidates
from river_graph.models.sampling_aware_river import (
    SamplingAwareRiverResidual,
    sampling_cell_features,
    shuffle_hydro_timing,
)

ROOT = Path('experiments/phase4_transfer/doc_sampling_river_v1')
STATE_ROOT = Path('experiments/phase4_transfer/doc_hydro_river_expansion_v1')
FIXED = (BASE, 'station_hidden_trees', 'dynamic_lagged', 'expanded_upstream')
ARMS = {'plain_upstream': ('none', 'raw'), 'sampling_age': ('age', 'raw'),
    'sampling_hydro': ('hydro', 'raw'), 'shuffled_sampling_hydro': ('hydro', 'shuffled'),
    'matched_sampling_upstream': ('hydro', 'matched'), 'matched_sampling_nonupstream': ('hydro', 'control')}
ARRAY_KEYS = ('query', 'features', 'values', 'valid', 'timing')


def prepare(partition, seed, metadata_root=ROOT):
    """Reuse exact frozen queries; reconstruct branch priors without refitting."""
    old = OBSERVED_ROOT/'runs'/f'split{partition}_seed{seed}'
    records = {}
    config = consumed(old, ['fitting_inputs.npz', 'source_roles.json', 'matching.json', 'predictions.parquet'], records)
    for path, content in config['sources'].items():
        if sha256_file(path) != content:
            raise ValueError('changed retained query/donor source')
        records[path] = content
    with np.load(old/'fitting_inputs.npz', allow_pickle=False) as saved:
        cache = {key: saved[key].copy() for key in saved.files}
    previous = pd.read_parquet(old/'predictions.parquet')
    state_run = STATE_ROOT/'runs'/old.name
    consumed(state_run, ['predictions.parquet'], records)
    state_panel = pd.read_parquet(state_run/'predictions.parquet')
    fixed = pd.concat([previous[previous.model_name.isin(FIXED)],
        state_panel[state_panel.model_name.eq('expanded_upstream')]], ignore_index=True)
    template = previous[previous.model_name.eq(BASE)].copy()
    data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
    if data['feature_channels'] != ['temperature', 'discharge']:
        raise ValueError('physical monthly discharge channel required')
    names, t = np.asarray(data['site_no'], str), data['y'].shape[1]
    with np.load(config['mask_path'], allow_pickle=False) as saved:
        source_cells, val_cells = saved['train'].copy(), saved['val'].copy()
    ids, receivers = np.unique(source_cells//t), np.unique(val_cells//t)
    folds = station_folds(source_cells, t, seed)
    attention = REFERENCE/'runs'/old.name
    # Resolve the retained source model's declared learning/OOF inputs.
    current_config = json.loads((attention/'config.json').read_text())
    parent, learning = (Path(current_config[f'{key}_run']) for key in ('parent', 'learning'))
    references = {}
    for a, b in combinations(range(5), 2):
        path = learning/f'pair_reference_{a}_{b}'
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
            references[a, b] = {**json.loads(path.with_suffix('.json').read_text()),
                'cells': saved['cells'].copy(), 'pred_z': saved['pred_z'].copy()}
    with np.load(parent/'source_oof.npz', allow_pickle=False) as saved:
        oof = saved['pred_z'].copy()
    truth = np.full_like(np.asarray(data['y']), np.nan)
    truth.ravel()[source_cells] = np.asarray(data['y']).ravel()[source_cells]
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    physical = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    source_banks, roles, matching = nested_banks(truth, names, source_cells, folds, references, edges, physical)
    donor_ids, residual = relative_source_residual_grid(truth, oof, source_cells)
    np.testing.assert_array_equal(donor_ids, ids)
    matched, control, val_matching, _ = matched_messages(edges, physical, names[receivers], names[ids], residual)
    banks = {'source': source_banks, 'validation': {'raw': path_candidates(edges, physical,
        names[receivers], names[ids], residual), 'matched': matched, 'control': control}}
    for name in ('sampling_metadata.npz', 'metadata_definition.json', 'metadata_complete.json'):
        records[str(metadata_root/name)] = sha256_file(metadata_root/name)
    with np.load(metadata_root/'sampling_metadata.npz', allow_pickle=False) as saved:
        metadata = saved['metadata'][ids].copy()
        receiver_phase = saved['receiver_phase'].copy()
        ends = saved['month_ends'].copy()
    visible = np.zeros((len(ids), t), bool)
    visible.ravel()[np.searchsorted(ids, source_cells//t)*t+source_cells % t] = True
    matrices = {}
    for role, cells in (('source', source_cells), ('validation', val_cells)):
        receiving_ids = ids if role == 'source' else receivers
        local_cells = np.searchsorted(receiving_ids, cells//t)*t+cells % t
        group = np.zeros(len(cells), np.int64)
        if role == 'source':
            for fold, stations in enumerate(folds):
                group[np.isin(cells//t, stations)] = fold
        for kind in ('raw', 'matched', 'control'):
            arrays = {key: cache[f'{role}_{kind}_{key}'] for key in ('query', 'features', 'values', 'valid')}
            if kind == 'control':
                # Same real-slot sampling metadata; only innovation/current-hydro
                # identity changes in the matched nonancestor control.
                timing = matrices[f'{role}_matched']['timing'].copy()
            else:
                timing = sampling_cell_features(banks[role][kind], local_cells, metadata,
                    receiver_phase[receiving_ids], ends, visible)
            arrays['timing'] = timing
            matrices[f'{role}_{kind}'] = arrays
            np.testing.assert_array_equal(banks[role][kind]['river_valid'][local_cells//t, local_cells % t], arrays['valid'])
            np.testing.assert_array_equal(banks[role][kind]['river_values'][local_cells//t, local_cells % t].astype(np.float32), arrays['values'])
        matrices[f'{role}_shuffled'] = {k: v.copy() for k, v in matrices[f'{role}_raw'].items()}
        matrices[f'{role}_shuffled']['timing'] = shuffle_hydro_timing(matrices[f'{role}_raw']['timing'],
            matrices[f'{role}_raw']['valid'], cells % t, group, seed)
    return {'config': config, 'matrices': matrices, 'records': records, 'template': template, 'fixed': fixed,
        'roles': roles, 'matching': {'source': matching, 'validation': val_matching},
        **{key: cache[key] for key in ('source_base', 'validation_base', 'source_y', 'validation_y',
            'source_cells', 'val_cells', 'nearest')}}


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('sampling execution changed')
        verify_files(run, 'complete.json', config)
        return
    started = time.monotonic()
    run.mkdir(parents=True, exist_ok=True)
    prepared = prepare(partition, seed, root)
    old = prepared['config']
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'q90_threshold_train', 'split_seed', 'seed')}, 'experiment': ROOT.name,
        'started_at': datetime.now(timezone.utc).isoformat(), 'runtime_snapshot_hash': runtime,
        'models': [*FIXED, *ARMS], 'arms': ARMS, 'lags': [0, 1, 3], 'epochs': 30, 'patience': 5,
        'heads': 2, 'dimensions': 32, 'base_frozen': True, 'sources': prepared['records'],
        'complete_base_training_predictions_are_oof': False,
        'evaluation_role': 'source_validation_development', 'receiving_water_quality': 'none',
        'sampling_reference': 'fixed receiving month-end; source result-row weighted dates',
        'source_phase': 'measured source sample flow and seven-day change; receiving fixed month-end phase',
        'source_monthly_values': 'unchanged environmental OOF innovations; no daily DOC interpolation',
        'matched_metadata': 'copied real-slot dates/flow; donor innovation/current hydro identity changes',
        'shuffle': 'source hydro metadata within prediction-month/lag/query-fold pools; date support fixed',
        'primary_arm': 'sampling_hydro', 'receiving_sampling_metadata': 'not read'}
    if (run/'config.json').exists():
        saved = json.loads((run/'config.json').read_text())
        config['started_at'] = saved['started_at']
        if saved != json.loads(json.dumps(config)):
            raise ValueError('sampling configuration changed')
    write_json(run/'config.json', config)
    write_json(run/'source_roles.json', prepared['roles'])
    write_json(run/'matching.json', prepared['matching'])
    cache = {f'{role}_{key}': value for role, arrays in prepared['matrices'].items() for key, value in arrays.items()}
    cache.update({key: prepared[key] for key in ('source_base', 'validation_base', 'source_y', 'validation_y',
        'source_cells', 'val_cells', 'nearest')})
    np.savez_compressed(run/'fitting_inputs.npz', **cache)
    files = ['config.json', 'source_roles.json', 'matching.json', 'fitting_inputs.npz']
    frames = [prepared['fixed']]

    def progress(arm, row):
        record = {'run': run.name, 'stage': arm, **row, 'elapsed_seconds': time.monotonic()-started}
        write_json(root/'progress.json', record)
        print(json.dumps(record), flush=True)

    for arm, (operator, kind) in ARMS.items():
        source, validation = (prepared['matrices'][f'{role}_{kind}'] for role in ('source', 'validation'))
        if (run/f'{arm}_complete.json').exists():
            verify_files(run, f'{arm}_complete.json', config)
            model = SamplingAwareRiverResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False))
        else:
            model = SamplingAwareRiverResidual(source['query'].shape[-1], source['features'].shape[-1],
                sampling_mode=operator, seed=seed, epochs=30, patience=5)
            model.fit(source, prepared['source_base'], prepared['source_y'], validation,
                prepared['validation_base'], prepared['validation_y'], tail_threshold=config['q90_threshold_train'],
                progress=lambda row, a=arm: progress(a, row))
            torch.save(model.to_payload(), run/f'{arm}.pt')
            write_json(run/f'{arm}.json', model.to_dict())
            bind_files(run, f'{arm}_complete.json', [run/f'{arm}.pt', run/f'{arm}.json'], config)
        predicted, diagnostics = model.predict(validation, prepared['validation_base'], diagnostics=True)
        np.testing.assert_array_equal(predicted[~diagnostics['support']], prepared['validation_base'][~diagnostics['support']])
        frame = prepared['template'].copy()
        frame['model_name'], frame['y_pred'] = arm, predicted
        frame['river_delta_native'] = predicted-prepared['validation_base']
        for key, value in diagnostics.items():
            if key == 'lag_mass':
                for i, lag in enumerate(config['lags']):
                    frame[f'river_lag_mass_{lag}'] = value[:, i]
            else:
                frame[f'river_{key}'] = value
        frame['nearest_path_km'] = np.where(np.isfinite(prepared['nearest']), prepared['nearest'], np.nan)
        frames.append(frame)
        np.savez_compressed(run/f'{arm}_diagnostics.npz', cells=prepared['val_cells'], **diagnostics)
        files += [f'{arm}{suffix}' for suffix in ('.pt', '.json', '_complete.json', '_diagnostics.npz')]
        progress(f'{arm}_saved', {'best_epoch': model.best_epoch_, 'mae': float(abs(predicted-prepared['validation_y']).mean())})
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/name for name in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    progress('complete', {'new_river_fits': len(ARMS), 'new_forest_fits': 0})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--splits', type=int, nargs='+', default=[142, 143, 144])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.root.mkdir(parents=True, exist_ok=True)
    prepare_metadata(args.root)
    (args.root/'training.pid').write_text(str(os.getpid())+'\n')
    snapshot = runtime_code_snapshot()
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_sampling_river_v1.py',
        'scripts/run_doc_dynamic_river_v1.py', 'scripts/run_unified_doc_spatial.py',
        'scripts/run_doc_tail_residual_v1.py', 'scripts/prepare_doc_sampling_river_v1.py',
        'tests/test_sampling_aware_river.py', str(ROOT/'study_plan.md')):
        snapshot[name] = sha256_file(name)
    path = args.root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('preserve executed version; changed training requires a new version')
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/'code_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))


if __name__ == '__main__':
    main()
