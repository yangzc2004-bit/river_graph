"""Joint river structure messages on the retained unmonitored DOC model."""
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
from run_doc_current_availability_attention_v1 import ROOT as PRIOR
from run_doc_current_availability_attention_v1 import model_views, prepare
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.river_structure_residual import (
    RiverStructureResidual,
    nested_path_candidates,
    path_candidates,
)

ROOT = Path('experiments/phase4_transfer/doc_river_structure_residual_v1')
ATLAS = Path('experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis')
ARMS = {'river_none': ('none', False), 'river_simple': ('simple', False),
        'river_structure': ('structure', False), 'river_rewired': ('structure', True)}


def prepare_river(partition, seed):
    prepared = prepare(partition, seed)
    prior = PRIOR/'runs'/f'split{partition}_seed{seed}'
    config = json.loads((prior/'config.json').read_text())
    verify_files(prior, 'complete.json', config)
    prepared['current'] = prior
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    structures = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    data, split = prepared['data'], prepared['split']
    t, ids, val_ids = data['y'].shape[1], prepared['source_ids'], prepared['validation_ids']
    names = np.asarray(data['site_no'], str)
    folds, references = station_folds(split['train'], t, seed), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = prepared['learning']/f'pair_reference_{a}_{b}'
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix('.json').read_text()),
                'cells': saved['cells'].copy(), 'pred_z': saved['pred_z'].copy()}
    donor_ids, residual = relative_source_residual_grid(data['y'], prepared['oof'], split['train'])
    np.testing.assert_array_equal(donor_ids, ids)
    prepared['river'] = {}
    for rewired in (False, True):
        source, roles = nested_path_candidates(data['y'], names, split['train'], folds, references,
            edges, structures, candidates=20, rewired=rewired, seed=seed)
        validation = path_candidates(edges, structures, names[val_ids], names[ids], residual,
                                    candidates=20, rewired=rewired, seed=seed)
        prepared['river'][rewired] = (source, validation, roles)
    return prepared


def views(prepared, rewired):
    source, validation = model_views(prepared, 'real')
    river_source, river_validation, _ = prepared['river'][rewired]
    return {**source, **river_source}, {**validation, **river_validation}


def integrate(prepared, memory, prediction):
    base = np.zeros(prepared['data']['y'].shape)
    base[prepared['validation_ids']] = prepared['validation_base']
    native = base.copy()
    native[prepared['validation_ids']] = prediction
    return memory.predict(base, native).ravel()[prepared['split']['val']]


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('river execution changed after fitting')
        verify_files(run, 'complete.json', config)
        return
    started = time.monotonic()
    prepared = prepare_river(partition, seed)
    old, data, split = prepared['old'], prepared['data'], prepared['split']
    ids, val_ids, val = prepared['source_ids'], prepared['validation_ids'], split['val']
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'q90_threshold_train', 'split_seed', 'seed')}, 'experiment': 'doc_river_structure_residual_v1',
        'started_at': datetime.now(timezone.utc).isoformat(), 'runtime_snapshot_hash': runtime,
        'current_run': str(prepared['current']), 'current_completion_hash': sha256_file(prepared['current']/'complete.json'),
        'training_settings': prepared['settings'], 'arms': ARMS,
        'models': ['available_real_integrated', 'available_real_native', 'station_hidden_trees',
                   *[name for arm in ARMS for name in (arm, f'{arm}_zero_river')]], 'lags': [0, 1, 3],
        'max_observation_age': 12, 'max_candidates': 20, 'max_path_km': 3000,
        'physical_inputs': {str(ATLAS/name): sha256_file(ATLAS/name)
            for name in ('station_structure.csv', 'station_edge_paths.csv')},
        'selection_role': 'source_validation', 'evaluation_role': 'source_validation_only',
        'receiving_water_quality': 'none', 'training': 'joint neural model on environmental station-OOF predictions',
        'donor_reference': 'query and donor folds excluded from tree fitting', 'torch_threads': torch.get_num_threads()}
    if (run/'config.json').exists():
        saved = json.loads((run/'config.json').read_text())
        if digest({k: v for k, v in saved.items() if k != 'started_at'}) != digest(
                {k: v for k, v in config.items() if k != 'started_at'}):
            raise ValueError('saved river version or input files changed')
        config = saved
    else:
        write_json(run/'config.json', config)

    def progress(arm, row):
        record = {'run': run.name, 'stage': arm, **row, 'elapsed_seconds': time.monotonic()-started}
        write_json(root/'progress.json', record)
        print(json.dumps(record), flush=True)

    files = ['config.json', 'river_inputs.npz', 'river_source_roles.json']
    np.savez_compressed(run/'river_inputs.npz', **{f'{mode}_{role}_{key}': values
        for mode, rewired in (('real', False), ('rewired', True))
        for role, arrays in zip(('source', 'validation'), prepared['river'][rewired][:2], strict=True)
        for key, values in arrays.items()})
    write_json(run/'river_source_roles.json', {str(key): item[2] for key, item in prepared['river'].items()})
    previous = pd.read_parquet(prepared['current']/'predictions.parquet')
    template = previous[previous.model_name.eq('available_real_integrated')].copy()
    np.testing.assert_array_equal(template.cell, val)
    frames = [previous[previous.model_name.isin(['available_real_integrated', 'available_real_native', 'station_hidden_trees'])].copy()]
    t = data['y'].shape[1]
    local = np.searchsorted(val_ids, val//t), val % t
    diag_cells = np.ravel_multi_index(local, (len(val_ids), t))
    base = np.zeros(data['y'].shape)
    base[val_ids] = prepared['validation_base']
    retained = prepared['retained']
    for arm, (mode, rewired) in ARMS.items():
        source, validation = views(prepared, rewired)
        stage = f'{arm}_complete.json'
        if (run/stage).exists():
            verify_files(run, stage, config)
            model = RiverStructureResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu'))
        else:
            model = RiverStructureResidual(retained.spatial, retained.temporal, retained.decay,
                                           river_mode=mode, **prepared['settings'])
            model.fit(source, prepared['source_base'], np.asarray(data['y'])[ids], prepared['source_mask'],
                validation, prepared['validation_base'], np.asarray(data['y'])[val_ids], prepared['validation_mask'],
                tail_threshold=config['q90_threshold_train'], selection_role='source_validation',
                progress=lambda row, name=arm: progress(name, row))
            torch.save(model.to_payload(), run/f'{arm}.pt')
            write_json(run/f'{arm}.json', model.to_dict())
            bind_files(run, stage, [run/f'{arm}.pt', run/f'{arm}.json'], config)
        prediction = model.predict(validation, prepared['validation_base'])
        memory = EcologicalResidualTransfer().fit(np.asarray(data['regime']), split['train'],
            np.maximum(0., np.expm1(prepared['oof'].ravel()[split['train']])),
            np.asarray(data['y']).ravel()[split['train']], n_months=t, validation_cells=val,
            validation_y=np.asarray(data['y']).ravel()[val], validation_context=base.ravel()[val],
            validation_temporal=prediction[local], selection_role='source_validation')
        final = integrate(prepared, memory, prediction)
        silent = {**validation, 'river_values': np.zeros_like(validation['river_values']),
                  'river_valid': np.zeros_like(validation['river_valid'])}
        zero_river = integrate(prepared, memory, model.predict(silent, prepared['validation_base']))
        diagnostics = model.river_diagnostics(validation, diag_cells)
        diagnostics['river_delta_effective'] = final-zero_river
        np.savez_compressed(run/f'{arm}_diagnostics.npz', cells=val, **diagnostics)
        write_json(run/f'{arm}_memory.json', memory.to_dict())
        for name, values in ((arm, final), (f'{arm}_zero_river', zero_river)):
            frame = template.copy()
            frame['model_name'], frame['y_pred'] = name, values
            frame['river_support'] = diagnostics['river_support']
            frame['river_prior_mass'] = diagnostics['river_prior_mass']
            frame['river_entropy'] = diagnostics['river_entropy']
            frame['river_delta_effective'] = diagnostics['river_delta_effective'] if name == arm else 0.
            for i, lag in enumerate(config['lags']):
                frame[f'river_lag_mass_{lag}'] = diagnostics['river_lag_mass'][:, i]
            frames.append(frame)
        files += [f'{arm}{suffix}' for suffix in ('.pt', '.json', '_complete.json', '_memory.json', '_diagnostics.npz')]
        progress(f'{arm}_saved', {'validation_mae': float(np.abs(final-template.y_true.to_numpy()).mean()),
            'best_epoch': model.best_epoch_, 'selected_scale': model.selected_scale_})
        del model, source, validation, memory
        gc.collect()
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/name for name in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    progress('complete', {'new_neural_fits': 4, 'new_forest_fits': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--splits', type=int, nargs='+', default=[142, 143, 144])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_only:
        for partition in args.splits:
            for seed in args.seeds:
                prepared = prepare_river(partition, seed)
                _, view = views(prepared, False)
                inputs = prepared['river'][False][1]
                summary = {'split': partition, 'seed': seed, 'source_stations': len(prepared['source_ids']),
                    'validation_stations': len(prepared['validation_ids']),
                    'validation_stations_with_paths': int((inputs['river_owner'] >= 0).any(-1).sum()),
                    'validation_months_with_source': int(inputs['river_valid'].any((2, 3)).sum())}
                retained = prepared['retained']
                model = RiverStructureResidual(retained.spatial, retained.temporal, retained.decay, **prepared['settings'])
                model._prepare_inputs(view)
                print(json.dumps(summary), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'training.pid').write_text(str(os.getpid())+'\n')
    snapshot = runtime_code_snapshot()
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_river_structure_residual_v1.py',
                 'scripts/run_doc_current_availability_attention_v1.py', 'scripts/run_doc_source_level_attention_v1.py',
                 'scripts/run_doc_relative_source_attention_v1.py', 'scripts/run_doc_current_source_attention_v1.py',
                 'scripts/run_doc_tail_residual_v1.py', 'scripts/run_unified_doc_spatial.py',
                 'tests/test_river_structure_residual.py', 'scripts/verify_doc_river_structure_residual_v1.py',
                 str(ROOT/'study_plan.md')):
        snapshot[name] = sha256_file(name)
    path = args.root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('changed execution code requires a new version; preserve completed stages')
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = args.root/'code_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    for partition in args.splits:
        for seed in args.seeds:
            run_one(args.root, partition, seed, digest(snapshot))
    write_json(args.root/'batch_complete.json', {'splits': args.splits, 'seeds': args.seeds,
        'runtime_snapshot_hash': digest(snapshot), 'completed_packages': len(args.splits)*len(args.seeds)})


if __name__ == '__main__':
    main()
