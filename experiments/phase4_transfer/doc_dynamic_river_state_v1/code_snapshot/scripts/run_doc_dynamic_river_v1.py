"""Train protected dynamic river messages on existing DOC source-role models."""
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
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.dynamic_river_residual import (
    DynamicRiverResidual,
    river_cell_features,
)
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.frozen_river_correction import matched_messages
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.river_structure_residual import path_candidates
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)
from river_graph.models.source_level_attention import source_level_input_view

ROOT = Path('experiments/phase4_transfer/doc_dynamic_river_v1')
REFERENCE = Path('experiments/phase4_transfer/doc_current_availability_attention_v1')
ATLAS = Path('experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis')
ARMS = {'static_upstream': ('static_same_month', 'raw'),
        'dynamic_same_month': ('dynamic_same_month', 'raw'),
        'dynamic_lagged': ('dynamic_lagged', 'raw'),
        'matched_upstream': ('dynamic_lagged', 'matched'),
        'matched_nonupstream': ('dynamic_lagged', 'control')}
BASE = 'available_real_integrated'
FIXED = (BASE, 'station_hidden_trees')


def consumed(run, filenames, records):
    """Verify only consumed retained assets; archived forests are not required.

    The original completion is retained verbatim. This is a scoped reader,
    not a claim that every archived asset has been restored or reverified.
    """
    config = json.loads((run/'config.json').read_text())
    completion = json.loads((run/'complete.json').read_text())
    if completion['config_hash'] != digest(config):
        raise ValueError('retained source config identity changed')
    for name in ('config.json', 'complete.json', *filenames):
        path = run/name
        actual = sha256_file(path)
        if name != 'complete.json' and completion['files'].get(name) != actual:
            raise ValueError(f'changed or unbound consumed source: {path}')
        records[str(path)] = actual
    return config


def nested_banks(truth, names, cells, folds, references, edges, physical):
    """Same query/donor-fold isolation for raw and availability-matched banks."""
    t, ids = truth.shape[1], np.unique(cells//truth.shape[1])
    output, roles, matching = {}, [], []
    for a, query in enumerate(folds):
        donor_cells = cells[~np.isin(cells//t, query)]
        reference = np.full(truth.shape, np.nan)
        for b, donor in enumerate(folds):
            if a == b:
                continue
            saved = references[tuple(sorted((a, b)))]
            hidden = set(query.tolist()+donor.tolist())
            if (set(saved['hidden_stations']) != hidden
                    or set(saved['fitted_stations']) != set(ids.tolist())-hidden):
                raise ValueError('river references must exclude query AND donor folds')
            expected = cells[np.isin(cells//t, list(hidden))]
            if not np.array_equal(saved['cells'], expected) or np.shape(saved['pred_z']) != expected.shape:
                raise ValueError('saved double-held reference population changed')
            take = np.isin(expected//t, donor)
            reference.ravel()[expected[take]] = np.asarray(saved['pred_z'])[take]
        donor_ids, residual = relative_source_residual_grid(truth, reference, donor_cells)
        matched, control, records, _ = matched_messages(edges, physical, names[query], names[donor_ids], residual)
        raw = path_candidates(edges, physical, names[query], names[donor_ids], residual)
        for kind, arrays in (('raw', raw), ('matched', matched), ('control', control)):
            # Owners become indices in the common source hydro grid.
            array = arrays['river_owner']
            arrays['river_owner'] = np.where(array >= 0, np.searchsorted(ids, donor_ids[array.clip(min=0)]), -1)
            if kind not in output:
                output[kind] = {key: np.zeros((len(ids), *value.shape[1:]), dtype=value.dtype)
                                for key, value in arrays.items()}
            for key, value in arrays.items():
                output[kind][key][np.searchsorted(ids, query)] = value
        roles.append({'query_fold': a, 'query_station_ids': query.tolist(),
            'library_station_ids': donor_ids.tolist(), 'query_labels_in_bank': False,
            'reference_exclusion': 'query AND donor folds', 'reference_is_environmental_oof': True})
        matching.extend({**record, 'query_fold': a} for record in records)
    return output, roles, matching


def queried_state(model, inputs, cells, base, memory, global_ids):
    """Frozen states and complete source predictions; no new base training."""
    arrays = model._prepare_inputs(inputs)
    cells = np.asarray(cells)
    hidden, delta = [], []
    with torch.no_grad():
        for start in range(0, len(cells), 512):
            selected = cells[start:start+512]
            hidden.append(model._hidden_cells(arrays, selected).numpy())
            delta.append(model._delta_cells(arrays, selected).numpy())
    hidden = np.concatenate(hidden)
    base_grid = np.zeros((memory.n_nodes_, memory.n_months_))
    native_grid = base_grid.copy()
    base_grid[global_ids] = np.nan_to_num(base)
    native_grid[global_ids] = np.nan_to_num(base)
    native = np.maximum(0., np.asarray(base).ravel()[cells]+model.selected_scale_*np.concatenate(delta))
    global_cells = global_ids[cells//memory.n_months_]*memory.n_months_+cells % memory.n_months_
    native_grid.ravel()[global_cells] = native
    integrated = memory.predict(base_grid, native_grid).ravel()[global_cells]
    hydro = np.asarray(inputs['extra'])[..., 30:38]
    month = cells % hydro.shape[1]
    row = cells//hydro.shape[1]
    current = hydro[row, month]
    previous = hydro[row, (month-1).clip(min=0)]
    previous = np.where((month > 0)[:, None], previous, 0.)
    query = np.concatenate([hidden, inputs['env'][row], current, current-previous,
                           np.log1p(integrated[:, None])/5], -1).astype(np.float32)
    return query, integrated


def prepare(partition, seed):
    prior = REFERENCE/'runs'/f'split{partition}_seed{seed}'
    records = {}
    config = consumed(prior, ['available_real.pt', 'available_real_memory.json',
        'attention_candidates.npz', 'predictions.parquet'], records)
    for kind in ('dataset', 'mask'):
        path = config[f'{kind}_path']
        actual = sha256_file(path)
        if actual != config[f'{kind}_hash']:
            raise ValueError('source data or station roles changed')
        records[path] = actual
    data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
    with np.load(config['mask_path'], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ('train', 'val', 'test', 'context')}
    t, names = data['y'].shape[1], np.asarray(data['site_no'], str)
    ids, val_ids = np.unique(split['train']//t), np.unique(split['val']//t)
    if np.intersect1d(ids, np.unique(np.r_[split['val'], split['test']]//t)).size:
        raise ValueError('receiver and source stations overlap')
    parent, learning, source = (Path(config[f'{key}_run']) for key in ('parent', 'learning', 'source'))
    consumed(parent, ['source_oof.npz', 'validation_inputs.npz'], records)
    consumed(source, ['joint_source_inputs.npz'], records)
    files = ['innovation_inputs.npz', *[f'pair_reference_{a}_{b}{suffix}'
        for a, b in combinations(range(5), 2) for suffix in ('.npz', '.json')]]
    consumed(learning, files, records)
    with np.load(source/'joint_source_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_station_ids'], ids)
        train_inputs = {key: saved[key].copy() for key in ('raw', 'env', 'age', 'support', 'extra')}
    with np.load(parent/'validation_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['cell'], split['val'])
        val_inputs = {key: saved[key].copy() for key in ('raw', 'env', 'age', 'support', 'extra')}
        validation_base = saved['context'].copy()
    with np.load(parent/'source_oof.npz', allow_pickle=False) as saved:
        oof = saved['pred_z'].copy()
    train_base = np.maximum(0., np.expm1(oof[ids]))
    with np.load(learning/'innovation_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_ids'], ids)
        np.testing.assert_array_equal(saved['validation_ids'], val_ids)
        train_inputs['extra'] = np.concatenate([train_inputs['extra'], saved['source_real']], -1)
        val_inputs['extra'] = np.concatenate([val_inputs['extra'], saved['validation_real']], -1)
    train_inputs['attention_reference'] = np.nan_to_num(train_base)
    val_inputs['attention_reference'] = validation_base
    with np.load(prior/'attention_candidates.npz', allow_pickle=False) as saved:
        def view(role, inputs):
            arrays = {name[len(role)+1:]: saved[name].copy() for name in saved.files if name.startswith(role+'_')}
            return source_level_input_view(inputs, arrays, 'real')
        train_view, val_view = view('source', train_inputs), view('validation', val_inputs)
    model = AvailableSourceAttentionResidual.from_payload(torch.load(prior/'available_real.pt',
            weights_only=False, map_location='cpu'))
    for name in ('spatial', 'temporal', 'decay', 'head'):
        getattr(model, name).requires_grad_(False)
    memory = EcologicalResidualTransfer.from_dict(json.loads((prior/'available_real_memory.json').read_text()))
    source_cells = np.searchsorted(ids, split['train']//t)*t+split['train'] % t
    val_cells = np.searchsorted(val_ids, split['val']//t)*t+split['val'] % t
    train_query, full_source = queried_state(model, train_view, source_cells, train_base, memory, ids)
    val_query, replay = queried_state(model, val_view, val_cells, validation_base, memory, val_ids)
    previous = pd.read_parquet(prior/'predictions.parquet')
    template = previous[previous.model_name.eq(BASE)].copy()
    np.testing.assert_array_equal(template.cell, split['val'])
    max_replay_error = float(np.max(np.abs(replay-template.y_pred)))
    np.testing.assert_allclose(replay, template.y_pred, rtol=2e-5, atol=2e-5)
    # Frozen reference uses original stored bits, independent of batch arithmetic.
    val_base = template.y_pred.to_numpy().copy()
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    physical = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    for name in ('station_edge_paths.csv', 'station_structure.csv'):
        records[str(ATLAS/name)] = sha256_file(ATLAS/name)
    folds = station_folds(split['train'], t, seed)
    references = {}
    for a, b in combinations(range(5), 2):
        prefix = learning/f'pair_reference_{a}_{b}'
        with np.load(prefix.with_suffix('.npz'), allow_pickle=False) as saved:
            references[(a, b)] = {**json.loads(prefix.with_suffix('.json').read_text()),
                'cells': saved['cells'].copy(), 'pred_z': saved['pred_z'].copy()}
    # Strip all nontraining truth before constructing either donor bank.
    source_truth = np.full_like(np.asarray(data['y']), np.nan)
    source_truth.ravel()[split['train']] = np.asarray(data['y']).ravel()[split['train']]
    train_banks, roles, train_matching = nested_banks(source_truth, names, split['train'],
            folds, references, edges, physical)
    donor_ids, residual = relative_source_residual_grid(source_truth, oof, split['train'])
    np.testing.assert_array_equal(donor_ids, ids)
    matched, control, val_matching, _ = matched_messages(edges, physical, names[val_ids], names[ids], residual)
    val_banks = {'raw': path_candidates(edges, physical, names[val_ids], names[ids], residual),
                 'matched': matched, 'control': control}
    hydro = train_inputs['extra'][..., 30:38]
    matrices = {}
    for role, banks, cells, inputs, query in (('source', train_banks, source_cells, train_inputs, train_query),
            ('validation', val_banks, val_cells, val_inputs, val_query)):
        for kind, bank in banks.items():
            matrices[f'{role}_{kind}'] = {**river_cell_features(bank, cells, inputs['extra'][..., 30:38], hydro),
                                         'query': query}
            del matrices[f'{role}_{kind}']['query_hydro']
        for key in ('valid',):
            np.testing.assert_array_equal(matrices[f'{role}_matched'][key], matrices[f'{role}_control'][key])
    lengths = np.expm1(val_banks['raw']['river_path'][..., 0]*np.log1p(3000))
    nearest = np.min(np.where(val_banks['raw']['river_owner'] >= 0, lengths, np.inf), axis=1)[val_cells//t]
    return {'config': config, 'matrices': matrices, 'source_base': full_source, 'validation_base': val_base,
        'source_y': np.asarray(data['y']).ravel()[split['train']], 'validation_y': template.y_true.to_numpy(),
        'template': template, 'fixed': previous[previous.model_name.isin(FIXED)].copy(), 'records': records,
        'roles': roles, 'matching': {'source': train_matching, 'validation': val_matching},
        'source_cells': split['train'], 'val_cells': split['val'], 'nearest': nearest,
        'replay_max_abs_mgL': max_replay_error}


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('dynamic river execution changed')
        verify_files(run, 'complete.json', config)
        return
    started = time.monotonic()
    run.mkdir(parents=True, exist_ok=True)
    prepared = prepare(partition, seed)
    old = prepared['config']
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
            'q90_threshold_train', 'split_seed', 'seed')}, 'experiment': ROOT.name,
        'started_at': datetime.now(timezone.utc).isoformat(), 'runtime_snapshot_hash': runtime,
        'models': [*FIXED, *ARMS], 'arms': ARMS, 'lags': [0, 1, 3], 'epochs': 30, 'patience': 5,
        'heads': 2, 'dimensions': 32, 'learning_rate': 1e-3, 'tail_weight': 2,
        'base_frozen': True, 'complete_base_training_predictions_are_oof': False,
        'donor_departures': 'environmental double-held query/donor OOF; source-only truth',
        'evaluation_role': 'source_validation_development', 'selection_role': 'source_validation',
        'receiving_water_quality': 'none', 'sources': prepared['records'],
        'fixed_base_replay_max_abs_mgL': prepared['replay_max_abs_mgL'],
        'archived_forests_needed': False, 'source_verification': 'consumed retained files only'}
    if (run/'config.json').exists():
        saved = json.loads((run/'config.json').read_text())
        config['started_at'] = saved['started_at']
        if saved != json.loads(json.dumps(config)):
            raise ValueError('changed dynamic river configuration')
    write_json(run/'config.json', config)
    write_json(run/'source_roles.json', prepared['roles'])
    write_json(run/'matching.json', prepared['matching'])
    cache = {f'{role}_{key}': value for role, arrays in prepared['matrices'].items() for key, value in arrays.items()}
    cache.update({key: prepared[key] for key in ('source_base', 'validation_base', 'source_y', 'validation_y',
                                                'source_cells', 'val_cells', 'nearest')})
    np.savez_compressed(run/'fitting_inputs.npz', **cache)
    files = ['config.json', 'source_roles.json', 'matching.json', 'fitting_inputs.npz']
    frames = [prepared['fixed']]

    def progress(arm, record):
        row = {'run': run.name, 'stage': arm, **record, 'elapsed_seconds': time.monotonic()-started}
        write_json(root/'progress.json', row)
        print(json.dumps(row), flush=True)

    for arm, (mode, bank) in ARMS.items():
        source, validation = (prepared['matrices'][f'{role}_{bank}'] for role in ('source', 'validation'))
        if (run/f'{arm}_complete.json').exists():
            verify_files(run, f'{arm}_complete.json', config)
            model = DynamicRiverResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False))
        else:
            model = DynamicRiverResidual(source['query'].shape[-1], source['features'].shape[-1],
                mode=mode, seed=seed, epochs=30, patience=5)
            model.fit(source, prepared['source_base'], prepared['source_y'],
                validation, prepared['validation_base'], prepared['validation_y'],
                tail_threshold=config['q90_threshold_train'], progress=lambda row, a=arm: progress(a, row))
            torch.save(model.to_payload(), run/f'{arm}.pt')
            write_json(run/f'{arm}.json', model.to_dict())
            bind_files(run, f'{arm}_complete.json', [run/f'{arm}.pt', run/f'{arm}.json'], config)
        predicted, diagnostics = model.predict(validation, prepared['validation_base'], diagnostics=True)
        np.testing.assert_array_equal(predicted[~diagnostics['support']], prepared['validation_base'][~diagnostics['support']])
        frame = prepared['template'].copy()
        frame['model_name'], frame['y_pred'] = arm, predicted
        frame['river_support'] = diagnostics['support']
        frame['river_delta_log'] = diagnostics['delta_log']
        frame['river_delta_native'] = predicted-prepared['validation_base']
        frame['river_prior_mass'] = diagnostics['prior_mass']
        frame['river_entropy'] = diagnostics['entropy']
        frame['nearest_path_km'] = np.where(np.isfinite(prepared['nearest']), prepared['nearest'], np.nan)
        for index, lag in enumerate(config['lags']):
            frame[f'river_lag_mass_{lag}'] = diagnostics['lag_mass'][:, index]
        frames.append(frame)
        np.savez_compressed(run/f'{arm}_diagnostics.npz', cells=prepared['val_cells'], **diagnostics)
        files += [f'{arm}{suffix}' for suffix in ('.pt', '.json', '_complete.json', '_diagnostics.npz')]
        progress(f'{arm}_saved', {'best_epoch': model.best_epoch_, 'validation_mae': float(np.abs(predicted-prepared['validation_y']).mean())})
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/name for name in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    progress('complete', {'new_neural_fits': len(ARMS), 'new_forest_fits': 0})
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--splits', type=int, nargs='+', default=[142, 143, 144])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'training.pid').write_text(str(os.getpid())+'\n')
    snapshot = runtime_code_snapshot()
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_dynamic_river_v1.py',
                 'scripts/run_unified_doc_spatial.py', 'scripts/run_doc_tail_residual_v1.py',
                 'tests/test_dynamic_river_residual.py', str(ROOT/'study_plan.md')):
        snapshot[name] = sha256_file(name)
    path = args.root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('preserve executed code; changed training requires a new version')
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
