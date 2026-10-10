"""Expand hydro-only real upstream nodes on the existing frozen DOC model."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_doc_hydro_river_expansion_v1 import DAILY, path_support
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_dynamic_river_state_v1 import ROOT as PREVIOUS_ROOT
from run_doc_dynamic_river_v1 import ATLAS, BASE, REFERENCE, consumed
from run_doc_dynamic_river_v1 import ROOT as OBSERVED_ROOT
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.dynamic_river_residual import DynamicRiverResidual
from river_graph.models.dynamic_river_state import (
    DynamicRiverState,
    environmental_states,
    state_cell_features,
    state_paths,
)
from river_graph.models.hydro_river_expansion import (
    FrozenHydroPreprocessing,
    eligible_hydro_nodes,
)
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)

ROOT = Path('experiments/phase4_transfer/doc_hydro_river_expansion_v1')
OBSERVED = 'dynamic_lagged'
FIXED = (BASE, 'station_hidden_trees', OBSERVED, 'observed_state',
         'matched_state_upstream', 'matched_state_nonupstream')
ARMS = {'expanded_upstream': ('raw', 'observed', 'dynamic'),
        'expanded_uniform': ('raw', 'observed', 'uniform'),
        'expanded_matched_upstream': ('matched', 'observed', 'dynamic'),
        'expanded_matched_nonupstream': ('control', 'observed', 'dynamic')}


def state_matrices(names, ids, validation_ids, source_cells, val_cells, folds,
                   edges, physical, states, availability, hydro):
    """All atlas hydro nodes except the whole receiving fold/role; no chemistry."""
    t = states.shape[1]
    matrices, roles, matching = {}, [], {'source': [], 'validation': []}

    def build(role, receivers, rows, positions):
        donor = eligible_hydro_nodes(len(names), receivers)
        real, control, records = state_paths(edges, physical, names[receivers], names[donor], availability[donor])
        for bank in (real, control):
            owner = bank['river_owner']
            bank['river_owner'] = np.where(owner >= 0, donor[owner.clip(min=0)], -1)
        arrays = [state_cell_features(bank, rows, hydro[receivers], hydro, states, availability)
                  for bank in (real, control)]
        common = arrays[0]['valid'] & arrays[1]['valid']
        matched = [{key: (common if key == 'valid' else np.where(common[..., None], value, 0.))
                    for key, value in array.items()} for array in arrays]
        for kind, array in zip(('raw', 'matched', 'control'), (arrays[0], *matched), strict=True):
            for key, value in array.items():
                target = matrices.setdefault(f'{role}_{kind}', {})
                if key not in target:
                    count = len(source_cells) if role == 'source' else len(val_cells)
                    target[key] = np.zeros((count, *value.shape[1:]), dtype=value.dtype)
                target[key][positions] = value
        matching[role].extend(records)
        roles.append({'role': role, 'query_station_ids': receivers.tolist(),
            'library_station_ids': donor.tolist(), 'query_labels_in_bank': False,
            'library_kind': 'hydro-only known-atlas covariates; separate from DOC experience library'})

    for query in folds:
        query = np.sort(query)
        selected = np.flatnonzero(np.isin(source_cells//t, query))
        local = np.searchsorted(query, source_cells[selected]//t)*t+source_cells[selected] % t
        build('source', query, local, selected)
    local = np.searchsorted(validation_ids, val_cells//t)*t+val_cells % t
    build('validation', validation_ids, local, np.arange(len(val_cells)))
    return matrices, roles, matching


def prepare(partition, seed, progress=None):
    old_run = OBSERVED_ROOT/'runs'/f'split{partition}_seed{seed}'
    records = {}
    old = consumed(old_run, ['fitting_inputs.npz', 'dynamic_lagged.pt', 'predictions.parquet'], records)
    for name, expected in old['sources'].items():
        if sha256_file(name) != expected:
            raise ValueError('retained execution input changed')
        records[name] = expected
    with np.load(old_run/'fitting_inputs.npz', allow_pickle=False) as saved:
        cache = {name: saved[name].copy() for name in saved.files}
    current = REFERENCE/'runs'/old_run.name
    config = json.loads((current/'config.json').read_text())
    data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
    with np.load(config['mask_path'], allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ('train', 'val', 'test', 'context')}
    t, names = data['y'].shape[1], np.asarray(data['site_no'], str)
    ids, val_ids = np.unique(split['train']//t), np.unique(split['val']//t)
    np.testing.assert_array_equal(cache['source_cells'], split['train'])
    np.testing.assert_array_equal(cache['val_cells'], split['val'])
    if np.intersect1d(ids, np.unique(np.r_[split['val'], split['test']]//t)).size:
        raise ValueError('new receiving stations cannot supply source states')
    daily, _, daily_records = load_daily_pack(DAILY, old['dataset_hash'], data['x'].shape[:2])
    records.update({daily_records['daily_features_path']: daily_records['daily_features_hash'],
                    daily_records['daily_metadata_path']: daily_records['daily_metadata_hash']})
    preprocessing = FrozenHydroPreprocessing.fit(data, split['train'])
    inputs = preprocessing.transform(data, daily)
    model = AvailableSourceAttentionResidual.from_payload(torch.load(current/'available_real.pt',
        weights_only=False, map_location='cpu'))
    for name in ('spatial', 'temporal', 'decay', 'head'):
        getattr(model, name).requires_grad_(False)
    states, availability = environmental_states(model, inputs, progress=progress)
    previous_run = PREVIOUS_ROOT/'runs'/old_run.name
    consumed(previous_run, ['fitting_inputs.npz', 'predictions.parquet'], records)
    with np.load(previous_run/'fitting_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(states[ids], saved['states'])
        np.testing.assert_array_equal(availability[ids], saved['availability'])
        previous_support = saved['validation_raw_valid'].any((1, 2))
    physical = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    matrices, roles, matching = state_matrices(names, ids, val_ids, split['train'], split['val'],
        station_folds(split['train'], t, seed), edges, physical, states, availability, daily)
    _, nearest, _ = path_support(names, val_ids, eligible_hydro_nodes(len(names), val_ids),
        availability, edges, physical, split['val'])
    # Queries and complete base stay identical to the preceding river experiment.
    for role in ('source', 'validation'):
        for bank in ('raw', 'matched', 'control'):
            matrices[f'{role}_{bank}']['query'] = cache[f'{role}_raw_query']
    observed = DynamicRiverResidual.from_payload(torch.load(old_run/'dynamic_lagged.pt', weights_only=False))
    observed_source = observed.predict({key: cache[f'source_raw_{key}']
        for key in ('query', 'features', 'values', 'valid')}, cache['source_base'])
    previous = pd.read_parquet(old_run/'predictions.parquet')
    template = previous[previous.model_name.eq(BASE)].copy()
    obs_frame = previous[previous.model_name.eq(OBSERVED)].copy()
    np.testing.assert_array_equal(obs_frame.cell, cache['val_cells'])
    # Validate saved observational anchor independently, then retain stored bits.
    replay = observed.predict({key: cache[f'validation_raw_{key}']
        for key in ('query', 'features', 'values', 'valid')}, cache['validation_base'])
    np.testing.assert_array_equal(replay, obs_frame.y_pred)
    return {'old': old, 'matrices': matrices, 'source_complete_base': cache['source_base'],
        'source_observed_base': observed_source, 'validation_complete_base': cache['validation_base'],
        'validation_observed_base': obs_frame.y_pred.to_numpy(), 'source_y': cache['source_y'],
        'validation_y': cache['validation_y'], 'source_cells': cache['source_cells'], 'val_cells': cache['val_cells'],
        'observed_support': cache['validation_raw_valid'].any((1, 2)), 'nearest': nearest,
        'template': template, 'fixed': pd.read_parquet(previous_run/'predictions.parquet').query('model_name in @FIXED').copy(),
        'records': records, 'roles': roles, 'matching': matching, 'states': states, 'availability': availability,
        'preprocessing': preprocessing.statistics, 'previous_support': previous_support}


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('state-message execution changed')
        verify_files(run, 'complete.json', config)
        return
    start = time.monotonic()
    run.mkdir(parents=True, exist_ok=True)

    def progress(stage, row):
        record = {'run': run.name, 'stage': stage, **row, 'elapsed_seconds': time.monotonic()-start}
        write_json(root/'progress.json', record)
        print(json.dumps(record), flush=True)

    prepared = prepare(partition, seed, lambda row: progress('environmental_state_encoding', row))
    old = prepared['old']
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'q90_threshold_train', 'split_seed', 'seed')}, 'experiment': ROOT.name,
        'started_at': datetime.now(timezone.utc).isoformat(), 'runtime_snapshot_hash': runtime,
        'models': [*FIXED, *ARMS], 'arms': ARMS, 'lags': [0, 1, 3], 'epochs': 30, 'patience': 5,
        'heads': 2, 'dimensions': 32, 'learning_rate': .001, 'tail_weight': 2,
        'state': 'frozen environmental-only GRU64 + ecology9 + daily8 + daily_change8 + monthly_hydro4',
        'state_available': 'measured temperature/flow or daily descriptor within preceding 12 months',
        'upstream_state_DOC_inputs': 'all values, visibility and support removed; never inferred DOC truth',
        'base_frozen': True, 'complete_base_training_predictions_are_oof': False,
        'observed_branch_frozen': True, 'combined_architecture': 'observed correction then independent state correction',
        'evaluation_role': 'source_validation_development', 'selection_role': 'source_validation',
        'receiving_water_quality': 'none', 'sources': prepared['records'],
        'matching': 'drainage area and label-free hydro availability; common edge-lag support',
        'historical_hidden_preprocessing': 'retained source-cell hydro and all-cohort static/ecology stats; source states replay exactly',
        'hydro_pool': 'all covariate-known ST357 nodes minus complete receiving fold/role',
        'water_quality_library_expanded': False,
        'known_atlas_covariates': True, 'external_inductive_evaluation': False,
        'previous_state_anchor': 'observed_state retained as fixed comparator, not stacked',
        'preprocessing': prepared['preprocessing']}
    if (run/'config.json').exists():
        config['started_at'] = json.loads((run/'config.json').read_text())['started_at']
        if json.loads((run/'config.json').read_text()) != json.loads(json.dumps(config)):
            raise ValueError('preserve saved state configuration')
    write_json(run/'config.json', config)
    write_json(run/'source_roles.json', prepared['roles'])
    write_json(run/'matching.json', prepared['matching'])
    cache = {f'{role}_{key}': value for role, arrays in prepared['matrices'].items() for key, value in arrays.items()}
    cache.update({key: prepared[key] for key in ('source_complete_base', 'source_observed_base',
        'validation_complete_base', 'validation_observed_base', 'source_y', 'validation_y', 'source_cells',
        'val_cells', 'observed_support', 'nearest', 'states', 'availability', 'previous_support')})
    np.savez_compressed(run/'fitting_inputs.npz', **cache)
    files = ['config.json', 'source_roles.json', 'matching.json', 'fitting_inputs.npz']
    frames = [prepared['fixed']]
    for arm, (bank, anchor, allocation) in ARMS.items():
        source, validation = (prepared['matrices'][f'{role}_{bank}'] for role in ('source', 'validation'))
        sb, vb = (prepared[f'{role}_{anchor}_base'] for role in ('source', 'validation'))
        if (run/f'{arm}_complete.json').exists():
            verify_files(run, f'{arm}_complete.json', config)
            fitted = DynamicRiverState.from_payload(torch.load(run/f'{arm}.pt', weights_only=False))
        else:
            fitted = DynamicRiverState(source['query'].shape[-1], source['features'].shape[-1],
                source['states'].shape[-1], allocation=allocation, seed=seed, epochs=30, patience=5)
            fitted.fit(source, sb, prepared['source_y'], validation, vb, prepared['validation_y'],
                tail_threshold=config['q90_threshold_train'], progress=lambda row, a=arm: progress(a, row))
            torch.save(fitted.to_payload(), run/f'{arm}.pt')
            write_json(run/f'{arm}.json', fitted.to_dict())
            bind_files(run, f'{arm}_complete.json', [run/f'{arm}.pt', run/f'{arm}.json'], config)
        prediction, diag = fitted.predict(validation, vb, diagnostics=True)
        np.testing.assert_array_equal(prediction[~diag['support']], vb[~diag['support']])
        frame = prepared['template'].copy()
        frame['model_name'], frame['y_pred'] = arm, prediction
        frame['previous_state_support'] = prepared['previous_support']
        frame['state_support'], frame['observed_support'] = diag['support'], prepared['observed_support']
        frame['state_delta_log'], frame['state_delta_native'] = diag['delta_log'], prediction-vb
        frame['state_prior_mass'], frame['state_entropy'] = diag['prior_mass'], diag['entropy']
        frame['nearest_path_km'] = np.where(np.isfinite(prepared['nearest']), prepared['nearest'], np.nan)
        for i, lag in enumerate(config['lags']):
            frame[f'state_lag_mass_{lag}'] = diag['lag_mass'][:, i]
        frames.append(frame)
        np.savez_compressed(run/f'{arm}_diagnostics.npz', cells=prepared['val_cells'], **diag)
        files += [f'{arm}{suffix}' for suffix in ('.pt', '.json', '_complete.json', '_diagnostics.npz')]
        progress(arm+'_saved', {'best_epoch': fitted.best_epoch_, 'validation_mae': float(abs(prediction-prepared['validation_y']).mean())})
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/name for name in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    progress('complete', {'new_neural_fits': len(ARMS), 'new_forest_fits': 0})
    del prepared, cache
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
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_hydro_river_expansion_v1.py', 'scripts/audit_doc_hydro_river_expansion_v1.py',
        'scripts/run_doc_daily_hydro_residual_v1.py', 'scripts/run_doc_dynamic_river_state_v1.py',
        'scripts/run_doc_dynamic_river_v1.py', 'scripts/run_unified_doc_spatial.py',
        'scripts/run_doc_tail_residual_v1.py', 'tests/test_hydro_river_expansion.py', str(ROOT/'study_plan.md')):
        snapshot[name] = sha256_file(name)
    path = args.root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('changed execution needs a new version')
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
