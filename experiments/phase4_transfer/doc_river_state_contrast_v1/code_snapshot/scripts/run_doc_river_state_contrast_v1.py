"""Contrast upstream and receiving environmental states on the frozen DOC model."""
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
from run_doc_dynamic_river_v1 import BASE, consumed
from run_doc_hydro_river_expansion_v1 import ROOT as PREVIOUS_ROOT
from run_doc_hydro_river_expansion_v1 import prepare as prepare_absolute
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.dynamic_river_state import DynamicRiverState
from river_graph.models.river_state_contrast import receiver_contrast

ROOT = Path('experiments/phase4_transfer/doc_river_state_contrast_v1')
OBSERVED = 'dynamic_lagged'
FIXED = (BASE, 'station_hidden_trees', OBSERVED, 'observed_state',
         'expanded_upstream', 'expanded_uniform',
         'expanded_matched_upstream', 'expanded_matched_nonupstream')
ARMS = {'contrast_upstream': ('raw', 'observed', 'dynamic'),
        'contrast_uniform': ('raw', 'observed', 'uniform'),
        'contrast_matched_upstream': ('matched', 'observed', 'dynamic'),
        'contrast_matched_nonupstream': ('control', 'observed', 'dynamic')}


def prepare(partition, seed, progress=None):
    """Rebuild absolute banks, then change ONLY their message values."""
    prepared = prepare_absolute(partition, seed, progress)
    previous_run = PREVIOUS_ROOT/'runs'/f'split{partition}_seed{seed}'
    consumed(previous_run, ['fitting_inputs.npz', 'predictions.parquet',
        'source_roles.json', 'matching.json'], prepared['records'])
    with np.load(previous_run/'fitting_inputs.npz', allow_pickle=False) as saved:
        for role, arrays in prepared['matrices'].items():
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, saved[f'{role}_{key}'])
        for key in ('source_cells', 'val_cells', 'states', 'availability',
                    'source_observed_base', 'validation_observed_base'):
            np.testing.assert_array_equal(prepared[key], saved[key])
        prepared['previous_support'] = saved['validation_raw_valid'].any((1, 2))
    prepared['fixed'] = pd.read_parquet(previous_run/'predictions.parquet').query('model_name in @FIXED').copy()
    for role in ('source', 'validation'):
        cells = prepared['source_cells'] if role == 'source' else prepared['val_cells']
        for bank in ('raw', 'matched', 'control'):
            name = f'{role}_{bank}'
            prepared['matrices'][name] = receiver_contrast(prepared['matrices'][name],
                cells, prepared['states'], copy=False)
    return prepared


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
        'observed_branch_frozen': True, 'combined_architecture': 'observed correction then independent environmental contrast correction',
        'evaluation_role': 'source_validation_development', 'selection_role': 'source_validation',
        'receiving_water_quality': 'none', 'sources': prepared['records'],
        'matching': 'drainage area and label-free hydro availability; common edge-lag support',
        'historical_hidden_preprocessing': 'retained source-cell hydro and all-cohort static/ecology stats; source states replay exactly',
        'hydro_pool': 'all covariate-known ST357 nodes minus complete receiving fold/role',
        'water_quality_library_expanded': False,
        'known_atlas_covariates': True, 'external_inductive_evaluation': False,
        'previous_state_anchor': 'expanded_upstream retained as fixed comparator, not stacked',
        'message_value': 'upstream environmental state at allowed lag minus receiver environmental state at current query month',
        'receiver_state_DOC_inputs': 'none; same frozen chemistry-free environmental encoding',
        'capacity_and_candidate_support_unchanged': True,
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
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_river_state_contrast_v1.py', 'scripts/run_doc_hydro_river_expansion_v1.py',
        'scripts/audit_doc_hydro_river_expansion_v1.py',
        'scripts/run_doc_daily_hydro_residual_v1.py', 'scripts/run_doc_dynamic_river_state_v1.py',
        'scripts/run_doc_dynamic_river_v1.py', 'scripts/run_unified_doc_spatial.py',
        'scripts/run_doc_tail_residual_v1.py', 'tests/test_hydro_river_expansion.py', 'tests/test_river_state_contrast.py', str(ROOT/'study_plan.md')):
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
