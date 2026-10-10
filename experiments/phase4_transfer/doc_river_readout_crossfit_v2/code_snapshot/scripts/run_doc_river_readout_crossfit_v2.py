"""Test river training anchors with conditional station-held scalar readouts."""
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
from run_doc_dynamic_river_v1 import BASE, REFERENCE, consumed
from run_doc_hydro_river_expansion_v1 import ROOT as PREVIOUS_ROOT
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.models.conditional_readout_crossfit_v2 import (
    combine_readout,
    conditional_crossfit,
    fit_readout,
    readout_design,
)
from river_graph.models.dynamic_river_state import DynamicRiverState
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)
from river_graph.models.source_level_attention import source_level_input_view

ROOT = Path('experiments/phase4_transfer/doc_river_readout_crossfit_v2')
OBSERVED = 'dynamic_lagged'
FIXED = (BASE, 'station_hidden_trees', OBSERVED, 'expanded_upstream', 'expanded_uniform',
         'expanded_matched_upstream', 'expanded_matched_nonupstream')
ARMS = {'refitted_state': ('raw', 'fitted', 'dynamic'),
        'crossfit_state': ('raw', 'crossfit', 'dynamic'),
        'crossfit_uniform': ('raw', 'crossfit', 'uniform'),
        'crossfit_matched_upstream': ('matched', 'crossfit', 'dynamic'),
        'crossfit_matched_nonupstream': ('control', 'crossfit', 'dynamic')}


def extract_design(partition, seed, records):
    """Read only retained assets; archived original forests are not required."""
    current = REFERENCE/'runs'/f'split{partition}_seed{seed}'
    config = consumed(current, ['available_real.pt', 'available_real_memory.json',
        'attention_candidates.npz'], records)
    parent, source, learning = (Path(config[f'{key}_run']) for key in ('parent', 'source', 'learning'))
    consumed(parent, ['source_oof.npz'], records)
    consumed(source, ['joint_source_inputs.npz'], records)
    consumed(learning, ['innovation_inputs.npz'], records)
    data = torch.load(config['dataset_path'], weights_only=False, map_location='cpu')
    with np.load(config['mask_path'], allow_pickle=False) as split:
        cells = split['train'].copy()
    t, ids = data['y'].shape[1], np.unique(cells//data['y'].shape[1])
    with np.load(source/'joint_source_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(ids, saved['source_station_ids'])
        inputs = {key: saved[key].copy() for key in ('raw', 'env', 'age', 'support', 'extra')}
    with np.load(learning/'innovation_inputs.npz', allow_pickle=False) as saved:
        inputs['extra'] = np.concatenate([inputs['extra'], saved['source_real']], -1)
    with np.load(parent/'source_oof.npz', allow_pickle=False) as saved:
        tree = np.maximum(0., np.expm1(saved['pred_z'][ids]))
    inputs['attention_reference'] = np.nan_to_num(tree)
    with np.load(current/'attention_candidates.npz', allow_pickle=False) as saved:
        candidates = {key[len('source_'):]: saved[key].copy() for key in saved.files if key.startswith('source_')}
    view = source_level_input_view(inputs, candidates, 'real')
    model = AvailableSourceAttentionResidual.from_payload(torch.load(current/'available_real.pt', weights_only=False))
    for name in ('spatial', 'temporal', 'decay', 'head'):
        getattr(model, name).requires_grad_(False)
    local = np.searchsorted(ids, cells//t)*t+cells % t
    design = readout_design(model, view, local)
    memory = json.loads((current/'available_real_memory.json').read_text())
    gamma = memory['selected']['gamma']
    native = tree.ravel()[local]
    norm = memory['normalization']
    coef = np.asarray(memory['coefficients'])[cells//t]
    memory_delta = norm['residual_scale']*(coef[:, 0]+coef[:, 1]*
        (np.log1p(native)-norm['log_context_mean'])/norm['log_context_sd'])
    offset, factor = native+gamma*memory_delta, (1-gamma)*model.selected_scale_
    frozen_head = torch.nn.Linear(design.shape[1], 1)
    with torch.no_grad():
        frozen_head.weight.copy_(torch.cat([model.head.linear.weight, model.head.output.weight], -1))
        frozen_head.bias.copy_(model.head.linear.bias)
    # Check that exported features really reproduce this retained readout.
    arrays = model._prepare_inputs(view)
    with torch.no_grad():
        expected = np.concatenate([model._delta_cells(arrays, local[start:start+512]).numpy()
                                  for start in range(0, len(local), 512)])
        exported = frozen_head(torch.as_tensor(design)).squeeze(-1).numpy()
    np.testing.assert_allclose(exported, expected, rtol=2e-5, atol=2e-5)
    native_clipped = np.maximum(0., native+model.selected_scale_*expected)
    original = np.maximum(0., native+(1-gamma)*(native_clipped-native)+gamma*memory_delta)
    return {'design': design, 'tree': native, 'offset': offset, 'factor': factor,
        'original': original, 'frozen_head': frozen_head.state_dict(), 'cells': cells,
        'y': np.asarray(data['y']).ravel()[cells].copy(), 'months': t,
        'folds': station_folds(cells, t, seed), 'gamma': gamma}


def observed_anchor(prediction, delta):
    result = np.maximum(0., np.expm1(np.log1p(prediction)+delta))
    result[delta == 0] = prediction[delta == 0]
    return result


def run_one(root, partition, seed, runtime):
    run = root/'runs'/f'split{partition}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('conditional execution changed')
        verify_files(run, 'complete.json', config)
        return
    run.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def progress(stage, row):
        record = {'run': run.name, 'stage': stage, **row, 'elapsed_seconds': time.monotonic()-start}
        write_json(root/'progress.json', record)
        print(json.dumps(record), flush=True)

    prior = PREVIOUS_ROOT/'runs'/run.name
    records = {}
    old = consumed(prior, ['fitting_inputs.npz', 'predictions.parquet', 'source_roles.json', 'matching.json'], records)
    for name, expected in old['sources'].items():
        if sha256_file(name) != expected:
            raise ValueError('consumed frozen input changed')
        records[name] = expected
    prepared = extract_design(partition, seed, records)
    with np.load(prior/'fitting_inputs.npz', allow_pickle=False) as saved:
        # Entire state bank is identical; do not save a second multi-GB copy.
        cache = {key: saved[key].copy() for key in ('source_complete_base', 'source_observed_base',
            'validation_complete_base', 'validation_observed_base', 'source_y', 'validation_y',
            'source_cells', 'val_cells', 'observed_support', 'nearest')}
    np.testing.assert_array_equal(prepared['cells'], cache['source_cells'])
    np.testing.assert_array_equal(prepared['y'], cache['source_y'])
    np.testing.assert_allclose(prepared['original'], cache['source_complete_base'], rtol=2e-5, atol=2e-5)
    obs_delta = np.log1p(cache['source_observed_base'])-np.log1p(cache['source_complete_base'])
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'q90_threshold_train', 'split_seed', 'seed')}, 'experiment': ROOT.name,
        'started_at': datetime.now(timezone.utc).isoformat(), 'runtime_snapshot_hash': runtime,
        'models': [*FIXED, *ARMS], 'arms': ARMS, 'sources': records,
        'river_input_run': str(prior), 'epochs': 30, 'patience': 5, 'readout_epochs': 30,
        'readout_selection': 'fixed 30 epochs; no new validation or held-label selection',
        'readout_coefficient': prepared['factor'], 'memory_gamma': prepared['gamma'],
        'readout_features': 'exact retained linear design plus frozen donor attention states',
        'complete_base_training_predictions_are_oof': False,
        'readout_loss_is_station_held': True,
        'conditional_scope': 'encoder/GRU/donor allocation and inputs/ecological coefficients/observed branch remain historical source fits',
        'question': 'conditional scalar-readout crossfit versus identically refitted in-sample readout',
        'validation_anchor': 'unchanged complete model plus frozen observed-DOC river branch',
        'query_and_message_features': 'unchanged preceding absolute environmental-state inputs',
        'receiving_water_quality': 'none', 'evaluation_role': 'source_validation_development',
        'external_inductive_evaluation': False, 'new_forest_fits': 0}
    if (run/'config.json').exists():
        config['started_at'] = json.loads((run/'config.json').read_text())['started_at']
        if json.loads((run/'config.json').read_text()) != json.loads(json.dumps(config)):
            raise ValueError('preserve saved conditional config')
    write_json(run/'config.json', config)
    files = ['config.json']
    if (run/'readouts_complete.json').exists():
        verify_files(run, 'readouts_complete.json', config)
        readouts = torch.load(run/'readouts.pt', weights_only=False)
        with np.load(run/'readout_inputs.npz', allow_pickle=False) as saved:
            fitted_base, crossfit_base = saved['source_fitted_base'].copy(), saved['source_crossfit_base'].copy()
        roles = json.loads((run/'readout_roles.json').read_text())
    else:
        settings = {'coefficient': prepared['factor'], 'seed': seed, 'epochs': 30,
                    'tail_threshold': config['q90_threshold_train'], 'tree': prepared['tree'],
                    'memory_gamma': prepared['gamma']}
        head, trace = fit_readout(prepared['design'], prepared['offset'], prepared['y'],
                                  np.arange(len(prepared['y'])), **settings)
        fitted_native = combine_readout(prepared['design'], head, prepared['offset'], prepared['factor'],
            tree=prepared['tree'], memory_gamma=prepared['gamma'])
        progress('all_source_readout', {'epochs': 30})
        crossfit_native, heads, roles = conditional_crossfit(prepared['design'], prepared['offset'],
            prepared['y'], prepared['cells'], prepared['folds'], n_months=prepared['months'], **settings)
        fitted_base, crossfit_base = (observed_anchor(value, obs_delta) for value in (fitted_native, crossfit_native))
        readouts = {'fitted': head.state_dict(), 'fold_heads': heads, 'fitted_trace': trace,
                    'frozen_head': prepared['frozen_head'],
                    'settings': {key: value for key, value in settings.items() if key != 'tree'}}
        torch.save(readouts, run/'readouts.pt')
        write_json(run/'readout_roles.json', roles)
        np.savez_compressed(run/'readout_inputs.npz', design=prepared['design'], tree=prepared['tree'],
            offset=prepared['offset'], obs_delta=obs_delta, source_fitted_base=fitted_base,
            source_crossfit_base=crossfit_base, source_cells=prepared['cells'], source_y=prepared['y'],
            original=prepared['original'], coefficient=np.array(prepared['factor']), memory_gamma=np.array(prepared['gamma']), months=np.array(prepared['months']))
        bind_files(run, 'readouts_complete.json', [run/name for name in ('readouts.pt', 'readout_roles.json', 'readout_inputs.npz')], config)
    files += ['readouts.pt', 'readout_roles.json', 'readout_inputs.npz', 'readouts_complete.json']
    panel = pd.read_parquet(prior/'predictions.parquet')
    frames = [panel[panel.model_name.isin(FIXED)].copy()]
    template = panel[panel.model_name.eq(BASE)].copy()
    progress('conditional_anchors', {'fitted_mae': float(abs(fitted_base-cache['source_y']).mean()),
        'crossfit_mae': float(abs(crossfit_base-cache['source_y']).mean()),
        'old_source_mae': float(abs(cache['source_observed_base']-cache['source_y']).mean()),
        'validation_mae': float(abs(cache['validation_observed_base']-cache['validation_y']).mean())})
    with np.load(prior/'fitting_inputs.npz', allow_pickle=False) as state_bank:
        for arm, (bank, anchor, allocation) in ARMS.items():
            source, validation = ({key: state_bank[f'{role}_{bank}_{key}'].copy()
                                   for key in ('query', 'features', 'states', 'valid')}
                                  for role in ('source', 'validation'))
            sb = fitted_base if anchor == 'fitted' else crossfit_base
            vb = cache['validation_observed_base']
            if (run/f'{arm}_complete.json').exists():
                verify_files(run, f'{arm}_complete.json', config)
                model = DynamicRiverState.from_payload(torch.load(run/f'{arm}.pt', weights_only=False))
            else:
                model = DynamicRiverState(source['query'].shape[-1], source['features'].shape[-1],
                    source['states'].shape[-1], allocation=allocation, seed=seed, epochs=30, patience=5)
                model.fit(source, sb, cache['source_y'], validation, vb, cache['validation_y'],
                    tail_threshold=config['q90_threshold_train'], progress=lambda row, a=arm: progress(a, row))
                torch.save(model.to_payload(), run/f'{arm}.pt')
                write_json(run/f'{arm}.json', model.to_dict())
                bind_files(run, f'{arm}_complete.json', [run/f'{arm}.pt', run/f'{arm}.json'], config)
            prediction, diag = model.predict(validation, vb, diagnostics=True)
            np.testing.assert_array_equal(prediction[~diag['support']], vb[~diag['support']])
            frame = template.copy()
            frame['model_name'], frame['y_pred'] = arm, prediction
            frame['state_support'], frame['observed_support'] = diag['support'], cache['observed_support']
            frame['state_delta_native'], frame['state_delta_log'] = prediction-vb, diag['delta_log']
            frame['state_prior_mass'], frame['state_entropy'] = diag['prior_mass'], diag['entropy']
            frame['nearest_path_km'] = np.where(np.isfinite(cache['nearest']), cache['nearest'], np.nan)
            for i, lag in enumerate((0, 1, 3)):
                frame[f'state_lag_mass_{lag}'] = diag['lag_mass'][:, i]
            frames.append(frame)
            np.savez_compressed(run/f'{arm}_diagnostics.npz', cells=cache['val_cells'], **diag)
            files += [f'{arm}{suffix}' for suffix in ('.pt', '.json', '_complete.json', '_diagnostics.npz')]
            progress(arm+'_saved', {'best_epoch': model.best_epoch_, 'validation_mae': float(abs(prediction-cache['validation_y']).mean())})
            del source, validation, model
            gc.collect()
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    bind_product(run, 'predictions.parquet', config, runtime, files)
    bind_files(run, 'complete.json', [run/name for name in (*files, 'predictions.parquet', 'predictions.meta.json')], config)
    progress('complete', {'readout_fits': 6, 'river_fits': len(ARMS), 'new_forest_fits': 0})
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
    for name in ('scripts/run_ladder.py', 'scripts/run_doc_river_readout_crossfit_v2.py',
        'scripts/run_doc_dynamic_river_v1.py', 'scripts/run_doc_hydro_river_expansion_v1.py',
        'scripts/run_unified_doc_spatial.py', 'scripts/run_doc_tail_residual_v1.py',
        'tests/test_conditional_readout_crossfit_v2.py', str(ROOT/'study_plan.md')):
        snapshot[name] = sha256_file(name)
    path = args.root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('changed execution requires new version')
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
