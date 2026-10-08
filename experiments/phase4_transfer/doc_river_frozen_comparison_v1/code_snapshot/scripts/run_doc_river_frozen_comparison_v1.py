"""Run four fixed DOC procedures on five whole-region withholding tasks."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, support_curves
from run_doc_tail_residual_v1 import bind_product
from run_unified_doc_spatial import bind_files, digest, verify_files, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS, development_labels
from river_graph.models.frozen_river_correction import (
    FrozenRiverCorrection,
    matched_messages,
    message_features,
)
from river_graph.models.relative_source_attention import relative_source_residual_grid

ROOT = Path('experiments/phase4_transfer/doc_river_frozen_comparison_v1')
REFERENCE = Path('experiments/phase4_transfer/doc_current_availability_attention_geographical_v1')
ATLAS = Path('experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis')
GEOMETRY = Path('experiments/phase4_transfer/doc_river_planform_typology_v1/analysis')
BASE_NAME = 'available_real_integrated'
ARMS = ('current_complete', 'simple_upstream', 'structure_upstream', 'matched_nonupstream')
CODE = ('scripts/run_ladder.py', 'scripts/run_doc_river_frozen_comparison_v1.py',
        'src/river_graph/models/frozen_river_correction.py',
        'src/river_graph/models/river_structure_residual.py',
        'src/river_graph/models/relative_source_attention.py',
        'scripts/run_doc_geographical_confirmation_v1.py', 'scripts/run_doc_tail_residual_v1.py',
        'scripts/run_unified_doc_spatial.py', 'tests/test_frozen_river_correction.py')
SHAPE_COLUMNS = ('basin_aspect', 'network_axis_ratio', 'drainage_density', 'mainstem_share',
                 'mainstem_sinuosity', 'tributary_balance', 'confluence_position')


def morphology_view(names, source_ids):
    frame = pd.read_csv(GEOMETRY/'station_planform.csv', dtype={'station': str}).set_index('station')
    raw = frame.reindex(names)[list(SHAPE_COLUMNS)].to_numpy(float)
    raw[:, :3] = np.log1p(raw[:, :3])
    source = raw[source_ids]
    median = np.nanmedian(source, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    quartile = np.nanquantile(source, [.25, .75], axis=0)
    width = np.where(np.isfinite(quartile[1]-quartile[0]), quartile[1]-quartile[0], 0.)
    width = np.maximum(width, .05)
    return np.clip((np.where(np.isfinite(raw), raw, median)-median)/width, -5, 5), {
        'columns': SHAPE_COLUMNS, 'median': median.tolist(), 'iqr': width.tolist(),
        'fitted_station_ids': source_ids.tolist()}


def prepare(region, seed):
    prior = REFERENCE/'runs'/f'huc4_{region}_seed{seed}'
    old = json.loads((prior/'config.json').read_text())
    verify_files(prior, 'complete.json', old)
    for kind in ('dataset', 'mask'):
        if sha256_file(old[f'{kind}_path']) != old[f'{kind}_hash']:
            raise ValueError('saved geographical inputs changed')
    data = torch.load(old['dataset_path'], weights_only=False, map_location='cpu')
    with np.load(old['mask_path'], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ('train', 'val', 'test', 'context')}
    data['y'] = development_labels(data, split)
    shape = data['y'].shape
    names = np.asarray(data['site_no'], str)
    with np.load(prior/'components.npz', allow_pickle=False) as saved:
        base = saved[BASE_NAME].copy()
    parent = Path(old['parent_run'])
    with np.load(parent/'oof.npz', allow_pickle=False) as saved:
        oof = saved['new'].copy()
    ids, residual = relative_source_residual_grid(data['y'], oof, split['train'])
    receivers = np.unique(np.r_[split['val'], split['test']]//shape[1])
    if np.intersect1d(ids, receivers).size:
        raise ValueError('calibration/target stations entered donor training')
    edges = pd.read_csv(ATLAS/'station_edge_paths.csv', dtype={'source': str, 'target': str})
    structures = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    true, fake, matching, raw_valid = matched_messages(edges, structures, names[receivers], names[ids], residual)
    daily, _, _ = load_daily_pack(DAILY_ROOT, old['dataset_hash'], shape)
    morphology, normal = morphology_view(names, ids)
    local = {role: np.searchsorted(receivers, split[role]//shape[1])*shape[1]+split[role] % shape[1]
             for role in ('val', 'test')}
    return {'old': old, 'prior': prior, 'data': data, 'split': split, 'names': names,
        'source_ids': ids, 'receivers': receivers, 'base': base, 'true': true, 'fake': fake,
        'matching': matching, 'raw_valid': raw_valid, 'daily': daily,
        'morphology': morphology, 'normalization': normal, 'local': local}


def features(prepared, role, arm):
    receiver = prepared['receivers']
    messages = prepared['fake' if arm == 'matched_nonupstream' else 'true']
    return message_features(messages, prepared['local'][role], prepared['base'][receiver],
        prepared['daily'][receiver], prepared['morphology'][receiver], structured=arm != 'simple_upstream')


def freeze_runtime(root):
    snapshot = {p: sha256_file(p) for p in (*CODE, str(ROOT/'study_plan.md'))}
    path = root/'runtime_snapshot.json'
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError('preserve fitted version; execution changed')
    if not path.exists():
        write_json(path, snapshot)
        for name in snapshot:
            target = root/'code_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def run_one(root, region, seed, runtime):
    run = root/'runs'/f'huc4_{region}_seed{seed}'
    if (run/'complete.json').exists():
        config = json.loads((run/'config.json').read_text())
        if config['runtime_snapshot_hash'] != runtime:
            raise ValueError('completed run belongs to a different version')
        verify_files(run, 'complete.json', config)
        return
    started = time.monotonic()
    p = prepare(region, seed)
    old, split, names = p['old'], p['split'], p['names']
    run.mkdir(parents=True, exist_ok=True)
    config = {**{key: old[key] for key in ('dataset_path', 'dataset_hash', 'mask_path', 'mask_hash',
        'target_huc4', 'split_seed', 'seed', 'q90_threshold_train')},
        'experiment': ROOT.name, 'started_at': datetime.now(timezone.utc).isoformat(),
        'runtime_snapshot_hash': runtime, 'models': list(ARMS),
        'base_run': str(p['prior']), 'base_completion_hash': sha256_file(p['prior']/'complete.json'),
        'correction_fit_role': 'cyclic-next held-gradient validation stations',
        'selection_role': 'source_validation_station_cv', 'base_validation_previously_used_for_selection': True,
        'evaluation_role': 'retrospective ST357 whole-HUC4 withholding; not external validation',
        'base_weights_frozen': True, 'complete_model_oof_claimed': False,
        'lags': [0, 1, 3], 'max_age_months': 12, 'max_candidates': 20,
        'alphas': FrozenRiverCorrection.ALPHAS, 'zero_correction_candidate': True,
        'physical_hashes': {str(folder/name): sha256_file(folder/name) for folder, name in (
            (ATLAS, 'station_edge_paths.csv'), (ATLAS, 'station_structure.csv'),
            (GEOMETRY, 'station_planform.csv'), (GEOMETRY, 'station_classes.csv'))}}
    if (run/'config.json').exists():
        previous = json.loads((run/'config.json').read_text())
        config['started_at'] = previous['started_at']
        if previous != json.loads(json.dumps(config)):
            raise ValueError('incomplete package belongs to a different saved design')
    write_json(run/'config.json', config)
    write_json(run/'morphology_normalization.json', p['normalization'])
    pd.DataFrame(p['matching'], columns=['receiver', 'slot', 'real_source', 'control_source',
        'source_is_nonancestor', 'source_log_area_difference']).to_csv(run/'matched_donors.csv', index=False)
    np.savez_compressed(run/'message_inputs.npz', receivers=p['receivers'], source_ids=p['source_ids'],
        raw_valid=p['raw_valid'], **{f'{kind}_{key}': value for kind in ('true', 'fake')
                                      for key, value in p[kind].items()})
    val = split['val']
    val_base, val_truth = p['base'].ravel()[val], np.asarray(p['data']['y']).ravel()[val]
    grids = {'current_complete': p['base'].copy()}
    files = ['config.json', 'message_inputs.npz', 'matched_donors.csv', 'morphology_normalization.json']
    scores = {}
    support = None
    for arm in ARMS[1:]:
        x, val_support = features(p, 'val', arm)
        model = FrozenRiverCorrection().fit(x, val_truth, val_base, names[val//p['base'].shape[1]])
        joblib.dump(model, run/f'{arm}.joblib')
        write_json(run/f'{arm}.json', model.to_dict())
        scores[arm] = model.to_dict()
        grid = p['base'].copy()
        for role in ('val', 'test'):
            xr, has_source = features(p, role, arm)
            cells = split[role]
            prediction = model.predict(xr, p['base'].ravel()[cells])
            np.testing.assert_array_equal(prediction[~has_source], p['base'].ravel()[cells][~has_source])
            grid.ravel()[cells] = prediction
            if role == 'test':
                if support is not None:
                    np.testing.assert_array_equal(support, has_source)
                support = has_source
        grids[arm] = grid
        files += [f'{arm}.joblib', f'{arm}.json']
        print(json.dumps({'run': run.name, 'arm': arm, 'cv_alpha': model.alpha_,
            'calibration_cells': len(val), 'supported_calibration_cells': int(val_support.sum()),
            'elapsed_seconds': time.monotonic()-started}), flush=True)
    np.savez_compressed(run/'components.npz', **grids)
    files.append('components.npz')
    bind_files(run, 'point_complete.json', [run/f for f in files], config)
    files.append('point_complete.json')
    # Open target labels only after saving every fitted state and full prediction.
    truth = np.asarray(torch.load(old['dataset_path'], weights_only=False, map_location='cpu')['y'], float)
    test, t = split['test'], p['base'].shape[1]
    frames = []
    for arm, grid in grids.items():
        frame = pd.DataFrame({'cell': test, 'station': names[test//t],
            'month': np.asarray(p['data']['months'], str)[test % t], 'model_name': arm, 'analyte': 'doc',
            'y_pred': grid.ravel()[test], 'y_true': truth.ravel()[test], 'visibility_role': 'test',
            'split_seed': int(region), 'target_huc4': region, 'seed': seed, 'k': 0,
            'river_supported': support, 'graph_delta': grid.ravel()[test]-p['base'].ravel()[test]})
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
    curves, adapters = support_curves(grids, p['data'], split, truth)
    curves['split_seed'], curves['target_huc4'], curves['seed'] = int(region), region, seed
    curves.to_parquet(run/'support_curves.parquet', index=False)
    write_json(run/'support_adapters.json', adapters)
    files.append('support_adapters.json')
    for name in ('predictions.parquet', 'support_curves.parquet'):
        bind_product(run, name, config, runtime, files)
        files += [name, name.replace('.parquet', '.meta.json')]
    bind_files(run, 'complete.json', [run/f for f in files], config)
    write_json(root/'progress.json', {'run': run.name, 'stage': 'complete', 'new_correction_fits': 3,
        'calibration_stations': len(np.unique(val//t)), 'test_stations': len(np.unique(test//t)),
        'test_cells': len(test), 'supported_test_cells': int(support.sum()), 'selection': scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--regions', nargs='+', choices=HUC4_BLOCKS, default=list(HUC4_BLOCKS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_only:
        for region in args.regions:
            p = prepare(region, args.seeds[0])
            print(json.dumps({'region': region, **{role: {'cells': len(p['split'][role]),
                'supported': int(features(p, role, 'structure_upstream')[1].sum())}
                for role in ('val', 'test')}}), flush=True)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    runtime = freeze_runtime(args.root)
    for region in args.regions:
        for seed in args.seeds:
            run_one(args.root, region, seed, runtime)
    write_json(args.root/'batch_complete.json', {'regions': args.regions, 'seeds': args.seeds,
        'procedures': ARMS, 'completed_packages': len(args.regions)*len(args.seeds),
        'runtime_snapshot_hash': runtime})


if __name__ == '__main__':
    main()
