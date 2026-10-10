"""Verify conditional readout isolation, fixed banks and all river checkpoints."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_river_readout_crossfit_v2 import (
    ARMS,
    FIXED,
    PREVIOUS_ROOT,
    ROOT,
    extract_design,
)
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.conditional_readout_crossfit_v2 import fit_readout
from river_graph.models.dynamic_river_state import DynamicRiverState


def verify(run, runtime, rebuild):
    config = json.loads((run/'config.json').read_text())
    verify_files(run, 'complete.json', config)
    if config['runtime_snapshot_hash'] != runtime or config['complete_base_training_predictions_are_oof']:
        raise ValueError('execution identity or conditional interpretation changed')
    for name, expected in config['sources'].items():
        if sha256_file(name) != expected:
            raise ValueError('consumed frozen input changed')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['config_hash'] != digest(config) or meta['prediction_sha256'] != sha256_file(run/'predictions.parquet')
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('prediction identity failed')
    panel = pd.read_parquet(run/'predictions.parquet')
    prior = PREVIOUS_ROOT/'runs'/run.name
    original = pd.read_parquet(prior/'predictions.parquet')
    pd.testing.assert_frame_equal(panel[panel.model_name.isin(FIXED)][original.columns].reset_index(drop=True),
        original[original.model_name.isin(FIXED)].reset_index(drop=True), check_exact=True)
    if (set(panel.model_name) != set(config['models']) or panel.duplicated(['model_name', 'cell']).any()
            or not np.isfinite(panel[['y_true', 'y_pred']]).all().all() or (panel.y_pred < 0).any()):
        raise ValueError('invalid complete prediction panel')
    with np.load(run/'readout_inputs.npz', allow_pickle=False) as saved:
        readout = {key: saved[key].copy() for key in saved.files}
    heads = torch.load(run/'readouts.pt', weights_only=False)
    roles = json.loads((run/'readout_roles.json').read_text())
    x, offset, coefficient = readout['design'], readout['offset'], float(readout['coefficient'])
    # Independently replay only the saved scalar heads, not the crossfit helper.
    def replay(state, rows):
        head = torch.nn.Linear(x.shape[1], 1)
        head.load_state_dict(state)
        with torch.no_grad():
            delta = head(torch.as_tensor(x[rows])).squeeze(-1).double().numpy()
        gamma = float(readout['memory_gamma'])
        native = np.maximum(0., offset[rows]+coefficient*delta+
            (1-gamma)*np.maximum(0., -readout['tree'][rows]-coefficient/(1-gamma)*delta))
        observed = np.maximum(0., np.expm1(np.log1p(native)+readout['obs_delta'][rows]))
        observed[readout['obs_delta'][rows] == 0] = native[readout['obs_delta'][rows] == 0]
        return observed
    np.testing.assert_array_equal(replay(heads['fitted'], np.arange(len(x))), readout['source_fitted_base'])
    all_rows = []
    for state, role in zip(heads['fold_heads'], roles, strict=True):
        train, query = np.asarray(role['train_rows']), np.asarray(role['query_rows'])
        if set(role['held_station_ids']) & set(role['fitted_station_ids']):
            raise ValueError('held station in new readout loss')
        np.testing.assert_array_equal(np.sort(np.r_[train, query]), np.arange(len(x)))
        np.testing.assert_array_equal(np.unique(readout['source_cells'][train]//int(readout['months'])), role['fitted_station_ids'])
        np.testing.assert_array_equal(np.unique(readout['source_cells'][query]//int(readout['months'])), np.sort(role['held_station_ids']))
        np.testing.assert_array_equal(replay(state, query), readout['source_crossfit_base'][query])
        all_rows.extend(query.tolist())
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(x)))
    role = roles[0]
    altered = readout['source_y'].copy()
    altered[np.asarray(role['query_rows'])] = np.nan
    changed, _ = fit_readout(x, offset, altered, np.asarray(role['train_rows']), tree=readout['tree'], **heads['settings'])
    for key, value in changed.state_dict().items():
        torch.testing.assert_close(value, heads['fold_heads'][0][key], rtol=0, atol=0)
    if rebuild:
        rebuilt = extract_design(config['split_seed'], config['seed'], {})
        for a, b in (('design', 'design'), ('tree', 'tree'), ('offset', 'offset'),
                     ('original', 'original'), ('source_cells', 'cells'), ('source_y', 'y')):
            np.testing.assert_array_equal(readout[a], rebuilt[b])
    rows = []
    with np.load(prior/'fitting_inputs.npz', allow_pickle=False) as bank:
        for arm, (kind, _, allocation) in ARMS.items():
            model = DynamicRiverState.from_payload(torch.load(run/f'{arm}.pt', weights_only=False))
            if model.config['allocation'] != allocation or model.config['seed'] != config['seed']:
                raise ValueError('wrong river operator')
            arrays = {key: bank[f'validation_{kind}_{key}'] for key in ('query', 'features', 'states', 'valid')}
            base = bank['validation_observed_base']
            prediction, diag = model.predict(arrays, base, diagnostics=True)
            frame = panel[panel.model_name.eq(arm)]
            np.testing.assert_array_equal(frame.cell, bank['val_cells'])
            np.testing.assert_array_equal(frame.y_pred, prediction)
            np.testing.assert_array_equal(prediction[~diag['support']], base[~diag['support']])
            with np.load(run/f'{arm}_diagnostics.npz', allow_pickle=False) as saved:
                for key, value in diag.items():
                    np.testing.assert_array_equal(saved[key], value)
            np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., atol=2e-6)
            rows.append({'arm': arm, 'best_epoch': model.best_epoch_, 'prediction_replay': 'bitwise'})
        np.testing.assert_array_equal(bank['validation_matched_valid'], bank['validation_control_valid'])
    return {'run': run.name, 'readout_heads': 6, 'new_readout_held_labels_perturbation': 'bitwise',
        'conditional_features_rebuilt': rebuild, 'complete_model_oof': False,
        'fixed_products': 'bitwise', 'state_banks': 'same consumed file', 'fits': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--rebuild-inputs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = sorted((args.root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {path.parent.name for path in paths} != expected:
        raise ValueError('all nine packages required')
    runtime = verify_runtime_snapshot(args.root)
    records = [verify(path.parent, runtime, args.rebuild_inputs) for path in paths]
    write_json(args.root/'verification/replay.json', records)
    print('Verified nine packages, 54 conditional readout heads and 45 river fits; conditional scope preserved.')


if __name__ == '__main__':
    main()
