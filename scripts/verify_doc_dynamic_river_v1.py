"""Replay the dynamic river comparison and its fixed reference products."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_dynamic_river_v1 import ARMS, BASE, FIXED, REFERENCE, ROOT, prepare
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.dynamic_river_residual import DynamicRiverResidual


def verify(run, runtime, rebuild=False):
    config = json.loads((run/'config.json').read_text())
    verify_files(run, 'complete.json', config)
    if config['runtime_snapshot_hash'] != runtime:
        raise ValueError('dynamic river identity changed')
    for name, content in config['sources'].items():
        if sha256_file(name) != content:
            raise ValueError('consumed source changed')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['config_hash'] != digest(config)
            or meta['prediction_sha256'] != sha256_file(run/'predictions.parquet')
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('prediction sidecar identity failed')
    panel = pd.read_parquet(run/'predictions.parquet')
    if set(panel.model_name) != set(config['models']) or panel.duplicated(['model_name', 'cell']).any():
        raise ValueError('complete fixed model panel required')
    if not np.isfinite(panel[['y_true', 'y_pred']]).all().all() or (panel.y_pred < 0).any():
        raise ValueError('nonfinite DOC result')
    prior = pd.read_parquet(REFERENCE/'runs'/run.name/'predictions.parquet')
    pd.testing.assert_frame_equal(panel[panel.model_name.isin(FIXED)][prior.columns].reset_index(drop=True),
        prior[prior.model_name.isin(FIXED)].reset_index(drop=True), check_exact=True)
    with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
        cache = {key: saved[key].copy() for key in saved.files}
    if rebuild:
        rebuilt = prepare(config['split_seed'], config['seed'])
        for role, arrays in rebuilt['matrices'].items():
            for key, value in arrays.items():
                np.testing.assert_array_equal(cache[f'{role}_{key}'], value)
        for key in ('source_base', 'validation_base', 'source_y', 'validation_y', 'source_cells', 'val_cells', 'nearest'):
            np.testing.assert_array_equal(cache[key], rebuilt[key])
    roles = json.loads((run/'source_roles.json').read_text())
    for row in roles:
        if set(row['query_station_ids']) & set(row['library_station_ids']):
            raise ValueError('receiving training stations enter donor bank')
    for key in ('valid',):
        for role in ('source', 'validation'):
            np.testing.assert_array_equal(cache[f'{role}_matched_{key}'], cache[f'{role}_control_{key}'])
    replay = []
    for arm, (mode, bank) in ARMS.items():
        model = DynamicRiverResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu'))
        if model.config['mode'] != mode or model.config['seed'] != config['seed']:
            raise ValueError('wrong dynamic river model')
        arrays = {key: cache[f'validation_{bank}_{key}'] for key in ('query', 'features', 'values', 'valid')}
        predicted, diag = model.predict(arrays, cache['validation_base'], diagnostics=True)
        stored = panel[panel.model_name.eq(arm)]
        np.testing.assert_array_equal(stored.cell, cache['val_cells'])
        np.testing.assert_array_equal(stored.y_pred, predicted)
        np.testing.assert_array_equal(predicted[~diag['support']], cache['validation_base'][~diag['support']])
        with np.load(run/f'{arm}_diagnostics.npz', allow_pickle=False) as saved:
            for key, value in diag.items():
                np.testing.assert_array_equal(saved[key], value)
        np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., rtol=0, atol=2e-6)
        silent = {**arrays, 'values': np.zeros_like(arrays['values']), 'valid': np.zeros_like(arrays['valid'])}
        np.testing.assert_array_equal(model.predict(silent, cache['validation_base']), cache['validation_base'])
        base = panel[panel.model_name.eq(BASE)]
        np.testing.assert_array_equal(base.cell, stored.cell)
        replay.append({'arm': arm, 'best_epoch': model.best_epoch_, 'epochs_run': model.epochs_run_,
                       'mae': float(np.abs(predicted-stored.y_true).mean()),
                       'unsupported_exact': True, 'prediction_replay': 'bitwise'})
    return {'run': run.name, 'fits': replay, 'bank_rebuild': 'exact' if rebuild else 'not requested',
            'identity': 'verified', 'source_scope': 'consumed retained assets'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--rebuild-inputs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = sorted((args.root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {path.parent.name for path in paths} != expected:
        raise ValueError('all nine packages are required')
    runtime = verify_runtime_snapshot(args.root)
    rows = [verify(path.parent, runtime, args.rebuild_inputs) for path in paths]
    write_json(args.root/'verification/replay.json', rows)
    print(f'Verified {len(rows)} packages, {sum(len(row["fits"]) for row in rows)} fits; exact no-source protection.')


if __name__ == '__main__':
    main()
