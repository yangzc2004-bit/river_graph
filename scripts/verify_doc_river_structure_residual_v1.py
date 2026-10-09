"""Replay source eligibility, sparse messages and integrated river predictions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_river_structure_residual_v1 import (
    ARMS,
    ROOT,
    integrate,
    prepare_river,
    views,
)
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.river_structure_residual import RiverStructureResidual


def verify(run, runtime):
    config = json.loads((run/'config.json').read_text())
    verify_files(run, 'complete.json', config)
    if runtime != config['runtime_snapshot_hash']:
        raise ValueError('river fitting identity changed')
    for name, expected in config['physical_inputs'].items():
        if sha256_file(name) != expected:
            raise ValueError('physical atlas changed')
    if sha256_file(config['current_run']+'/complete.json') != config['current_completion_hash']:
        raise ValueError('fixed complete reference changed')
    prepared = prepare_river(config['split_seed'], config['seed'])
    with np.load(run/'river_inputs.npz', allow_pickle=False) as saved:
        for name, rewired in (('real', False), ('rewired', True)):
            for role, arrays in zip(('source', 'validation'), prepared['river'][rewired][:2], strict=True):
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[f'{name}_{role}_{key}'], value)
    roles = {str(key): item[2] for key, item in prepared['river'].items()}
    if roles != json.loads((run/'river_source_roles.json').read_text()):
        raise ValueError('river receiving fold exclusion changed')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['config_hash'] != digest(config) or meta['prediction_sha256'] != sha256_file(run/'predictions.parquet')
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('river prediction identity failed')
    panel = pd.read_parquet(run/'predictions.parquet')
    prior = pd.read_parquet(prepared['current']/'predictions.parquet')
    fixed = ['available_real_integrated', 'available_real_native', 'station_hidden_trees']
    pd.testing.assert_frame_equal(panel[panel.model_name.isin(fixed)][prior.columns].reset_index(drop=True),
        prior[prior.model_name.isin(fixed)].reset_index(drop=True), check_exact=True)
    t, val, ids = prepared['data']['y'].shape[1], prepared['split']['val'], prepared['validation_ids']
    local = np.searchsorted(ids, val//t), val % t
    cells = np.ravel_multi_index(local, (len(ids), t))
    replay = []
    for arm, (mode, rewired) in ARMS.items():
        model = RiverStructureResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu'))
        if model.river_config != {'river_mode': mode} or digest(model._config()) != digest(config['training_settings']):
            raise ValueError('river operator or training settings changed')
        _, view = views(prepared, rewired)
        memory = EcologicalResidualTransfer.from_dict(json.loads((run/f'{arm}_memory.json').read_text()))
        predicted = integrate(prepared, memory, model.predict(view, prepared['validation_base']))
        stored = panel[panel.model_name.eq(arm)]
        np.testing.assert_array_equal(stored.cell, val)
        np.testing.assert_array_equal(stored.y_pred, predicted)
        silent = {**view, 'river_values': np.zeros_like(view['river_values']),
                  'river_valid': np.zeros_like(view['river_valid'])}
        zero = integrate(prepared, memory, model.predict(silent, prepared['validation_base']))
        np.testing.assert_array_equal(panel[panel.model_name.eq(f'{arm}_zero_river')].y_pred, zero)
        diag = model.river_diagnostics(view, cells)
        diag['river_delta_effective'] = predicted-zero
        with np.load(run/f'{arm}_diagnostics.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['cells'], val)
            for key, value in diag.items():
                np.testing.assert_array_equal(saved[key], value)
                if not np.isfinite(value).all():
                    raise ValueError('nonfinite river diagnostics')
        np.testing.assert_allclose(diag['river_prior_mass']+diag['river_lag_mass'].sum(-1), 1., rtol=0, atol=2e-6)
        if (diag['river_delta_effective'][diag['river_support'] == 0] != 0).any():
            raise ValueError('river correction exists without allowed source observations')
        replay.append({'arm': arm, 'best_epoch': model.best_epoch_, 'epochs': model.epochs_run_,
            'selected_scale': model.selected_scale_, 'validation_mae': float(np.abs(stored.y_pred-stored.y_true).mean())})
    return {'run': run.name, 'identity': 'verified', 'bank_replay': 'exact',
        'prediction_replay': 'bitwise', 'role': 'source_validation_development', 'fits': replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = sorted((args.root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if not complete or (not args.available_only and {p.parent.name for p in complete} != expected):
        raise ValueError('all nine fixed source packages are required for the final report')
    runtime = verify_runtime_snapshot(args.root)
    rows = [verify(path.parent, runtime) for path in complete]
    write_json(args.root/'verification/replay.json', rows)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
