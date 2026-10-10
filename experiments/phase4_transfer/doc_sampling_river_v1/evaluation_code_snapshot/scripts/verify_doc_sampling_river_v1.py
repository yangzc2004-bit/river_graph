"""Replay all sampling branches and optionally reconstruct causal priors."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from prepare_doc_sampling_river_v1 import prepare_metadata
from run_doc_sampling_river_v1 import (
    ARMS,
    ARRAY_KEYS,
    BASE,
    FIXED,
    OBSERVED_ROOT,
    ROOT,
    STATE_ROOT,
    prepare,
)
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.sampling_aware_river import SamplingAwareRiverResidual


def verify(run, runtime, rebuild):
    config = json.loads((run/'config.json').read_text())
    verify_files(run, 'complete.json', config)
    if config['runtime_snapshot_hash'] != runtime:
        raise ValueError('sampling execution identity changed')
    for path, content in config['sources'].items():
        if sha256_file(path) != content:
            raise ValueError('consumed fixed source changed')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['config_hash'] != digest(config) or meta['prediction_sha256'] != sha256_file(run/'predictions.parquet')
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('product identity mismatch')
    panel = pd.read_parquet(run/'predictions.parquet')
    if set(panel.model_name) != set(config['models']) or panel.duplicated(['model_name', 'cell']).any():
        raise ValueError('incomplete model panel')
    if not np.isfinite(panel[['y_pred', 'y_true']]).all().all() or (panel.y_pred < 0).any():
        raise ValueError('invalid DOC products')
    for directory, models in ((OBSERVED_ROOT, [name for name in FIXED if name != 'expanded_upstream']),
        (STATE_ROOT, ['expanded_upstream'])):
        prior = pd.read_parquet(directory/'runs'/run.name/'predictions.parquet')
        pd.testing.assert_frame_equal(panel[panel.model_name.isin(models)][prior.columns].reset_index(drop=True),
            prior[prior.model_name.isin(models)].reset_index(drop=True), check_exact=True)
    with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
        cache = {key: saved[key].copy() for key in saved.files}
    if rebuild:
        rebuilt = prepare(config['split_seed'], config['seed'], run.parent.parent)
        for role, arrays in rebuilt['matrices'].items():
            for key, value in arrays.items():
                np.testing.assert_array_equal(cache[f'{role}_{key}'], value)
        for key in ('source_base', 'validation_base', 'source_y', 'validation_y', 'source_cells', 'val_cells', 'nearest'):
            np.testing.assert_array_equal(cache[key], rebuilt[key])
    roles = json.loads((run/'source_roles.json').read_text())
    for row in roles:
        if set(row['query_station_ids']) & set(row['library_station_ids']):
            raise ValueError('held receiver fold in source library')
    for role in ('source', 'validation'):
        np.testing.assert_array_equal(cache[f'{role}_matched_timing'], cache[f'{role}_control_timing'])
        np.testing.assert_array_equal(cache[f'{role}_matched_valid'], cache[f'{role}_control_valid'])
        np.testing.assert_array_equal(cache[f'{role}_raw_timing'][..., :5], cache[f'{role}_shuffled_timing'][..., :5])
        np.testing.assert_array_equal(cache[f'{role}_raw_values'], cache[f'{role}_shuffled_values'])
        for bank in ('raw', 'matched', 'control', 'shuffled'):
            if np.any(cache[f'{role}_{bank}_timing'][~cache[f'{role}_{bank}_valid']] != 0):
                raise ValueError('hidden date/flow metadata in river slots')
    fits = []
    for arm, (operator, bank) in ARMS.items():
        model = SamplingAwareRiverResidual.from_payload(torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu'))
        if model.config['sampling_mode'] != operator or model.config['seed'] != config['seed']:
            raise ValueError('wrong sampling operator')
        arrays = {key: cache[f'validation_{bank}_{key}'] for key in ARRAY_KEYS}
        prediction, diagnostics = model.predict(arrays, cache['validation_base'], diagnostics=True)
        stored = panel[panel.model_name.eq(arm)]
        np.testing.assert_array_equal(stored.cell, cache['val_cells'])
        np.testing.assert_array_equal(stored.y_pred, prediction)
        np.testing.assert_array_equal(stored.y_true, cache['validation_y'])
        np.testing.assert_allclose(diagnostics['lag_mass'].sum(-1)+diagnostics['prior_mass'], 1., atol=2e-6)
        np.testing.assert_array_equal(prediction[~diagnostics['support']], cache['validation_base'][~diagnostics['support']])
        with np.load(run/f'{arm}_diagnostics.npz', allow_pickle=False) as saved:
            for key, value in diagnostics.items():
                np.testing.assert_array_equal(saved[key], value)
        base = panel[panel.model_name.eq(BASE)]
        np.testing.assert_array_equal(base.cell, stored.cell)
        fits.append({'arm': arm, 'best_epoch': model.best_epoch_, 'prediction_replay': 'bitwise',
            'mae': float(abs(prediction-stored.y_true).mean()), 'unsupported_base': 'bitwise'})
    np.testing.assert_array_equal(panel[panel.model_name.eq('plain_upstream')].y_pred.to_numpy(),
        panel[panel.model_name.eq('dynamic_lagged')].y_pred.to_numpy())
    return {'run': run.name, 'fits': fits, 'priors_rebuilt': 'exact' if rebuild else 'not requested',
        'fixed_products': 'bitwise unchanged', 'identity': 'verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--rebuild-inputs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare_metadata(args.root)
    paths = sorted((args.root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in paths} != expected:
        raise ValueError('nine complete source-development packages required')
    runtime = verify_runtime_snapshot(args.root)
    rows = [verify(p.parent, runtime, args.rebuild_inputs) for p in paths]
    write_json(args.root/'verification/replay.json', rows)
    print(f'Verified {len(rows)} packages and 54 river fits; fixed products unchanged.')


if __name__ == '__main__':
    main()
