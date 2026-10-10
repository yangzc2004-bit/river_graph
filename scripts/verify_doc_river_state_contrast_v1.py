"""Replay contrast values and fixed absolute-state/observed DOC anchors."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from run_doc_river_state_contrast_v1 import ARMS, FIXED, PREVIOUS_ROOT, ROOT, prepare
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.dynamic_river_state import DynamicRiverState


def verify(run, runtime, rebuild=False):
    config = json.loads((run/'config.json').read_text())
    verify_files(run, 'complete.json', config)
    if config['runtime_snapshot_hash'] != runtime:
        raise ValueError('state execution identity changed')
    for name, expected in config['sources'].items():
        if sha256_file(name) != expected:
            raise ValueError('consumed source changed')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['config_hash'] != digest(config)
            or meta['prediction_sha256'] != sha256_file(run/'predictions.parquet')
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('product sidecar failed')
    panel = pd.read_parquet(run/'predictions.parquet')
    prior = pd.read_parquet(PREVIOUS_ROOT/'runs'/run.name/'predictions.parquet')
    pd.testing.assert_frame_equal(panel[panel.model_name.isin(FIXED)][prior.columns].reset_index(drop=True),
        prior[prior.model_name.isin(FIXED)].reset_index(drop=True), check_exact=True)
    if set(panel.model_name) != set(config['models']) or panel.duplicated(['model_name', 'cell']).any():
        raise ValueError('fixed complete panel missing or duplicated')
    if not np.isfinite(panel[['y_true', 'y_pred']]).all().all() or (panel.y_pred < 0).any():
        raise ValueError('nonfinite state product')
    with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
        cache = {key: saved[key].copy() for key in saved.files}
    # Independent subtraction from the preceding raw-value bank. No contrast
    # helper is used here; attention inputs and eligibility must stay exact.
    with np.load(PREVIOUS_ROOT/'runs'/run.name/'fitting_inputs.npz', allow_pickle=False) as absolute:
        for role in ('source', 'validation'):
            cells = cache['source_cells'] if role == 'source' else cache['val_cells']
            receiver = absolute['states'].reshape(-1, cache['states'].shape[-1])[cells]
            for bank in ('raw', 'matched', 'control'):
                prefix = f'{role}_{bank}'
                for key in ('features', 'query', 'valid'):
                    np.testing.assert_array_equal(cache[f'{prefix}_{key}'], absolute[f'{prefix}_{key}'])
                source = absolute[f'{prefix}_states']
                bank_valid = absolute[f'{prefix}_valid']
                for start in range(0, len(cells), 512):
                    stop = min(start+512, len(cells))
                    valid = bank_valid[start:stop]
                    expected = np.where(valid[..., None],
                        source[start:stop]-receiver[start:stop, None, None], 0.)
                    np.testing.assert_array_equal(cache[f'{prefix}_states'][start:stop], expected)
            np.testing.assert_array_equal(cache[f'{role}_observed_base'], absolute[f'{role}_observed_base'])
        np.testing.assert_array_equal(cache['previous_support'], cache['validation_raw_valid'].any((1, 2)))
    if rebuild:
        rebuilt = prepare(config['split_seed'], config['seed'])
        for role, arrays in rebuilt['matrices'].items():
            for key, value in arrays.items():
                np.testing.assert_array_equal(cache[f'{role}_{key}'], value)
        for key in ('source_complete_base', 'source_observed_base', 'validation_complete_base',
            'validation_observed_base', 'source_y', 'validation_y', 'source_cells', 'val_cells',
            'observed_support', 'nearest', 'states', 'availability', 'previous_support'):
            np.testing.assert_array_equal(cache[key], rebuilt[key])
    roles = json.loads((run/'source_roles.json').read_text())
    for row in roles:
        if set(row['query_station_ids']) & set(row['library_station_ids']):
            raise ValueError('query station in state library')
    for role in ('source', 'validation'):
        np.testing.assert_array_equal(cache[f'{role}_matched_valid'], cache[f'{role}_control_valid'])
        np.testing.assert_array_equal(cache[f'{role}_matched_features'][..., :8],
                                     cache[f'{role}_control_features'][..., :8])
    rows = []
    for arm, (bank, anchor, allocation) in ARMS.items():
        model = DynamicRiverState.from_payload(torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu'))
        if model.config['allocation'] != allocation or model.config['seed'] != config['seed']:
            raise ValueError('wrong state model')
        arrays = {key: cache[f'validation_{bank}_{key}'] for key in ('query', 'features', 'states', 'valid')}
        base = cache[f'validation_{anchor}_base']
        prediction, diag = model.predict(arrays, base, diagnostics=True)
        stored = panel[panel.model_name.eq(arm)]
        np.testing.assert_array_equal(stored.cell, cache['val_cells'])
        np.testing.assert_array_equal(stored.y_pred, prediction)
        np.testing.assert_array_equal(prediction[~diag['support']], base[~diag['support']])
        with np.load(run/f'{arm}_diagnostics.npz', allow_pickle=False) as saved:
            for key, value in diag.items():
                np.testing.assert_array_equal(saved[key], value)
        np.testing.assert_allclose(diag['prior_mass']+diag['lag_mass'].sum(-1), 1., atol=2e-6)
        silent = {**arrays, 'states': arrays['states']*0, 'features': arrays['features']*0,
                  'valid': np.zeros_like(arrays['valid'])}
        np.testing.assert_array_equal(model.predict(silent, base), base)
        rows.append({'arm': arm, 'best_epoch': model.best_epoch_, 'epochs_run': model.epochs_run_,
            'prediction_replay': 'bitwise', 'no_state_preserves_anchor': True})
    return {'run': run.name, 'fits': rows, 'fixed_anchors': 'bitwise original',
            'contrast_subtraction_and_unchanged_candidates': 'exact',
            'state_and_matching_rebuild': 'exact' if rebuild else 'not requested'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--rebuild-inputs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = sorted((args.root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in paths} != expected:
        raise ValueError('all nine development packages required')
    runtime = verify_runtime_snapshot(args.root)
    records = [verify(path.parent, runtime, args.rebuild_inputs) for path in paths]
    write_json(args.root/'verification/replay.json', records)
    print(f'Verified {len(records)} packages, 36 upstream-to-local contrast fits; fixed anchors and absent-state predictions exact.')


if __name__ == '__main__':
    main()
