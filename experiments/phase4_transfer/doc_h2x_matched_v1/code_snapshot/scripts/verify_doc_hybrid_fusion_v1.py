"""Replay all new/control checkpoints and independently recompute DOC scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import run_doc_dynamic_reconstruction_v1 as base
import run_doc_hybrid_fusion_v1 as study
import torch
from run_doc_fusion_benchmark_development_v1 import verify
from run_doc_river_architecture_comparison_v1 import digest, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_dynamic_reconstruction import DynamicRiverDOC


def independent(part):
    y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
    residual = p - y
    sse, sst = float(np.dot(residual, residual)), float(np.sum((y - y.mean())**2))
    means = part.groupby('station')[['y_true', 'y_pred']].transform('mean')
    a, b = y - means.y_true.to_numpy(float), p - means.y_pred.to_numpy(float)
    logged = pd.DataFrame({'station': part.station, 'a': np.log1p(y), 'b': np.log1p(p)})
    logcenter = logged[['a', 'b']] - logged.groupby('station')[['a', 'b']].transform('mean')
    ordered = part.sort_values(['station', 'month'])
    dy, dp, stations, month = (ordered[c].to_numpy(float) for c in ('y_true', 'y_pred', 'station', 'month'))
    valid = (np.diff(stations) == 0) & (np.diff(month) == 1)
    yd, pdiff = np.diff(dy)[valid], np.diff(dp)[valid]
    corr = float(np.corrcoef(yd, pdiff)[0, 1]) if len(yd) > 2 and np.std(yd) > 1e-6 and np.std(pdiff) > 1e-6 else np.nan
    return {'mae': float(sum(abs(residual)) / len(y)),
            'log1p_mae': float(np.mean(abs(np.log1p(p) - np.log1p(y)))),
            'rmse': float((sse / len(y))**.5), 'r2': 1 - sse / sst,
            'station_centered_r2': 1 - float(np.sum((b-a)**2)) / float(np.sum(a*a)),
            'log1p_station_centered_r2': 1 - float(np.sum((logcenter.b-logcenter.a)**2)) / float(np.sum(logcenter.a**2)),
            'within_station_sd_ratio': float(np.sqrt(np.sum(b*b) / np.sum(a*a))),
            'monthly_delta_mae': float(abs(pdiff-yd).mean()), 'monthly_delta_correlation': corr,
            'zero_change_delta_mae': float(abs(yd).mean()), 'monthly_pairs': len(yd)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=study.ROOT)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(1)
    protocol = json.loads((root / 'protocol.json').read_text())
    gate = json.loads((root / 'all_selections.json').read_text())
    assert gate['protocol_hash'] == digest(protocol)
    assert gate['test_used_for_selection'] is False
    assert gate['architecture_predeclared'] == protocol['model_config']
    assert gate['baseline_dependencies_hash'] == digest(protocol['baseline_dependencies'])
    assert protocol['joint_primary'] == ['mae', 'log1p_mae'] and protocol['uses_doc_inputs'] is False
    assert sha256_file(base.DATASET) == protocol['dataset_sha256']
    assert sha256_file(base.NODES) == protocol['nodes_sha256']
    assert sha256_file(study.PARENT / 'protocol.json') == protocol['parent_protocol_sha256']
    assert sha256_file(study.PARENT / 'analysis/independent_audit.json') == protocol['parent_audit_sha256']
    for file in study.CODE:
        if file != 'scripts/run_ladder.py':
            assert sha256_file(file) == protocol['runtime_snapshot'][file], file
    for name, h in protocol['mask_sha256'].items():
        assert sha256_file(root / 'masks' / f'{name}.npz') == h
    data = torch.load(base.DATASET, weights_only=False, map_location='cpu')
    parts, seen, replay_error, metric_error, feature_error = [], {}, 0., 0., 0.
    parent_summary = pd.read_csv(study.PARENT / 'analysis/by_seed.csv')
    with np.load(json.loads((study.PARENT / 'protocol.json').read_text())['daily']) as archive:
        parent_daily = archive['full'].copy()
    covariate_error = 0.
    validation_best_epoch = []
    with threadpool_limits(limits=1):
        for task in protocol['tasks']:
            d = study.directory(root, task)
            ctx = study.context(root, task, data)
            info = json.loads((d / 'info.json').read_text())
            if task['arm'] == study.NEW_ARM:
                config = study.identity(protocol, task)
                assert gate['new_fit_receipts'][task['id']] == sha256_file(d / 'selection_complete.json')
                verify(d, 'selection_complete.json', config)
                verify(d, 'test_complete.json', config)
                model = study.make_model(protocol)
                trace = json.loads((d / 'trace.json').read_text())
                best = trace[0]
                for row in trace[1:]:
                    if row['validation_mae'] < best['validation_mae'] - 1e-6:
                        best = row
                assert info['best_epoch'] == best['epoch'] and abs(info['validation_mae']-best['validation_mae']) < 1e-8
                assert info['epochs_run'] <= protocol['settings']['max_epochs']
                assert info['parameters'] == protocol['parameters'][study.NEW_ARM]
                validation_best_epoch.append(info['best_epoch'])
            else:
                for name, h in protocol['baseline_dependencies'][task['id']].items():
                    assert sha256_file(d / name) == h, (task['id'], name)
                verify(d, 'selection_complete.json')
                verify(d, 'test_complete.json')
                if task['arm'] == 'monthly_log':
                    model = DynamicRiverDOC(**{k: protocol['settings'][k] for k in ('hidden', 'temporal_hidden', 'heads', 'layers', 'dropout')})
                else:
                    model = joblib.load(d / 'model.joblib')
            if task['arm'] in ('monthly_log', study.NEW_ARM):
                model.load_state_dict(torch.load(d / 'model.pt', weights_only=True, map_location='cpu'))
            assert json.loads((d / 'normalization.json').read_text()) == ctx['inputs']['normalization']
            # Rebuild the parent monthly context with its real daily product:
            # monthly-view zero channels must make the feature tensors identical.
            with np.load(study.PARENT / 'masks' / f"{task['mask']}.npz") as old_mask:
                for role in ('train', 'val', 'test'):
                    np.testing.assert_array_equal(ctx['split'][role], old_mask[role])
            if task['arm'] == study.NEW_ARM:
                original = next(t for t in protocol['tasks'] if t['mask'] == task['mask'] and t['seed'] == task['seed'] and t['arm'] == 'monthly_log')
                old_norm = json.loads((study.directory(root, original) / 'normalization.json').read_text())
                assert old_norm == ctx['inputs']['normalization']
                old_ctx = base.context(study.PARENT, original, data, parent_daily)
                for name in ('windows', 'environment', 'season'):
                    error = float((ctx['inputs'][name] - old_ctx['inputs'][name]).abs().max())
                    covariate_error = max(covariate_error, error)
                    assert error == 0., name
                for role in ('train', 'val', 'test'):
                    for name in ('rows', 'relation', 'upstream_allowed', 'downstream_allowed'):
                        torch.testing.assert_close(ctx['graphs'][role][name], old_ctx['graphs'][role][name], atol=0, rtol=0)
            for role, name in (('val', 'validation_predictions'), ('test', 'predictions')):
                part = pd.read_parquet(d / f'{name}.parquet').sort_values('cell')
                cells = part.cell.to_numpy()
                assert not part.cell.duplicated().any()
                np.testing.assert_array_equal(cells, np.sort(ctx['split'][role]))
                np.testing.assert_array_equal(part.y_true.to_numpy(), np.asarray(data['y']).ravel()[cells])
                assert np.isfinite(part.y_pred).all() and (part.y_pred >= 0).all()
                if task['arm'] in ('monthly_log', study.NEW_ARM):
                    pred = base.neural_cells(model, ctx, ctx['split'][role], info['scales'], protocol['settings']['batch_months'])
                    prediction = pd.Series(pred, index=ctx['split'][role]).loc[cells].to_numpy()
                else:
                    features = base.tree_features(ctx, cells)
                    assert features.shape[1] == info['features'] == 113
                    raw = model.predict(features).astype(float)
                    prediction = np.clip(np.expm1(np.minimum(raw, 15.)) if info['target'] == 'log1p' else raw, 0., np.expm1(15.))
                error = float(np.max(abs(prediction - part.y_pred.to_numpy())))
                replay_error = max(replay_error, error)
                assert error < 1e-5, (task['id'], role, error)
                key = (task['mask'], task['seed'], role)
                fingerprint = part[['cell', 'y_true']].to_numpy()
                if key in seen:
                    np.testing.assert_array_equal(fingerprint, seen[key])
                else:
                    seen[key] = fingerprint
                if role == 'val' and task['arm'] in ('monthly_log', study.NEW_ARM):
                    error = abs(independent(part)['mae'] - info['validation_mae'])
                    assert error < 1e-6
                if role == 'test':
                    parts.append(part)
            if task['arm'] != study.NEW_ARM:
                original = parent_summary[(parent_summary.track == task['track']) &
                    (parent_summary.gap_months == task['gap_months']) & (parent_summary.seed == task['seed']) &
                    (parent_summary.model_name == task['arm'])]
                assert len(original) == 1
    frames = pd.concat(parts, ignore_index=True)
    records = [{'track': track, 'gap_months': gap, 'seed': seed, 'model_name': arm, **independent(part)}
               for (track, gap, seed, arm), part in frames.groupby(['track', 'gap_months', 'seed', 'model_name'])]
    values = pd.DataFrame(records).set_index(['track', 'gap_months', 'seed', 'model_name'])
    actual = pd.read_csv(root / 'analysis/by_seed.csv').set_index(['track', 'gap_months', 'seed', 'model_name'])
    for key, row in values.iterrows():
        for metric in ('mae', 'log1p_mae', 'rmse', 'r2', 'station_centered_r2',
                       'log1p_station_centered_r2', 'within_station_sd_ratio',
                       'monthly_delta_mae', 'monthly_delta_correlation',
                       'zero_change_delta_mae', 'monthly_pairs'):
            # Float32 consecutive-month subtraction in the original scoring
            # helper differs from this float64 replay only at rounding scale.
            error = abs(row[metric] - actual.loc[key, metric])
            metric_error = max(metric_error, float(error))
            assert error < 1e-6, (key, metric, error)
    for arm in ('monthly_log', 'rf_monthly', 'xgb_monthly'):
        a = actual.reset_index().query('model_name == @arm')
        b = parent_summary.query('model_name == @arm')
        merged = a.merge(b, on=['track', 'gap_months', 'seed', 'model_name'], suffixes=('_new', '_parent'), validate='one_to_one')
        for metric in ('mae', 'log1p_mae', 'rmse', 'r2', 'monthly_delta_mae'):
            feature_error = max(feature_error, float(abs(merged[metric+'_new'] - merged[metric+'_parent']).max()))
    assert feature_error < 1e-10
    report = {'passed': True, 'comparison_tasks': len(protocol['tasks']), 'new_fits': 24,
              'replayed_roles': 2*len(protocol['tasks']), 'prediction_replay_max_abs': replay_error,
              'metric_recompute_max_abs': metric_error, 'baseline_metric_difference': feature_error,
              'covariate_tensor_max_abs_difference': covariate_error,
              'matched_queries_original_truth_and_normalization': True, 'validation_only_selection': True,
              'original_monthly_input_only': True, 'prior_endpoints_unchanged': True,
              'new_best_epochs': validation_best_epoch, 'summary_sha256': sha256_file(root / 'analysis/summary.csv')}
    write_json(root / 'analysis/independent_audit.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
