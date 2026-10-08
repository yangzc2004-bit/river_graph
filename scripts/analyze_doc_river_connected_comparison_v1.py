"""Replay and analyze the nested four-procedure river comparison."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from analyze_doc_geographical_confirmation_v1 import additional_metrics, summaries
from analyze_doc_river_frozen_comparison_v1 import CONTRASTS, attach_groups, effect
from run_doc_river_connected_comparison_v1 import ROOT, prepare
from run_doc_river_frozen_comparison_v1 import ARMS, GEOMETRY
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file


def verify(run, runtime):
    config = json.loads((run/'config.json').read_text())
    if config['runtime_snapshot_hash'] != runtime:
        raise ValueError('incorrect connected station execution')
    for name in ('point_complete.json', 'complete.json'):
        verify_files(run, name, config)
    p = prepare(config['split_seed'], config['seed'])
    cells, t = p['cells'], p['dataset']['y'].shape[1]
    with np.load(run/'features.npz', allow_pickle=False) as saved:
        for key, expected in [('cells', cells), ('base', p['base']), ('support', p['support']), *p['matrices'].items()]:
            np.testing.assert_array_equal(saved[key], expected)
    with np.load(run/'message_inputs.npz', allow_pickle=False) as saved:
        for kind in ('true', 'fake'):
            for name, expected in p[kind].items():
                np.testing.assert_array_equal(saved[f'{kind}_{name}'], expected)
    states = json.loads((run/'readout_states.json').read_text())
    predicted = {'current_complete': p['base']}
    for arm in ARMS[1:]:
        output = np.empty(len(cells))
        for fold, held in enumerate(p['folds']):
            state = next(s for s in states if s['model_name'] == arm and s['outer_fold'] == fold)
            if (set(state['calibration_station_ids']) & set(held)
                    or set(p['source_ids']) & set(held)):
                raise ValueError('outer held station enters calibration or donor bank')
            query = np.isin(cells//t, held)
            model = joblib.load(run/f'{arm}_fold{fold}.joblib')
            output[query] = model.predict(p['matrices'][arm][query], p['base'][query])
        np.testing.assert_array_equal(output[~p['support']], p['base'][~p['support']])
        predicted[arm] = output
    with np.load(run/'point_predictions.npz', allow_pickle=False) as saved:
        for arm in ARMS:
            np.testing.assert_array_equal(saved[arm], predicted[arm])
    frame = pd.read_parquet(run/'predictions.parquet')
    meta = json.loads((run/'predictions.meta.json').read_text())
    if (meta['prediction_sha256'] != sha256_file(run/'predictions.parquet') or meta['config_hash'] != digest(config)
            or meta['run_identity_sha256'] != run_identity_sha256(digest(config), config['started_at'], runtime)):
        raise ValueError('unbound outer-query predictions')
    if (set(frame.model_name) != set(ARMS) or frame.duplicated(['model_name', 'cell']).any()
            or not frame.visibility_role.eq('outer_query').all()):
        raise ValueError('incomplete outer-query panel')
    for arm in ARMS:
        rows = frame[frame.model_name.eq(arm)]
        np.testing.assert_array_equal(rows.cell, cells)
        np.testing.assert_array_equal(rows.y_true, np.asarray(p['dataset']['y']).ravel()[cells])
        np.testing.assert_array_equal(rows.station, p['names'][cells//t])
        np.testing.assert_array_equal(rows.month, np.asarray(p['dataset']['months'], str)[cells % t])
        np.testing.assert_array_equal(rows.y_pred, predicted[arm])
    return {'run': run.name, 'outer_station_exclusion': True, 'baseline_frozen': True,
        'replay': 'bitwise', 'cells': len(cells)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    snapshot = json.loads((args.root/'runtime_snapshot.json').read_text())
    for name, expected in snapshot.items():
        if sha256_file(args.root/'code_snapshot'/name) != expected:
            raise ValueError('saved connected execution source changed')
    forms = pd.read_csv(GEOMETRY/'station_classes.csv', dtype={'station': str}).set_index('station')
    frames, thresholds, verification, selection, cv, sources, support = [], {}, [], [], [], {}, []
    for partition in (142, 143, 144):
        for seed in (42, 43, 44):
            run = args.root/'runs'/f'split{partition}_seed{seed}'
            verification.append(verify(run, digest(snapshot)))
            config = json.loads((run/'config.json').read_text())
            frame = attach_groups(run, pd.read_parquet(run/'predictions.parquet'), forms)
            frames.append(frame)
            thresholds[(partition, seed)] = config['q90_threshold_train']
            rows = frame[frame.model_name.eq('current_complete')]
            support.append({'split_seed': partition, 'seed': seed, 'n_stations': rows.station.nunique(),
                'n_cells': len(rows), 'supported_stations': rows[rows.river_supported].station.nunique(),
                'supported_cells': int(rows.river_supported.sum()), 'raw_supported_cells': int(rows.raw_supported.sum())})
            for row in json.loads((run/'readout_states.json').read_text()):
                selection.append({'split_seed': partition, 'seed': seed, **{key: row[key] for key in (
                    'model_name', 'outer_fold', 'selected_alpha', 'coefficient_norm', 'supported_calibration_stations',
                    'supported_query_stations', 'n_calibration_cells')}, 'zero_selected': row['selected_alpha'] is None})
                for value in row['station_cv_scores']:
                    cv.append({'split_seed': partition, 'seed': seed, 'model_name': row['model_name'],
                        'outer_fold': row['outer_fold'], **value, 'zero_reference_mae': row['station_cv_scores'][0]['validation_mae']})
            sources[str(run/'predictions.parquet')] = sha256_file(run/'predictions.parquet')
    output = args.root/'analysis'
    output.mkdir(exist_ok=True)
    panel = pd.concat(frames, ignore_index=True)
    runs, parts, summary, stations = summaries(panel, thresholds)
    for name, table in (('run_metrics', runs), ('partition_metrics', parts), ('summary', summary),
                        ('station_metrics', stations), ('q90_diagnostics', additional_metrics(panel, thresholds))):
        table.to_csv(output/f'{name}.csv', index=False)
    effects, strata, station_effects = [], [], []
    for candidate, reference in CONTRASTS:
        for subset, view in (('all', panel), ('supported', panel[panel.river_supported]),
                             ('unsupported', panel[~panel.river_supported]), ('q90', panel[panel['tail']])):
            result, pair = effect(view, candidate, reference, args.bootstrap_draws, subset=subset)
            effects.append(result)
            if subset == 'all':
                rows = pair.groupby(['split_seed', 'station'], as_index=False).agg(
                    candidate_mae=('candidate_error', 'mean'), reference_mae=('reference_error', 'mean'), n_cells=('cell', 'nunique'))
                rows['candidate'], rows['reference'] = candidate, reference
                station_effects.append(rows)
        for column in ('form', 'distance_group'):
            for group, view in panel.groupby(column):
                result, _ = effect(view, candidate, reference, args.bootstrap_draws, stratum=column, group=group)
                strata.append(result)
    pd.DataFrame(effects).to_csv(output/'paired_effects.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'form_support_effects.csv', index=False)
    pd.concat(station_effects, ignore_index=True).to_csv(output/'station_effects.csv', index=False)
    pd.DataFrame(selection).to_csv(output/'readout_selection.csv', index=False)
    pd.DataFrame(cv).to_csv(output/'station_cv_scores.csv', index=False)
    pd.DataFrame(support).to_csv(output/'availability_by_run.csv', index=False)
    base_rows = panel[panel.model_name.eq('current_complete')]
    unique = base_rows.groupby('cell', as_index=False).agg(station=('station', 'first'),
        form=('form', 'first'), form_status=('form_status', 'first'), river_supported=('river_supported', 'any'))
    unique.groupby(['form', 'form_status'], as_index=False).agg(n_stations=('station', 'nunique'),
        n_cells=('cell', 'size'), supported_cells=('river_supported', 'sum')).to_csv(output/'form_population.csv', index=False)
    write_json(args.root/'verification'/'replay.json', verification)
    write_json(output/'sources.json', {'analyzer_sha256': sha256_file(__file__), 'inputs': sources,
        'bootstrap_draws': args.bootstrap_draws, 'cluster': 'station jointly across overlapping splits and seeds',
        'estimand': 'seed-mean cell error, then equal split means',
        'n_unique_cells': len(unique), 'n_unique_stations': unique.station.nunique(),
        'n_supported_cells': int(unique.river_supported.sum()), 'n_supported_stations': unique[unique.river_supported].station.nunique(),
        'unique_support_definition': 'supported in at least one saved partition; inference libraries differ across partitions',
        'scope': 'nested station-held retrospective source development; base previously selected on these labels'})
    print(summary[['model_name', 'mae', 'q90_mae', 'station_equal_mae']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
