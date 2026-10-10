"""Analyze incremental date/flow correspondence gains on fixed DOC queries."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_sampling_river_v1 import ARMS, BASE, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json

COMPARISONS = [('sampling_hydro', 'plain_upstream'), ('sampling_hydro', 'sampling_age'),
    ('sampling_hydro', 'shuffled_sampling_hydro'), ('sampling_hydro', 'dynamic_lagged'),
    ('sampling_hydro', 'expanded_upstream'), ('sampling_hydro', BASE),
    ('sampling_hydro', 'station_hidden_trees'), ('sampling_age', 'plain_upstream'),
    ('matched_sampling_upstream', 'matched_sampling_nonupstream')]


def analyze(root, draws):
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in (root/'runs').glob('*/complete.json')} != expected:
        raise ValueError('all nine fixed packages required')
    verify_runtime_snapshot(root)
    panel, thresholds = load_panel(root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    for name, frame in (('run_metrics', runs), ('partition_metrics', partitions), ('summary', summary)):
        frame.to_csv(out/f'{name}.csv', index=False)
    detail, fitting, support = [], [], []
    for path in sorted((root/'runs').glob('*/complete.json')):
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        diagnostic = panel[panel.model_name.eq('sampling_hydro') & panel.split_seed.eq(config['split_seed']) & panel.seed.eq(config['seed'])].copy()
        for arm in ARMS:
            state = json.loads((run/f'{arm}.json').read_text())
            weights = torch.load(run/f'{arm}.pt', weights_only=False, map_location='cpu')['state']['timing_encoder.weight'].numpy()
            fitting.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'arm': arm,
                'best_epoch': state['best_epoch'], 'epochs_run': state['epochs_run'],
                'zero_selected': state['best_epoch'] == 0, 'parameters': state['trainable_parameters'],
                'timing_encoder_norm': float(np.linalg.norm(weights)),
                'date_encoder_norm': float(np.linalg.norm(weights[:, :5])),
                'flow_encoder_norm': float(np.linalg.norm(weights[:, 5:]))})
        with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
            timing, valid = saved['validation_raw_timing'], saved['validation_raw_valid']
        count = valid.sum((1, 2))
        available = count > 0
        age = np.min(np.where(valid, timing[..., 0]*396, np.inf), axis=(1, 2))
        phase_fraction = (timing[..., 10]*valid).sum((1, 2))/np.maximum(count, 1)
        span = (timing[..., 2]*31*valid).sum((1, 2))/np.maximum(count, 1)
        receiver_phase = ((timing[..., 11] > 0) & valid).any((1, 2))
        multi = ((timing[..., 3] > np.log(2)/np.log(32)+1e-7) & valid).any((1, 2))
        diagnostic['youngest_source_age_days'] = np.where(available, age, np.nan)
        diagnostic['source_sample_span_days'] = span
        diagnostic['source_phase_fraction'] = phase_fraction
        diagnostic['receiver_phase_available'] = receiver_phase
        diagnostic['support_group'] = np.where(available, 'observed_upstream', 'no_observed_upstream')
        diagnostic['age_group'] = np.select([~available, age <= 31, age <= 93],
            ['no_observed_upstream', 'source_age_le31_days', 'source_age_32_93_days'], default='source_age_gt93_days')
        diagnostic['phase_group'] = np.select([~available, phase_fraction == 0, (phase_fraction > 0) & receiver_phase],
            ['no_observed_upstream', 'no_source_phase', 'source_and_receiver_phase'], default='source_phase_only')
        detail.append(diagnostic)
        support.append({'split_seed': config['split_seed'], 'seed': config['seed'],
            'total_cells': len(valid), 'supported_cells': int(available.sum()), 'supported_fraction': float(available.mean()),
            'candidate_lag_slots': int(valid.sum()), 'flow_known_slots': int(((timing[..., 5] > 0) & valid).sum()),
            'phase_known_slots': int(((timing[..., 10] > 0) & valid).sum()),
            'source_and_receiver_phase_cells': int(((phase_fraction > 0) & receiver_phase).sum()),
            'multiple_sample_day_cells': int(multi.sum()),
            'median_youngest_source_age_days': float(np.median(age[available])),
            'mean_source_sample_span_days': float(span[available].mean())})
    diagnostic = pd.concat(detail, ignore_index=True)
    effects, strata, stations = [], [], []
    for candidate, reference in COMPARISONS:
        pair = paired(panel, candidate, reference)
        for region in ('overall', 'q90', 'log1p'):
            selected = pair.copy()
            if region == 'q90':
                q = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = selected[selected.y_true_candidate >= q]
            if region == 'log1p':
                selected['candidate_error'] = abs(np.log1p(selected.y_pred_candidate)-np.log1p(selected.y_true_candidate))
                selected['reference_error'] = abs(np.log1p(selected.y_pred_reference)-np.log1p(selected.y_true_reference))
            direction = selected.groupby(['split_seed', 'seed'])[['candidate_error', 'reference_error']].mean()
            part = direction.groupby('split_seed').mean()
            effects.append({'candidate': candidate, 'reference': reference, 'region': region,
                **joint_station_bootstrap(selected, draws=draws),
                'improved_packages': int((direction.candidate_error < direction.reference_error).sum()),
                'improved_partitions': int((part.candidate_error < part.reference_error).sum())})
        losses = pair.groupby(['split_seed', 'station', 'cell'], as_index=False)[['candidate_error', 'reference_error']].mean()
        station = losses.groupby(['split_seed', 'station'], as_index=False).agg(candidate_mae=('candidate_error', 'mean'),
            reference_mae=('reference_error', 'mean'), n_cells=('cell', 'size'))
        station['candidate'], station['reference'] = candidate, reference
        station['delta_mae'] = station.candidate_mae-station.reference_mae
        stations.append(station)
        if candidate != 'sampling_hydro' or reference not in ('plain_upstream', 'expanded_upstream', BASE):
            continue
        columns = ['support_group', 'age_group', 'phase_group']
        joined = pair.merge(diagnostic[['split_seed', 'seed', 'cell', *columns]],
            on=['split_seed', 'seed', 'cell'], validate='one_to_one')
        for column in columns:
            for name, group in joined.groupby(column):
                strata.append({'candidate': candidate, 'reference': reference, 'stratum': name,
                    'grouping': column, **joint_station_bootstrap(group, draws=draws)})
    pd.DataFrame(effects).to_csv(out/'paired_effects.csv', index=False)
    stratified = pd.DataFrame(strata)
    stratified['sparse_station_group'] = stratified.n_stations_unique < 20
    stratified.to_csv(out/'stratified_effects.csv', index=False)
    station = pd.concat(stations, ignore_index=True)
    station.to_csv(out/'station_effects.csv', index=False)
    station.groupby(['candidate', 'reference', 'split_seed'])[['candidate_mae', 'reference_mae']].mean().groupby(
        ['candidate', 'reference']).mean().reset_index().to_csv(out/'station_equal.csv', index=False)
    diagnostic.to_parquet(out/'query_diagnostics.parquet', index=False)
    pd.DataFrame(fitting).to_csv(out/'fitting_summary.csv', index=False)
    pd.DataFrame(support).to_csv(out/'source_support.csv', index=False)
    fields = ['river_prior_mass', 'river_entropy', 'river_sample_age_mass_days', 'river_sample_flow_mass',
        'river_sample_phase_mass', 'river_lag_mass_0', 'river_lag_mass_1', 'river_lag_mass_3']
    supported = panel[panel.model_name.isin(ARMS) & panel.river_support.eq(True)]
    supported.groupby(['model_name', 'split_seed', 'seed'])[fields].mean().groupby('model_name').mean().to_csv(out/'operator_summary.csv')
    write_json(out/'analysis_definition.json', {'bootstrap_draws': draws, 'primary_model': 'sampling_hydro',
        'estimand': 'seed mean within partition; three partitions equal', 'cluster': 'whole station jointly across partitions',
        'role': 'selected source-development validation', 'units': 'mg/L except log1p',
        'strata': 'descriptive input support; sampling metadata never reads receiving labels/dates',
        'unique_stations': int(diagnostic.station.nunique()), 'unique_station_months': int(diagnostic.cell.nunique()),
        'partition_cell_occurrences': int(diagnostic[['split_seed', 'cell']].drop_duplicates().shape[0])})
    print(summary[['model_name', 'mae', 'q90_mae', 'log_mae']].to_string(index=False))
    frame = pd.DataFrame(effects)
    print(frame[frame.region.eq('overall')][['candidate', 'reference', 'relative_gain_pct',
        'gain_ci_low_pct', 'gain_ci_high_pct', 'improved_packages']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    analyze(args.root, args.bootstrap_draws)


if __name__ == '__main__':
    main()
