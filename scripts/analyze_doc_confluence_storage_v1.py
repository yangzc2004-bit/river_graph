"""Quantify explicit mixing/storage gains and their observed-source coverage."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_confluence_storage_v1 import ARMS, BASE, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json

COMPARISONS = [('confluence_storage', 'plain_upstream'), ('confluence_storage', 'confluence_mixing'),
    ('confluence_storage', 'dynamic_lagged'), ('confluence_storage', 'expanded_upstream'),
    ('confluence_storage', BASE), ('confluence_storage', 'station_hidden_trees'),
    ('confluence_mixing', 'plain_upstream'), ('plain_upstream', 'dynamic_lagged'),
    ('matched_storage_upstream', 'matched_storage_nonupstream')]


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
    diagnostic = panel[panel.model_name.eq('confluence_storage')].copy()
    supported = diagnostic.river_support.eq(True)
    diagnostic['support_group'] = np.where(supported, 'observed_upstream', 'no_observed_upstream')
    diagnostic['flow_group'] = np.select([~supported, diagnostic.river_measured_flow_mass > .5],
        ['no_observed_upstream', 'mostly_measured_discharge'], default='mostly_area_proxy')
    diagnostic['mixing_group'] = np.select([~supported, diagnostic.river_frontier_count > 1],
        ['no_observed_upstream', 'multiple_represented_branches'], default='one_represented_branch')
    diagnostic['storage_group'] = np.select([~supported, diagnostic.river_storage_mass > .1],
        ['no_observed_upstream', 'mapped_storage_over_0_1'], default='mapped_storage_0_0_1')
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
        if candidate != 'confluence_storage' or reference not in ('plain_upstream', 'dynamic_lagged', BASE):
            continue
        columns = ['support_group', 'flow_group', 'mixing_group', 'storage_group']
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
    fitting, support = [], []
    for path in sorted((root/'runs').glob('*/complete.json')):
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        for arm in ARMS:
            state = json.loads((run/f'{arm}.json').read_text())
            fitting.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'arm': arm,
                'best_epoch': state['best_epoch'], 'epochs_run': state['epochs_run'],
                'zero_selected': state['best_epoch'] == 0, 'parameters': state['trainable_parameters']})
        with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
            valid, frontier = saved['validation_raw_valid'], saved['validation_raw_frontier']
            measured = saved['validation_raw_measured_flow']
            count = frontier.sum(1)
            support.append({'split_seed': config['split_seed'], 'seed': config['seed'],
                'supported_cells': int(valid.any((1, 2)).sum()), 'total_cells': len(valid),
                'supported_fraction': float(valid.any((1, 2)).mean()),
                'candidate_lag_slots': int(valid.sum()), 'frontier_lag_slots': int(frontier.sum()),
                'nested_slots_removed': int((valid & ~frontier).sum()),
                'measured_flow_slots': int(measured.sum()), 'area_proxy_slots': int((frontier & ~measured).sum()),
                'multiple_branches_cell_fraction': float((count > 1).any(-1).mean())})
    pd.DataFrame(fitting).to_csv(out/'fitting_summary.csv', index=False)
    pd.DataFrame(support).to_csv(out/'source_support.csv', index=False)
    fields = ['river_prior_mass', 'river_entropy', 'river_frontier_count', 'river_measured_flow_mass',
        'river_attenuation_mean', 'river_storage_mass', 'river_lag_mass_0', 'river_lag_mass_1', 'river_lag_mass_3']
    supported = panel[panel.model_name.isin(ARMS) & panel.river_support.eq(True)]
    supported.groupby(['model_name', 'split_seed', 'seed'])[fields].mean().groupby('model_name').mean().to_csv(out/'operator_summary.csv')
    write_json(out/'analysis_definition.json', {'bootstrap_draws': draws, 'primary_model': 'confluence_storage',
        'estimand': 'seed mean within partition; three partitions equal', 'cluster': 'whole station jointly across partitions',
        'role': 'selected source-development validation', 'units': 'mg/L except log1p',
        'strata': 'descriptive operator diagnostics; selected branch allocation, not causal subgroups',
        'unique_stations': int(diagnostic.station.nunique()),
        'unique_station_months': int(diagnostic.cell.nunique()),
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
