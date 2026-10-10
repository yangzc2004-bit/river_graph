"""Compare conditional residual anchors and river prediction gains."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_river_readout_crossfit_v2 import ARMS, BASE, OBSERVED, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json
from scipy.stats import wasserstein_distance


def analyze(root, draws):
    paths = sorted((root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in paths} != expected:
        raise ValueError('all nine source-development packages required')
    verify_runtime_snapshot(root)
    panel, thresholds = load_panel(root)
    runs, parts, summary = metric_summary(panel, thresholds)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    for name, frame in (('run_metrics', runs), ('partition_metrics', parts), ('summary', summary)):
        frame.to_csv(out/f'{name}.csv', index=False)
    comparisons = [('crossfit_state', 'refitted_state'), ('crossfit_state', 'expanded_upstream'),
        ('crossfit_state', BASE), ('crossfit_state', OBSERVED), ('crossfit_state', 'station_hidden_trees'),
        ('refitted_state', 'expanded_upstream'), ('crossfit_state', 'crossfit_uniform'),
        ('crossfit_matched_upstream', 'crossfit_matched_nonupstream')]
    effects, strata, stations = [], [], []
    diagnostic = panel[panel.model_name.eq('crossfit_state')].copy()
    diagnostic['support_group'] = np.select([
        diagnostic.observed_support.eq(True) & diagnostic.state_support.eq(True),
        diagnostic.observed_support.eq(False) & diagnostic.state_support.eq(True),
        diagnostic.observed_support.eq(True) & diagnostic.state_support.eq(False)],
        ['observed_and_state', 'state_without_DOC', 'observed_without_state'], default='neither_channel')
    diagnostic['distance_group'] = np.select([diagnostic.nearest_path_km.isna(),
        diagnostic.nearest_path_km <= 50, diagnostic.nearest_path_km <= 200],
        ['no_eligible_path', 'path_0_50_km', 'path_50_200_km'], default='path_over_200_km')
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ('overall', 'q90', 'log1p'):
            subset = pair.copy()
            if region == 'q90':
                q = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                subset = subset[subset.y_true_candidate >= q]
            if region == 'log1p':
                subset['candidate_error'] = abs(np.log1p(subset.y_pred_candidate)-np.log1p(subset.y_true_candidate))
                subset['reference_error'] = abs(np.log1p(subset.y_pred_reference)-np.log1p(subset.y_true_reference))
            direction = subset.groupby(['split_seed', 'seed'])[['candidate_error', 'reference_error']].mean()
            partition = direction.groupby('split_seed').mean()
            effects.append({'candidate': candidate, 'reference': reference, 'region': region,
                **joint_station_bootstrap(subset, draws=draws),
                'improved_packages': int((direction.candidate_error < direction.reference_error).sum()),
                'improved_partitions': int((partition.candidate_error < partition.reference_error).sum())})
        losses = pair.groupby(['split_seed', 'station', 'cell'], as_index=False)[['candidate_error', 'reference_error']].mean()
        station = losses.groupby(['split_seed', 'station'], as_index=False).agg(
            candidate_mae=('candidate_error', 'mean'), reference_mae=('reference_error', 'mean'), n_cells=('cell', 'size'))
        station['candidate'], station['reference'] = candidate, reference
        station['delta_mae'] = station.candidate_mae-station.reference_mae
        stations.append(station)
        if candidate != 'crossfit_state' or reference not in ('refitted_state', 'expanded_upstream', BASE):
            continue
        joined = pair.merge(diagnostic[['split_seed', 'seed', 'cell', 'support_group', 'distance_group',
            'state_support', 'observed_support']], on=['split_seed', 'seed', 'cell'], validate='one_to_one')
        groups = [('usable_state', joined[joined.state_support.eq(True)])]
        groups += [(str(name), group) for col in ('support_group', 'distance_group') for name, group in joined.groupby(col)]
        for name, group in groups:
            if not group.empty:
                strata.append({'candidate': candidate, 'reference': reference, 'stratum': name,
                    **joint_station_bootstrap(group, draws=draws)})
    effects = pd.DataFrame(effects)
    effects.to_csv(out/'paired_effects.csv', index=False)
    strata = pd.DataFrame(strata)
    strata['sparse_station_group'] = strata.n_stations_unique < 20
    strata.to_csv(out/'stratified_effects.csv', index=False)
    station = pd.concat(stations, ignore_index=True)
    station.to_csv(out/'station_effects.csv', index=False)
    station.groupby(['candidate', 'reference', 'split_seed'])[['candidate_mae', 'reference_mae']].mean().groupby(
        ['candidate', 'reference']).mean().reset_index().to_csv(out/'station_equal.csv', index=False)
    anchors, fitting, coverage = [], [], []
    for path in paths:
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        with np.load(config['river_input_run']+'/fitting_inputs.npz', allow_pickle=False) as saved, np.load(run/'readout_inputs.npz', allow_pickle=False) as readout:
            vy = saved['validation_y']
            vz = np.log1p(vy)-np.log1p(saved['validation_observed_base'])
            for name, value in (('original_fitted', saved['source_observed_base']),
                ('refitted_fitted', readout['source_fitted_base']), ('conditional_crossfit', readout['source_crossfit_base']),
                ('validation_anchor', saved['validation_observed_base'])):
                y = vy if name == 'validation_anchor' else saved['source_y']
                cells = saved['val_cells'] if name == 'validation_anchor' else saved['source_cells']
                tail = y >= config['q90_threshold_train']
                residual = np.log1p(y)-np.log1p(value)
                anchors.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'anchor': name,
                    'mae': float(abs(value-y).mean()), 'log_mae': float(abs(residual).mean()),
                    'q90_mae': float(abs(value[tail]-y[tail]).mean()), 'log_bias': float(residual.mean()),
                    'log_residual_q10': float(np.quantile(residual, .1)), 'log_residual_q90': float(np.quantile(residual, .9)),
                    'wasserstein_to_validation_log_residual': float(wasserstein_distance(residual, vz)),
                    'n_cells': len(y), 'n_stations': len(np.unique(cells//int(readout['months'])))})
            coverage.append({'split_seed': config['split_seed'], 'seed': config['seed'],
                'state_fraction': float(saved['validation_raw_valid'].any((1, 2)).mean()),
                'observed_fraction': float(saved['observed_support'].mean()),
                'matched_fraction': float(saved['validation_matched_valid'].any((1, 2)).mean())})
        for arm in ARMS:
            fitted = json.loads((run/f'{arm}.json').read_text())
            fitting.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'arm': arm,
                'best_epoch': fitted['best_epoch'], 'epochs_run': fitted['epochs_run'], 'selected_zero': fitted['best_epoch'] == 0})
    pd.DataFrame(anchors).to_csv(out/'training_anchor_diagnostics.csv', index=False)
    pd.DataFrame(fitting).to_csv(out/'fitting_summary.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'source_support.csv', index=False)
    diagnostic.to_parquet(out/'query_diagnostics.parquet', index=False)
    write_json(out/'analysis_definition.json', {'bootstrap_draws': draws, 'primary': 'crossfit_state versus refitted_state',
        'estimand': 'seed mean cell loss within partition, then equal three partitions',
        'resampling': 'paired station multiplicities jointly across partitions',
        'scope': 'source-validation development; conditional readout crossfit, not complete model OOF',
        'training_anchor_validation_distance': 'unadjusted descriptive Wasserstein distance; populations differ',
        'primary_not_selected_after_results': True, 'sparse_strata': 'fewer than20 stations descriptive; redraw counts retained'})
    lines = ['# Conditional station-held readout development', '', '| Model | MAE mg/L | Q90 MAE | log1p MAE |', '|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model_name} | {row.mae:.6f} | {row.q90_mae:.6f} | {row.log_mae:.6f} |')
    lines += ['', '| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |', '|---|---:|---:|']
    for row in effects[effects.region.eq('overall')].itertuples():
        lines.append(f'| {row.candidate} / {row.reference} | {row.relative_gain_pct:.3f} [{row.gain_ci_low_pct:.3f}, {row.gain_ci_high_pct:.3f}] | {row.improved_packages}/9 |')
    lines += ['', 'Only newly fitted scalar-readout losses are station-held. All retained representations and river inputs remain conditional source fits.',
        'Identical final inference anchor and state banks. This is development, not geographic/external confirmation.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    print(summary[['model_name', 'mae', 'q90_mae', 'log_mae']].to_string(index=False))
    print(effects[effects.region.eq('overall')][['candidate', 'reference', 'relative_gain_pct', 'gain_ci_low_pct', 'gain_ci_high_pct']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    analyze(args.root, args.bootstrap_draws)


if __name__ == '__main__':
    main()
