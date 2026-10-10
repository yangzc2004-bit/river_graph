"""Estimate protected river-message gains over the complete current DOC model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_dynamic_river_v1 import ARMS, BASE, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json


def analyze(root, draws=5000):
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {path.parent.name for path in (root/'runs').glob('*/complete.json')} != expected:
        raise ValueError('all nine fixed development packages required')
    verify_runtime_snapshot(root)
    panel, thresholds = load_panel(root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    for name, frame in (('run_metrics', runs), ('partition_metrics', partitions), ('summary', summary)):
        frame.to_csv(out/f'{name}.csv', index=False)
    comparisons = [(arm, BASE) for arm in ARMS]
    comparisons += [('dynamic_lagged', name) for name in ('static_upstream', 'dynamic_same_month', 'station_hidden_trees')]
    comparisons += [('matched_upstream', 'matched_nonupstream'), ('matched_upstream', 'dynamic_lagged')]
    results, strata, stations = [], [], []
    diagnostics = panel[panel.model_name.eq('dynamic_lagged')].copy()
    age_rows, support_rows, fitting = [], [], []
    for path in sorted((root/'runs').glob('*/complete.json')):
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
            valid, features = saved['validation_raw_valid'], saved['validation_raw_features']
            age = np.min(np.where(valid, features[..., 32]*13., np.inf), axis=(1, 2))
            age_rows.append(pd.DataFrame({'split_seed': config['split_seed'], 'seed': config['seed'],
                'cell': saved['val_cells'], 'minimum_source_age': age}))
            support_rows.append({'split_seed': config['split_seed'], 'seed': config['seed'],
                'raw_supported_fraction': float(valid.any((1, 2)).mean()),
                'matched_supported_fraction': float(saved['validation_matched_valid'].any((1, 2)).mean()),
                'source_supported_fraction': float(saved['source_raw_valid'].any((1, 2)).mean())})
        for arm in ARMS:
            state = json.loads((run/f'{arm}.json').read_text())
            fitting.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'arm': arm,
                'best_epoch': state['best_epoch'], 'epochs_run': state['epochs_run'],
                'selected_zero': state['best_epoch'] == 0,
                'trainable_parameters_allocated': state['trainable_parameters']})
    diagnostics = diagnostics.merge(pd.concat(age_rows), on=['split_seed', 'seed', 'cell'], validate='one_to_one')
    diagnostics['age_group'] = np.select([diagnostics.minimum_source_age <= 1,
        diagnostics.minimum_source_age <= 6, np.isfinite(diagnostics.minimum_source_age)],
        ['age_0_1', 'age_2_6', 'age_7_12'], default='no_observed_source')
    diagnostics['distance_group'] = np.select([diagnostics.nearest_path_km.isna(),
        diagnostics.nearest_path_km <= 50, diagnostics.nearest_path_km <= 200],
        ['no_eligible_path', 'path_0_50_km', 'path_50_200_km'], default='path_over_200_km')
    for candidate, reference in comparisons:
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
            results.append({'candidate': candidate, 'reference': reference, 'region': region,
                **joint_station_bootstrap(selected, draws=draws),
                'improved_packages': int((direction.candidate_error < direction.reference_error).sum()),
                'improved_partitions': int((part.candidate_error < part.reference_error).sum()),
                'evaluation_role': 'selected_source_validation_development'})
        losses = pair.groupby(['split_seed', 'station', 'cell'], as_index=False)[['candidate_error', 'reference_error']].mean()
        station = losses.groupby(['split_seed', 'station'], as_index=False).agg(
            candidate_mae=('candidate_error', 'mean'), reference_mae=('reference_error', 'mean'), n_cells=('cell', 'size'))
        station['candidate'], station['reference'] = candidate, reference
        station['delta_mae'] = station.candidate_mae-station.reference_mae
        stations.append(station)
        if candidate != 'dynamic_lagged' or reference != BASE:
            continue
        joined = pair.merge(diagnostics[['split_seed', 'seed', 'cell', 'river_support', 'age_group', 'distance_group']],
                            on=['split_seed', 'seed', 'cell'], validate='one_to_one')
        groups = [('observed_upstream', joined[joined.river_support.eq(True)]),
                  ('no_observed_upstream', joined[joined.river_support.eq(False)])]
        groups += [(str(name), group) for column in ('age_group', 'distance_group') for name, group in joined.groupby(column)]
        for name, group in groups:
            if not group.empty:
                strata.append({'candidate': candidate, 'reference': reference, 'stratum': name,
                               **joint_station_bootstrap(group, draws=draws)})
    effects, station_effects = pd.DataFrame(results), pd.concat(stations, ignore_index=True)
    effects.to_csv(out/'paired_effects.csv', index=False)
    pd.DataFrame(strata).to_csv(out/'stratified_effects.csv', index=False)
    station_effects.to_csv(out/'station_effects.csv', index=False)
    diagnostics.to_parquet(out/'query_diagnostics.parquet', index=False)
    pd.DataFrame(support_rows).to_csv(out/'source_support.csv', index=False)
    pd.DataFrame(fitting).to_csv(out/'fitting_summary.csv', index=False)
    per_station = station_effects.groupby(['candidate', 'reference', 'split_seed'])[['candidate_mae', 'reference_mae']].mean()
    per_station.groupby(['candidate', 'reference']).mean().reset_index().to_csv(out/'station_equal.csv', index=False)
    diag_summary = diagnostics.groupby(['split_seed', 'seed']).agg(
        support_fraction=('river_support', 'mean'), prior_mass=('river_prior_mass', 'mean'),
        entropy=('river_entropy', 'mean'), absolute_delta_mgL=('river_delta_native', lambda x: abs(x).mean()),
        lag_0=('river_lag_mass_0', 'mean'), lag_1=('river_lag_mass_1', 'mean'), lag_3=('river_lag_mass_3', 'mean'))
    diag_summary.reset_index().to_csv(out/'attention_summary.csv', index=False)
    primary = effects[effects.candidate.eq('dynamic_lagged') & effects.reference.eq(BASE) & effects.region.eq('overall')].iloc[0]
    write_json(out/'analysis_definition.json', {'bootstrap_draws': draws, 'units': 'native mg/L except log1p rows',
        'estimand': 'seed mean loss within partition, then equal three partitions',
        'resampling': 'same whole-station multiplicities jointly across partitions',
        'selection_optimism': 'base and branch selected on this source-validation population',
        'primary_model': 'dynamic_lagged', 'primary_gain_pct': float(primary.relative_gain_pct),
        'no_external_or_geographical_confirmation': True})
    lines = ['# Dynamic river development results', '', '## Complete fixed comparison', '',
             '| Model | MAE mg/L | Q90 MAE | log1p MAE |', '|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model_name} | {row.mae:.6f} | {row.q90_mae:.6f} | {row.log_mae:.6f} |')
    lines += ['', '## Paired native-MAE contrasts', '', '| Candidate / reference | Gain % [95% station CI] | Positive packages |', '|---|---:|---:|']
    for row in effects[effects.region.eq('overall')].itertuples():
        lines.append(f'| {row.candidate} / {row.reference} | {row.relative_gain_pct:.3f} [{row.gain_ci_low_pct:.3f}, {row.gain_ci_high_pct:.3f}] | {row.improved_packages}/9 |')
    lines += ['', 'Development results on retained ST357 source roles; not independent confirmation.',
              'All five fitted arms are retained, including zero-selected and negative results.']
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
