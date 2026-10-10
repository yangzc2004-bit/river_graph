"""Quantify upstream latent-state gains against BOTH complete and observed anchors."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_river_state_contrast_v1 import ARMS, BASE, OBSERVED, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json


def analyze(root, draws=5000):
    paths = sorted((root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in paths} != expected:
        raise ValueError('all nine packages required')
    verify_runtime_snapshot(root)
    panel, thresholds = load_panel(root)
    runs, parts, summary = metric_summary(panel, thresholds)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    for name, frame in (('run_metrics', runs), ('partition_metrics', parts), ('summary', summary)):
        frame.to_csv(out/f'{name}.csv', index=False)
    comparisons = [(arm, BASE) for arm in ARMS]
    comparisons += [('contrast_upstream', 'expanded_upstream'), ('contrast_upstream', OBSERVED),
        ('contrast_upstream', 'station_hidden_trees'), ('contrast_upstream', 'contrast_uniform'),
        ('contrast_matched_upstream', 'contrast_matched_nonupstream'),
        ('contrast_matched_upstream', 'expanded_matched_upstream'),
        ('contrast_matched_nonupstream', 'expanded_matched_nonupstream')]
    effects, strata, stations = [], [], []
    diagnostic = panel[panel.model_name.eq('contrast_upstream')].copy()
    diagnostic['coverage_group'] = np.select([
        diagnostic.state_support.eq(True) & diagnostic.previous_state_support.eq(False),
        diagnostic.state_support.eq(True) & diagnostic.previous_state_support.eq(True),
        diagnostic.state_support.eq(False) & diagnostic.previous_state_support.eq(True)],
        ['newly_supported', 'previously_supported', 'lost_support'], default='still_unsupported')
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
        if candidate != 'contrast_upstream' or reference not in (BASE, OBSERVED, 'expanded_upstream'):
            continue
        joined = pair.merge(diagnostic[['split_seed', 'seed', 'cell', 'support_group', 'distance_group',
            'state_support', 'observed_support', 'coverage_group']], on=['split_seed', 'seed', 'cell'], validate='one_to_one')
        groups = [('no_observed_DOC', joined[joined.observed_support.eq(False)]),
                  ('observed_DOC', joined[joined.observed_support.eq(True)]),
                  ('usable_state', joined[joined.state_support.eq(True)])]
        groups += [(str(name), group) for col in ('support_group', 'distance_group', 'coverage_group') for name, group in joined.groupby(col)]
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
    diagnostic.to_parquet(out/'query_diagnostics.parquet', index=False)
    fitting, coverage = [], []
    for path in paths:
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        with np.load(run/'fitting_inputs.npz', allow_pickle=False) as saved:
            supported = saved['validation_raw_valid'].any((1, 2))
            observed = saved['observed_support']
            coverage.append({'split_seed': config['split_seed'], 'seed': config['seed'],
                'observed_fraction': float(observed.mean()), 'state_fraction': float(supported.mean()),
                'previous_state_fraction': float(saved['previous_support'].mean()),
                'new_support_fraction': float((supported & ~saved['previous_support']).mean()),
                'lost_support_fraction': float((~supported & saved['previous_support']).mean()),
                'state_without_DOC_fraction': float((supported & ~observed).mean()),
                'neither_fraction': float((~supported & ~observed).mean()),
                'matched_state_fraction': float(saved['validation_matched_valid'].any((1, 2)).mean())})
        for arm in ARMS:
            state = json.loads((run/f'{arm}.json').read_text())
            fitting.append({'split_seed': config['split_seed'], 'seed': config['seed'], 'arm': arm,
                'best_epoch': state['best_epoch'], 'epochs_run': state['epochs_run'],
                'selected_zero': state['best_epoch'] == 0, 'parameters': state['trainable_parameters']})
    pd.DataFrame(coverage).to_csv(out/'source_support.csv', index=False)
    pd.DataFrame(fitting).to_csv(out/'fitting_summary.csv', index=False)
    diagnostic.groupby(['split_seed', 'seed']).agg(
        state_fraction=('state_support', 'mean'), prior_mass=('state_prior_mass', 'mean'),
        entropy=('state_entropy', 'mean'), abs_delta=('state_delta_native', lambda x: abs(x).mean()),
        lag0=('state_lag_mass_0', 'mean'), lag1=('state_lag_mass_1', 'mean'), lag3=('state_lag_mass_3', 'mean')
        ).reset_index().to_csv(out/'attention_summary.csv', index=False)
    write_json(out/'analysis_definition.json', {'bootstrap_draws': draws, 'primary': 'contrast_upstream',
        'estimand': 'seed mean cell loss within partition, then equal three partitions',
        'resampling': 'paired whole-station multiplicities jointly across partitions',
        'selection_role': 'same source-validation population; not independent confirmation',
        'comparisons': 'complete, observed-DOC and preceding absolute-state anchors',
        'hydro_pool': 'known-atlas environmental covariates; receiving fold/role excluded',
        'sparse_strata': 'fewer than 20 stations descriptive only; empty-partition draws are redrawn and counted',
        'latent_states_are_DOC_observations': False,
        'new_mechanism': 'source environmental state at allowed lag minus receiver current environmental state',
        'candidate_coverage_unchanged': True})
    lines = ['# Upstream-to-local environmental contrast development results', '', '| Model | MAE mg/L | Q90 MAE | log1p MAE |', '|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model_name} | {row.mae:.6f} | {row.q90_mae:.6f} | {row.log_mae:.6f} |')
    lines += ['', '| Candidate / reference | MAE reduction % [95% station CI] | Positive packages |', '|---|---:|---:|']
    for row in effects[effects.region.eq('overall')].itertuples():
        lines.append(f'| {row.candidate} / {row.reference} | {row.relative_gain_pct:.3f} [{row.gain_ci_low_pct:.3f}, {row.gain_ci_high_pct:.3f}] | {row.improved_packages}/9 |')
    lines += ['', 'Source-role development; complete and observed branch selection used these roles.',
              'All fixed arms retained. Latent environmental states are distinct from measured DOC.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    print(summary[['model_name', 'mae', 'q90_mae', 'log_mae']].to_string(index=False))
    print(effects[effects.region.eq('overall')][['candidate', 'reference', 'relative_gain_pct',
        'gain_ci_low_pct', 'gain_ci_high_pct']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    analyze(args.root, args.bootstrap_draws)


if __name__ == '__main__':
    main()
