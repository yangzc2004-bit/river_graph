"""Separate river connectivity, path conditioning and joint-refit effects."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_river_structure_residual_v1 import ARMS, ATLAS, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json

from river_graph.experiments.provenance import sha256_file

PROFILES = ('low_order', 'chain', 'confluence', 'mainstem', 'storage')


def analyze(root, draws=5000, available_only=False):
    completions = sorted((root/'runs').glob('*/complete.json'))
    expected = {f'split{s}_seed{seed}' for s in (142, 143, 144) for seed in (42, 43, 44)}
    if not completions or (not available_only and {p.parent.name for p in completions} != expected):
        raise ValueError('complete nine packages before the final analysis')
    verify_runtime_snapshot(root)
    panel, thresholds = load_panel(root)
    if not np.isfinite(panel[['y_true', 'y_pred']]).all().all() or (panel.y_pred < 0).any():
        raise ValueError('finite nonnegative DOC predictions required')
    runs, partitions, summary = metric_summary(panel, thresholds)
    # The shared summary helper assumes three complete splits. Partial package
    # inspection explicitly uses only its observed source-development scope.
    for metric in ('q90_mae', 'r2', 'log_r2'):
        means = partitions.groupby(['model_name', 'k'])[metric].mean()
        counts = partitions.groupby(['model_name', 'k'])[metric].count()
        for index, row in summary.iterrows():
            key = (row.model_name, row.k)
            summary.loc[index, metric] = means[key] if counts[key] == panel.split_seed.nunique() else np.nan
    out = root/('partial_analysis' if available_only else 'analysis')
    out.mkdir(exist_ok=True)
    for name, frame in (('run_metrics', runs), ('partition_metrics', partitions), ('summary', summary)):
        frame.to_csv(out/f'{name}.csv', index=False)
    comparisons = [(arm, reference) for arm in ARMS for reference in
                   ('available_real_integrated', 'station_hidden_trees')]
    comparisons += [(arm, 'river_none') for arm in ('river_simple', 'river_structure', 'river_rewired')]
    comparisons += [('river_structure', 'river_simple'), ('river_structure', 'river_rewired')]
    comparisons += [(arm, f'{arm}_zero_river') for arm in ARMS if arm != 'river_none']
    descriptors = panel[panel.model_name.eq('river_structure')][['split_seed', 'seed', 'station', 'cell', 'river_support']].copy()
    descriptors['upstream_availability'] = np.where(descriptors.river_support > 0, 'observed_upstream', 'no_observed_upstream')
    path_rows, availability_rows = [], []
    for completion in completions:
        run = completion.parent
        config = json.loads((run/'config.json').read_text())
        with np.load(config['mask_path'], allow_pickle=False) as mask:
            val = mask['val'].copy()
        with np.load(run/'river_inputs.npz', allow_pickle=False) as saved:
            real_valid = saved['real_validation_river_valid']
            t = real_valid.shape[1]
            index = np.searchsorted(np.unique(val//t), val//t), val % t
            owner = saved['real_validation_river_owner']
            lengths = np.expm1(saved['real_validation_river_path'][..., 0]*np.log1p(3000.))
            nearest = np.min(np.where(owner >= 0, lengths, np.inf), axis=1)
            nearest[~np.isfinite(nearest)] = np.nan
            counts = real_valid[index].any(-1).sum(-1)
            path_rows.append(pd.DataFrame({'split_seed': config['split_seed'], 'seed': config['seed'],
                'cell': val, 'river_unique_sources': counts,
                'rewired_unique_sources': saved['rewired_validation_river_valid'][index].any(-1).sum(-1),
                'nearest_path_km': nearest[index[0]]}))
            for bank in ('real', 'rewired'):
                valid = saved[f'{bank}_validation_river_valid'][index]
                unique_sources = valid.any(-1).sum(-1)
                availability_rows.append({'split_seed': config['split_seed'], 'seed': config['seed'],
                    'bank': bank, 'mean_unique_observed_sources': float(unique_sources.mean()),
                    'mean_valid_source_lag_candidates': float(valid.sum((1, 2)).mean()),
                    'supported_cell_fraction': float((unique_sources > 0).mean()),
                    'eligible_path_stations': int((owner >= 0).any(-1).sum())})
    descriptors = descriptors.merge(pd.concat(path_rows), on=['split_seed', 'seed', 'cell'], validate='one_to_one')
    descriptors['path_distance'] = np.select([descriptors.nearest_path_km.isna(),
        descriptors.nearest_path_km <= 50, descriptors.nearest_path_km <= 200],
        ['no_eligible_path', 'path_0_to_50_km', 'path_50_to_200_km'], default='path_over_200_km')
    pd.DataFrame(availability_rows).to_csv(out/'source_availability.csv', index=False)
    descriptors.to_csv(out/'query_path_support.csv', index=False)
    physical = pd.read_csv(ATLAS/'station_structure.csv', dtype={'station': str})
    effects, strata, station_effects = [], [], []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ('overall', 'q90'):
            selected = pair
            if region == 'q90':
                q = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate >= q]
            if selected.empty:
                continue
            direction = selected.groupby(['split_seed', 'seed'])[['candidate_error', 'reference_error']].mean()
            part_direction = direction.groupby('split_seed').mean()
            effects.append({'candidate': candidate, 'reference': reference, 'region': region,
                **joint_station_bootstrap(selected, draws=draws),
                'improved_packages': int((direction.candidate_error < direction.reference_error).sum()),
                'improved_partitions': int((part_direction.candidate_error < part_direction.reference_error).sum()),
                'expected_splits': panel.split_seed.nunique()})
        cells = pair.groupby(['split_seed', 'station', 'cell'], as_index=False)[['candidate_error', 'reference_error']].mean()
        stations = cells.groupby(['split_seed', 'station'], as_index=False).agg(
            candidate_mae=('candidate_error', 'mean'), reference_mae=('reference_error', 'mean'), n_cells=('cell', 'size'))
        stations['candidate'], stations['reference'] = candidate, reference
        stations['delta_mae'] = stations.candidate_mae-stations.reference_mae
        station_effects.append(stations)
        if candidate != 'river_structure' or reference not in ('available_real_integrated', 'river_none', 'river_rewired', 'river_simple'):
            continue
        joined = pair.merge(descriptors, on=['split_seed', 'seed', 'station', 'cell'], validate='one_to_one')\
            .merge(physical[['station', *PROFILES]], on='station', validate='many_to_one')
        groups = [(name, joined[joined[name]]) for name in PROFILES]
        groups += [(name, joined[joined.upstream_availability.eq(name)])
                   for name in ('observed_upstream', 'no_observed_upstream')]
        groups += [(name, joined[joined.path_distance.eq(name)]) for name in joined.path_distance.unique()]
        both = (joined.river_unique_sources > 0) & (joined.rewired_unique_sources > 0)
        groups += [('both_banks_observed', joined[both]),
                   ('near_paths_both_banks_observed', joined[both & (joined.nearest_path_km <= 50)])]
        for group_name, selected in groups:
            if selected.empty:
                continue
            estimate = joint_station_bootstrap(selected, draws=draws)
            strata.append({'candidate': candidate, 'reference': reference, 'stratum': group_name,
                'interpretation': 'overlapping physical profile' if group_name in PROFILES else 'source support or fixed distance band',
                'small_station_sample': selected.station.nunique() < 8, 'expected_splits': panel.split_seed.nunique(), **estimate})
    pd.DataFrame(effects).to_csv(out/'paired_effects.csv', index=False)
    refit = paired(panel, 'river_none', 'available_real_integrated')
    refit['prediction_difference'] = np.abs(refit.y_pred_candidate-refit.y_pred_reference)
    refit.groupby(['split_seed', 'seed'], as_index=False).agg(
        mean_absolute_prediction_difference=('prediction_difference', 'mean'),
        max_absolute_prediction_difference=('prediction_difference', 'max'),
        fixed_reference_mae=('reference_error', 'mean'), matched_refit_mae=('candidate_error', 'mean')
    ).to_csv(out/'refit_reproducibility.csv', index=False)
    pd.DataFrame(strata).to_csv(out/'structure_support_effects.csv', index=False)
    pd.concat(station_effects).to_csv(out/'station_effects.csv', index=False)
    panel['absolute_error'] = np.abs(panel.y_pred-panel.y_true)
    panel['signed_error'] = panel.y_pred-panel.y_true
    station = panel.groupby(['split_seed', 'seed', 'model_name', 'station'], as_index=False).agg(
        mae=('absolute_error', 'mean'), bias=('signed_error', 'mean'), n_cells=('cell', 'size'))
    station.to_csv(out/'station_metrics.csv', index=False)
    station.groupby(['split_seed', 'seed', 'model_name'])[['mae', 'bias']].mean()\
        .groupby(['split_seed', 'model_name']).mean().groupby('model_name').mean()\
        .rename(columns={'mae': 'station_equal_mae', 'bias': 'station_equal_bias'}).to_csv(out/'station_equal_summary.csv')
    diagnostic = panel[panel.model_name.isin(ARMS)].copy()
    diagnostic['river_available'] = diagnostic.river_support > 0
    diagnostic['absolute_river_correction'] = np.abs(diagnostic.river_delta_effective)
    columns = ['river_support', 'river_available', 'river_prior_mass', 'river_entropy', 'absolute_river_correction',
               'river_lag_mass_0', 'river_lag_mass_1', 'river_lag_mass_3']
    diagnostic.groupby(['split_seed', 'seed', 'model_name'])[columns].mean().reset_index().to_csv(out/'river_diagnostics.csv', index=False)
    config_hashes = {p.parent.name: sha256_file(p) for p in completions}
    write_json(out/'sources.json', {'evaluation_role': 'source_validation_development', 'partial': available_only,
        'bootstrap_draws': draws,
        'support_units': 'river_support counts valid source-lag candidates; river_unique_sources counts distinct sources',
        'distance_bands_km': [0, 50, 200, 'over200', 'no eligible path'],
        'common_support_diagnostic': 'added after first full readout; predictor-only intersection of real and rewired banks with >=1 observed source', 'aggregation': 'seed mean per cell, then equal split means; shared station bootstrap',
        'analyzer_sha256': sha256_file(__file__), 'completed_packages': config_hashes,
        'csvs': {p.name: sha256_file(p) for p in out.glob('*.csv')}})
    lines = ['# River structure development results', '',
        'DOC reconstruction at receiving stations without water-quality inputs. These are source-validation',
        'development results; geographical and external confirmation products remain unchanged.', '',
        f'Completed packages: {len(completions)}. Partial report: {available_only}.', '',
        '## Reconstruction error', '', '| Model | MAE mg/L | Q90 MAE |', '|---|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.model_name} | {row.mae:.6f} | {row.q90_mae:.6f} |')
    lines += ['', '## River comparisons', '', '| Candidate vs reference | MAE reduction % and 95% interval |', '|---|---:|']
    for row in pd.DataFrame(effects).query("region == 'overall'").itertuples():
        lines.append(f'| {row.candidate} vs {row.reference} | {row.relative_gain_pct:.3f} [{row.gain_ci_low_pct:.3f}, {row.gain_ci_high_pct:.3f}] |')
    lines += ['', 'The matched no-river arm separates joint retraining from added connectivity.',
        'The fitted zero-river product isolates the direct message contribution within each fitted network.',
        'Rewiring preserves candidate slots and matches source drainage area; actual observation availability',
        'can differ and is reported. Attention weights are predictive allocation, not physical transport rates.',
        'Structure profiles overlap and subgroup comparisons remain exploratory.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')
    print(summary[['model_name', 'mae', 'q90_mae']].to_string(index=False), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    analyze(args.root, args.bootstrap_draws, args.available_only)


if __name__ == '__main__':
    main()
