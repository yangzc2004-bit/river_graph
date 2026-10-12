"""Matched H2X postprocessing: paired scores, intervals and reviewable tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_doc_h2x_matched_v1 as study
import torch
from report_doc_dynamic_reconstruction_v1 import (
    METRICS,
    metrics_from_statistics,
    sufficient_statistics,
)

from river_graph.experiments.provenance import sha256_file

NAMES = {'h2x_matched': 'H2X: matched-input adaptation',
         'monthly_log': 'RiverDOCNet: original three-part fusion',
         'hybrid_direct': 'RiverDOCNet: explicit-history/direct-readout fusion',
         'rf_monthly': 'History random forest', 'xgb_monthly': 'History XGBoost'}
ORDER = ('h2x_matched', 'monthly_log', 'hybrid_direct', 'rf_monthly', 'xgb_monthly')
REFERENCES = ('monthly_log', 'hybrid_direct', 'rf_monthly', 'xgb_monthly')
ERRORS = ('mae', 'log1p_mae', 'rmse', 'monthly_delta_mae')


def table(frame, columns):
    rows = ['| Model | ' + ' | '.join(columns) + ' |',
            '| --- | ' + ' | '.join(['---'] * len(columns)) + ' |']
    values = frame.set_index('model_name')
    for arm in ORDER:
        cells = []
        for metric in columns:
            number = values.loc[arm, metric]
            best = values[metric].min() if metric in ERRORS else values[metric].max()
            value = f'{number:.4f}'
            cells.append('**' + value + '**' if number == best and metric != 'within_station_sd_ratio' else value)
        rows.append('| ' + NAMES[arm] + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def bootstrap(frame, by_seed, track, basins, gaps):
    rng = np.random.default_rng(20261012)
    draws = rng.multinomial(len(basins), np.repeat(1 / len(basins), len(basins)), size=10000)
    stat_rows = [{'gap_months': int(gap), 'seed': int(seed), 'model_name': arm, 'cluster': cluster,
                  **sufficient_statistics(part)} for (gap, seed, arm, cluster), part in
                 frame.groupby(['gap_months', 'seed', 'model_name', 'cluster'])]
    stats = pd.DataFrame(stat_rows)
    columns = [c for c in stats if c not in ('gap_months', 'seed', 'model_name', 'cluster')]
    distributions = {}
    records = []
    for gap in gaps:
        estimated = {}
        for arm in study.ARMS:
            repetitions = []
            for seed in (42, 43, 44):
                part = stats[(stats.gap_months == gap) & (stats.seed == seed) & (stats.model_name == arm)]
                matrix = part.set_index('cluster')[columns].reindex(basins, fill_value=0)
                totals = draws @ matrix.to_numpy(float)
                repetitions.append(metrics_from_statistics({key: totals[:, i] for i, key in enumerate(columns)}))
                exact = metrics_from_statistics(matrix.sum().to_dict())
                reference = by_seed[(by_seed.track == track) & (by_seed.gap_months == gap) &
                                    (by_seed.seed == seed) & (by_seed.model_name == arm)].iloc[0]
                for metric in METRICS:
                    assert abs(exact[metric] - reference[metric]) < 1e-6, (track, gap, seed, arm, metric)
            estimated[arm] = {m: np.mean([r[m] for r in repetitions], axis=0) for m in METRICS}
        distributions[gap] = estimated
    if track == 'time':
        distributions[0] = {arm: {metric: np.mean([distributions[g][arm][metric] for g in gaps], axis=0)
                                         for metric in METRICS} for arm in study.ARMS}
    for gap, estimates in distributions.items():
        points = by_seed[(by_seed.track == track) & ((by_seed.gap_months == gap) if gap else True)]
        point = points.groupby('model_name').mean(numeric_only=True)
        for reference in REFERENCES:
            for metric in METRICS:
                delta = estimates[study.NEW_ARM][metric] - estimates[reference][metric]
                assert np.isfinite(delta).all(), (track, gap, reference, metric)
                lower, upper = np.quantile(delta, [.025, .975])
                records.append({'track': track, 'gap_months': gap, 'reference': reference,
                                'metric': metric, 'difference': point.loc[study.NEW_ARM, metric]-point.loc[reference, metric],
                                'lower_95': lower, 'upper_95': upper, 'clusters': len(basins), 'draws': len(draws)})
    return stats, pd.DataFrame(records)


def main():
    root, out = study.ROOT, study.ROOT / 'analysis'
    assert (root / 'batch_complete.json').exists()
    audit = json.loads((out / 'independent_audit.json').read_text())
    assert audit['passed']
    protocol = json.loads((root / 'protocol.json').read_text())
    by_seed = pd.read_csv(out / 'by_seed.csv')
    summary = pd.read_csv(out / 'summary.csv')
    frames = pd.concat([pd.read_parquet(study.directory(root, t) / 'predictions.parquet')
                        for t in protocol['tasks']], ignore_index=True)
    space = frames[frames.track == 'space'].copy()
    space['cluster'] = space.target_huc4.astype(str)
    ss, si = bootstrap(space, by_seed, 'space', sorted(space.cluster.unique()), [0])
    nodes = pd.read_csv(protocol['nodes'], dtype={'site_no': str, 'huc_cd': str})
    assert sha256_file(protocol['nodes']) == protocol['nodes_sha256']
    assert nodes.huc_cd.str.fullmatch(r'\d+').all()
    width = nodes.huc_cd.str.len().map({7: 8, 8: 8, 11: 12, 12: 12})
    assert width.notna().all()
    nodes['huc4'] = [h.zfill(int(w))[:4] for h, w in zip(nodes.huc_cd, width)]
    data = torch.load(protocol['dataset'], weights_only=False, map_location='cpu')
    mapping = dict(enumerate(nodes.set_index('site_no').huc4.loc[list(data['site_no'])].tolist()))
    temporal = frames[frames.track == 'time'].copy()
    temporal['cluster'] = temporal.station.map(mapping)
    assert temporal.cluster.notna().all()
    ts, ti = bootstrap(temporal, by_seed, 'time', sorted(temporal.cluster.unique()), [3, 6, 12])
    ss.to_csv(out / 'spatial_sufficient_statistics.csv', index=False)
    ts.to_csv(out / 'temporal_sufficient_statistics.csv', index=False)
    intervals = pd.concat([si, ti], ignore_index=True)
    intervals.to_csv(out / 'paired_bootstrap.csv', index=False)
    paired = []
    for (track, gap, seed), part in by_seed.groupby(['track', 'gap_months', 'seed']):
        p = part.set_index('model_name')
        for reference in REFERENCES:
            for metric in METRICS:
                paired.append({'track': track, 'gap_months': gap, 'seed': seed,
                               'reference': reference, 'metric': metric,
                               'difference': p.loc[study.NEW_ARM, metric]-p.loc[reference, metric]})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / 'paired_effects_by_seed.csv', index=False)
    counts = paired.groupby(['track', 'gap_months', 'reference', 'metric'], as_index=False).agg(
        difference=('difference', 'mean'), negative_repetitions=('difference', lambda x: int((x < 0).sum())),
        positive_repetitions=('difference', lambda x: int((x > 0).sum())), repetitions=('difference', 'size'))
    counts.to_csv(out / 'paired_effects.csv', index=False)
    time = by_seed[by_seed.track == 'time'].groupby('model_name', as_index=False)[list(METRICS)].mean()
    time['aggregation'] = 'equal mean of three gap lengths and three fixed repetitions'
    time.to_csv(out / 'temporal_mean_metrics.csv', index=False)
    concentration = ['mae', 'log1p_mae', 'rmse', 'r2']
    dynamics = ['station_centered_r2', 'log1p_station_centered_r2',
                'monthly_delta_mae', 'monthly_delta_correlation']
    spatial = summary[summary.track == 'space']
    time_values, spatial_values = time.set_index('model_name'), spatial.set_index('model_name')
    changes = []
    for track, values in [('time', time_values), ('space', spatial_values)]:
        for reference in REFERENCES:
            row = {'track': track, 'reference': reference}
            for m in ('mae', 'log1p_mae', 'rmse', 'monthly_delta_mae'):
                row[m + '_percent_error_reduction'] = 100*(1-values.loc[study.NEW_ARM, m]/values.loc[reference, m])
            changes.append(row)
    facts = {'model_config': protocol['model_config'], 'parameters': protocol['parameters'],
             'changes': changes, 'time_metrics': time.to_dict('records'),
             'space_metrics': spatial[['model_name', *METRICS]].to_dict('records'),
             'spatial_huc4_clusters': space.cluster.nunique(),
             'temporal_huc4_clusters': temporal.cluster.nunique(),
             'training_epochs': [json.loads((study.directory(root, t) / 'info.json').read_text())['epochs_run']
                                 for t in protocol['tasks'] if t['arm'] == study.NEW_ARM],
             'space_queries_per_seed': int(space[space.seed == 42].query('model_name == "h2x_matched"').shape[0]),
             'space_stations': int(space.station.nunique()),
             'time_stations': int(temporal.station.nunique())}
    (out / 'interpretation_inputs.json').write_text(json.dumps(facts, indent=2)+'\n')
    lines = ['# Historical H2X under matched current DOC inputs', '',
             ('All 24 new H2X fits and the 96 immutable current-model/control packages completed. '
             'All 120 selected checkpoints were replayed independently on validation and test '
             'queries. The masks, query truth, normalization and local information budget are matched.'), '',
             '## Important identity of the H2X row', '',
             ('This is **H2X matched-input adaptation**, not a literal reproduction of the historical experiment. '
             'The released environmental encoder and two transport GCN layers are retained unmodified '
             '(hidden 64, environmental embedding 32). The current twelve-month driver history is flattened. '
             'Observed DOC inputs are absent; static edge gates read only the existing environmental '
             'features of their two endpoints instead of the old additional NHD edge attributes. '
             'No GRU, Transformer, new measurements or post-test tuning is introduced.'), '',
             (f"H2X has {protocol['parameters'][study.NEW_ARM]:,} parameters, versus "
             f"{protocol['parameters']['monthly_log']:,} for original fusion and "
             f"{protocol['parameters']['hybrid_direct']:,} for the explicit-history update. "
             'The optimizer, log loss, monthly batch schedule and 60-epoch maximum match the current fusion protocol. '
             'Native validation MAE alone chooses checkpoints. This tests the adapted architecture and capacity '
             'under the fixed information/budget contract, not one isolated component.'), '',
             '## Temporal reconstruction', '', table(time, concentration), '', table(time, dynamics), '',
             ('Overview scores are the equal mean across 3/6/12-month gaps and three fixed repetitions. '
             'The gap lengths reuse the same scored cores, so there are not nine independent ecological repetitions.'), '']
    for gap in (3, 6, 12):
        lines.extend([f'## {gap}-month gaps', '', table(summary[(summary.track == 'time') &
                      (summary.gap_months == gap)], concentration), ''])
    lines.extend(['## Spatial reconstruction', '', table(spatial, concentration), '',
                  ('Spatial scores pool 7,262 observed queries at 104 fully held-out DOC stations '
                  'across five HUC4 regions within seed, then average three seeds. All models have '
                  'complete coverage on these queries.'), '',
                  '## Point-estimate conclusions', ''])
    for track, values in [('Temporal', time_values), ('Spatial', spatial_values)]:
        lines.append(f"{track} native-MAE ranking: " + ' < '.join(NAMES[a] for a in values.mae.sort_values().index) + '.')
    lines.extend(['',
                  ('Under the predeclared native/log-MAE primary endpoints, matched-input H2X does '
                  'not displace the current fusion architecture or solve its temporal gap to the '
                  'tree baselines. It improves temporal point MAE relative to original fusion, '
                  'but has higher overall primary errors than the explicit-history update and both trees. '
                  'It has lower native/log-MAE than the update at three and six months, but higher '
                  'errors at twelve months. This is not a uniform advantage over gap lengths.'), '',
                  ('H2X has the lowest spatial RMSE and highest predictive R-squared in this five-model '
                  'comparison, while original fusion retains the lowest spatial native/log-MAE. '
                  'This metric tradeoff does not authorize switching the primary endpoint after '
                  'seeing results. Overall H2X-versus-fusion primary difference intervals include '
                  'zero in both tasks; H2X-versus-random-forest spatial primary intervals are '
                  'below zero. Five spatial clusters and repeated development use limit generalization.'), '',
                  ('H2X temporal monthly-change MAE is 2.1839 mg/L, close to the zero-change '
                  'reference (2.1840), and higher than original fusion (2.1666), random forest '
                  '(2.1328) and XGBoost (2.1459). Its native station-centered R-squared is '
                  '0.0368, versus 0.0666 for XGBoost. Lower concentration error relative to '
                  'original fusion therefore does not establish a comprehensive improvement '
                  'in temporal change reconstruction.'), ''])
    for row in changes:
        lines.append(f"- {row['track']}, H2X versus {NAMES[row['reference']]}: native MAE error reduction "
                     f"{row['mae_percent_error_reduction']:.2f}%; log-MAE error reduction "
                     f"{row['log1p_mae_percent_error_reduction']:.2f}% (negative means deterioration).")
    lines.extend(['', '## Paired uncertainty', '',
                  '| Task | Reference | Metric | H2X minus reference | Lower 95% | Upper 95% |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for r in intervals[(intervals.gap_months == 0) & intervals.metric.isin(['mae', 'log1p_mae'])].itertuples():
        lines.append(f'| {r.track} | {NAMES[r.reference]} | {r.metric} | {r.difference:.5f} | {r.lower_95:.5f} | {r.upper_95:.5f} |')
    lines.extend(['', ('Intervals use 10,000 paired HUC4-block resamples with the same weights across models, '
                  'seeds and gap lengths. Spatial inference has five clusters; temporal inference has '
                  f'{temporal.cluster.nunique()} clusters. These are exploratory, unadjusted comparisons on '
                  'repeatedly examined development tasks. An interval spanning zero does not establish a stable '
                  'population lead, and a point-score win is not global SOTA.'), '',
                  ('The input adaptations, cohort, masking and training schedule prevent transferring the '
                  'old mixed-571 ranking directly. This experiment cannot identify which historical '
                  'protocol/input change caused any ranking reversal.'), '',
                  ('The target is the mean of available DOC samples within each calendar month. '
                  'Observed-query evaluation does not validate natural missing-cell truth, continuous '
                  'daily DOC, arbitrary river reaches or cross-river use.'), '',
                  ('Evidence: protocol.json, manifest.json, per-fit prediction sidecars, summary.csv, '
                  'by_seed.csv, paired_bootstrap.csv and independent_audit.json in '
                  'experiments/phase4_transfer/doc_h2x_matched_v1. All previous results are unchanged.')])
    report = Path(protocol['repository']) / 'docs/doc_h2x_matched_results_20261012.md'
    report.write_text('\n'.join(lines)+'\n')
    outputs = ['spatial_sufficient_statistics.csv', 'temporal_sufficient_statistics.csv',
               'paired_bootstrap.csv', 'paired_effects_by_seed.csv', 'paired_effects.csv',
               'temporal_mean_metrics.csv', 'interpretation_inputs.json']
    (out / 'report_provenance.json').write_text(json.dumps({
        'script_sha256': sha256_file(__file__),
        'statistics_helper_sha256': sha256_file('scripts/report_doc_dynamic_reconstruction_v1.py'),
        'summary_sha256': sha256_file(out / 'summary.csv'),
        'audit_sha256': sha256_file(out / 'independent_audit.json'),
        'report': str(report), 'report_sha256': sha256_file(report),
        'outputs': {n: sha256_file(out / n) for n in outputs}, 'model_selection_modified': False}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'report': str(report), 'changes': changes}))


if __name__ == '__main__':
    main()

