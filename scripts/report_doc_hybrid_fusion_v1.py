"""Post-audit paired uncertainty and result tables; no fitting or selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_doc_hybrid_fusion_v1 as study
import torch
from report_doc_dynamic_reconstruction_v1 import (
    METRICS,
    metrics_from_statistics,
    sufficient_statistics,
)

from river_graph.experiments.provenance import sha256_file

NAMES = {'monthly_log': 'RiverDOCNet: original fusion',
         'hybrid_direct': 'RiverDOCNet: explicit lag and direct readout',
         'rf_monthly': 'History random forest', 'xgb_monthly': 'History XGBoost'}
ERRORS = ('mae', 'log1p_mae', 'rmse', 'monthly_delta_mae')
REFERENCES = ('monthly_log', 'rf_monthly', 'xgb_monthly')


def table(frame, columns):
    rows = ['| Model | ' + ' | '.join(columns) + ' |', '| --- | ' + ' | '.join(['---'] * len(columns)) + ' |']
    f = frame.set_index('model_name')
    for arm in study.ARMS:
        values = []
        for metric in columns:
            value = f.loc[arm, metric]
            best = f[metric].min() if metric in ERRORS else f[metric].max()
            formatted = f'{value:.4f}'
            values.append('**' + formatted + '**' if value == best and metric != 'within_station_sd_ratio' else formatted)
        rows.append('| ' + NAMES[arm] + ' | ' + ' | '.join(values) + ' |')
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


def plot_effects(intervals, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = intervals[(intervals.gap_months == 0) & (intervals.reference == 'monthly_log')]
    panels = [('mae', 'MAE difference (mg/L)\nLower is better'),
              ('log1p_station_centered_r2', 'Centered log-R² difference\nHigher is better')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for ax, (metric, title) in zip(axes, panels):
        p = selected[selected.metric == metric].set_index('track').loc[['time', 'space']]
        for i, (_, row) in enumerate(p.iterrows()):
            ax.plot([row.lower_95, row.upper_95], [i, i], color='#276b9c', linewidth=2)
            ax.scatter([row.difference], [i], color='#276b9c', zorder=3)
        ax.axvline(0, color='0.4', linewidth=1, linestyle='--')
        ax.set_yticks([0, 1], ['Temporal gap average', 'Spatial queries'])
        ax.set_xlabel(title)
        ax.set_ylim(-.55, 1.55)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Explicit lag/direct readout versus original fusion\nExploratory 95% paired intervals')
    fig.savefig(output / 'paired_architecture_effects.png', dpi=180)
    fig.savefig(output / 'paired_architecture_effects.pdf')
    plt.close(fig)


def main():
    root = study.ROOT
    out = root / 'analysis'
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
    spatial_stats, spatial_intervals = bootstrap(space, by_seed, 'space', sorted(space.cluster.unique()), [0])
    nodes = pd.read_csv(protocol['nodes'], dtype={'site_no': str, 'huc_cd': str})
    assert sha256_file(protocol['nodes']) == protocol['nodes_sha256']
    assert nodes.huc_cd.str.fullmatch(r'\d+').all()
    width = nodes.huc_cd.str.len().map({7: 8, 8: 8, 11: 12, 12: 12})
    assert width.notna().all()
    nodes['huc4'] = [h.zfill(int(w))[:4] for h, w in zip(nodes.huc_cd, width)]
    data = torch.load(protocol['dataset'], weights_only=False, map_location='cpu')
    mapping = dict(enumerate(nodes.set_index('site_no').huc4.loc[list(data['site_no'])].tolist()))
    time = frames[frames.track == 'time'].copy()
    time['cluster'] = time.station.map(mapping)
    assert time.cluster.notna().all()
    temporal_stats, temporal_intervals = bootstrap(time, by_seed, 'time', sorted(time.cluster.unique()), [3, 6, 12])
    spatial_stats.to_csv(out / 'spatial_sufficient_statistics.csv', index=False)
    temporal_stats.to_csv(out / 'temporal_sufficient_statistics.csv', index=False)
    intervals = pd.concat([spatial_intervals, temporal_intervals], ignore_index=True)
    intervals.to_csv(out / 'paired_bootstrap.csv', index=False)
    paired = []
    for (track, gap, seed), part in by_seed.groupby(['track', 'gap_months', 'seed']):
        a = part.set_index('model_name')
        for reference in REFERENCES:
            for metric in METRICS:
                paired.append({'track': track, 'gap_months': gap, 'seed': seed,
                               'reference': reference, 'metric': metric,
                               'difference': a.loc[study.NEW_ARM, metric]-a.loc[reference, metric]})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / 'paired_effects_by_seed.csv', index=False)
    counts = paired.groupby(['track', 'gap_months', 'reference', 'metric'], as_index=False).agg(
        difference=('difference', 'mean'), negative_repetitions=('difference', lambda x: int((x < 0).sum())),
        positive_repetitions=('difference', lambda x: int((x > 0).sum())), repetitions=('difference', 'size'))
    counts.to_csv(out / 'paired_effects.csv', index=False)
    plot_effects(intervals, out)
    temporal = pd.read_csv(out / 'temporal_overview.csv')
    mean_metrics = temporal[['model_name', *METRICS]].copy()
    mean_metrics['aggregation'] = 'equal mean of 3/6/12-month gap metrics and three fixed repetitions'
    mean_metrics.to_csv(out / 'temporal_mean_metrics.csv', index=False)
    original, updated = temporal.set_index('model_name').loc[['monthly_log', study.NEW_ARM]].to_dict('records')
    spatial = summary[summary.track == 'space'].set_index('model_name')
    changes = {metric: {'temporal_percent_error_reduction': 100*(1-updated[metric]/original[metric]),
                       'spatial_percent_error_reduction': 100*(1-spatial.loc[study.NEW_ARM, metric]/spatial.loc['monthly_log', metric])}
               for metric in ('mae', 'log1p_mae', 'rmse', 'monthly_delta_mae')}
    facts = {'fixed_architecture': protocol['model_config'], 'changes_vs_original': changes,
             'parameters': protocol['parameters'], 'temporal_huc4_clusters': time.cluster.nunique(),
             'time_overview': mean_metrics.to_dict('records'),
             'spatial_overview': spatial.reset_index()[['model_name', *METRICS]].to_dict('records'),
             'new_fit_epochs': [json.loads((study.directory(root, t) / 'info.json').read_text())['epochs_run']
                                for t in protocol['tasks'] if t['arm'] == study.NEW_ARM]}
    (out / 'interpretation_inputs.json').write_text(json.dumps(facts, indent=2)+'\n')
    concentration = ['mae', 'log1p_mae', 'rmse', 'r2']
    dynamics = ['station_centered_r2', 'log1p_station_centered_r2', 'within_station_sd_ratio',
                'monthly_delta_mae', 'monthly_delta_correlation']
    lines = ['# RiverDOCNet explicit-history/direct-readout results', '',
             'The update improves temporal concentration MAE/log-MAE relative to the matched original, but does not beat history random forest or XGBoost on those temporal point scores. Temporal change reconstruction does not improve consistently. Spatial MAE/log-MAE point scores deteriorate slightly while RMSE and predictive R-squared improve.', '',
             'The three information components are retained. The candidate adds a flattened-history MLP and a direct local readout. It was fixed before target scoring. All new checkpoints were selected solely using native validation MAE.', '',
             'Only 24 new candidate fits were needed. The 72 original/control fit packages are immutable reused dependencies. All 96 comparison packages and both prediction roles were independently replayed. Inputs, original labels, masks and normalization are matched. Tree controls use the same local environmental/history information without DOC context or river messages.', '',
             'The first frozen verification entry omitted initialization of the pinned XGBoost dependency path. A separately recorded wrapper loaded the unchanged parent dependency set and executed the unchanged frozen numerical verifier. No fitting, selection, predictions or metrics were modified. Its exact executed source is retained in evaluation_code_snapshot and bound by verification_environment_repair.json and evaluation_source_receipt.json.', '',
             'The original has 12,853 parameters; the update has 17,617 (+37.07%). The eight unused daily channels remain zero. Shared backbone initialization, original log loss, batching, optimizer and 60-epoch budget are retained. The comparison tests the combined architecture/capacity change, not isolated causal effects of either addition.', '',
             '## Spatial reconstruction', '', table(summary[summary.track == 'space'], concentration), '',
             'Metrics pool the same 7,262 observed queries at 104 held-out stations across five HUC4 regions within seed, then average three seeds.', '',
             '## Temporal gap average', '', table(temporal, concentration), '', table(temporal, dynamics), '',
             'The overview is the equal mean of the three declared gap lengths and three mask/optimization seeds. Query cores are reused across gaps; these are not nine independent repetitions.', '']
    for gap in (3, 6, 12):
        lines.extend([f'## {gap}-month surrounding gaps', '', table(summary[(summary.track == 'time') & (summary.gap_months == gap)], concentration), ''])
    lines.extend(['## Changes relative to original fusion', '',
                  '| Metric | Temporal error reduction (%) | Spatial error reduction (%) |', '| --- | --- | --- |'])
    for metric, change in changes.items():
        lines.append(f"| {metric} | {change['temporal_percent_error_reduction']:.3f} | {change['spatial_percent_error_reduction']:.3f} |")
    temporal_values = temporal.set_index('model_name')
    lines.extend(['', '## Interpretation', '',
                  f"Temporal concentration MAE falls from {original['mae']:.4f} to {updated['mae']:.4f} mg/L ({changes['mae']['temporal_percent_error_reduction']:.2f}%), and log-MAE falls by {changes['log1p_mae']['temporal_percent_error_reduction']:.2f}%. Every one of the nine fixed gap/mask/seed combinations has lower native MAE than the original. The exploratory HUC4-blocked equal-task intervals for both primary changes exclude zero; this is conditional development evidence, not independent confirmation.", '',
                  f"The temporal MAE remains above history random forest ({temporal_values.loc['rf_monthly', 'mae']:.4f}) and history XGBoost ({temporal_values.loc['xgb_monthly', 'mae']:.4f}) at every gap length. Their log-MAE point scores are also lower. Differences against the trees have wide intervals including zero, so neither stable population inferiority nor superiority is established. The tested update has not achieved the intended temporal benchmark lead.", '',
                  f"Temporal native station-centered R-squared falls from {original['station_centered_r2']:.4f} to {updated['station_centered_r2']:.4f}. Monthly change MAE rises from {original['monthly_delta_mae']:.4f} to {updated['monthly_delta_mae']:.4f} mg/L and exceeds the zero-change reference ({temporal_values.loc['hybrid_direct', 'zero_change_delta_mae']:.4f}). The variability ratio rises from {original['within_station_sd_ratio']:.3f} to {updated['within_station_sd_ratio']:.3f}, but increased amplitude is not increased dynamic accuracy. The small log-centered and delta-correlation gains have intervals including zero.", '',
                  f"Spatial MAE increases by {-changes['mae']['spatial_percent_error_reduction']:.2f}% and log-MAE by {-changes['log1p_mae']['spatial_percent_error_reduction']:.2f}%. Both primary difference intervals include zero. RMSE improves by {changes['rmse']['spatial_percent_error_reduction']:.2f}%; the update still has lower spatial primary point errors than the two tree controls. Thus spatial competitiveness is retained, but a joint-primary upgrade over original fusion is not established.", '',
                  'This is a useful but incomplete concentration-accuracy improvement. It does not justify declaring a final temporal-dynamics model or SOTA. The result shows that the proposed combination of explicit lag encoding and direct readout, at this fixed training budget, is insufficient to resolve the gap to strong tree baselines. The bundle and added capacity prevent attribution to either architectural addition alone. No additional training or model selection is performed by this report.'])
    selected = intervals[(intervals.gap_months == 0) & intervals.metric.isin(['mae', 'log1p_mae', 'log1p_station_centered_r2', 'monthly_delta_mae'])]
    lines.extend(['', '## Paired differences and uncertainty', '',
                  '| Track | Reference | Metric | New minus reference | Lower 95% | Upper 95% |', '| --- | --- | --- | --- | --- | --- |'])
    for row in selected.itertuples():
        lines.append(f'| {row.track} | {NAMES[row.reference]} | {row.metric} | {row.difference:.5f} | {row.lower_95:.5f} | {row.upper_95:.5f} |')
    lines.extend(['', '![Paired architecture effects](../experiments/phase4_transfer/doc_hybrid_fusion_v1/analysis/paired_architecture_effects.png)', '',
                  'Intervals use 10,000 paired HUC4-block resamples with identical weights across methods, seeds and gap lengths, pooling sufficient statistics within seed and then averaging. Spatial inference has only five clusters; temporal inference has '+str(time.cluster.nunique())+' clusters. The reused development cohort, shared training sets, basin connectivity and unadjusted multiple comparisons limit interpretation. Gap-zero temporal differences refer to equal-task averages, not pooled duplicate core observations.', '',
                  'Station-centered diagnostics remove each station\'s observed and predicted evaluation means separately. They do not measure concentration calibration or use DOC values as inference inputs. A standard-deviation ratio closer to one alone does not establish improved reconstruction. Changes compare consecutive observed months only.', '',
                  'The target remains the mean of available DOC samples in a calendar month, often one sampling day. The observed-query evaluation does not validate naturally missing cells, true continuously sampled monthly means, daily DOC or arbitrary river locations. Higher scores on this development comparison do not establish global SOTA, cross-river transfer or optimal components.', '',
                  'Evidence: summary.csv, by_seed.csv, temporal_overview.csv, paired_bootstrap.csv and independent_audit.json under experiments/phase4_transfer/doc_hybrid_fusion_v1/analysis. Baseline dependencies and runtime/dataset/mask hashes are bound by protocol.json and manifest.json.'])
    report = Path(protocol['repository']) / 'docs/doc_hybrid_fusion_results_20261012.md'
    report.write_text('\n'.join(lines)+'\n')
    outputs = ['spatial_sufficient_statistics.csv', 'temporal_sufficient_statistics.csv', 'paired_bootstrap.csv', 'temporal_mean_metrics.csv',
               'paired_effects_by_seed.csv', 'paired_effects.csv', 'interpretation_inputs.json',
               'paired_architecture_effects.png', 'paired_architecture_effects.pdf']
    (out / 'report_provenance.json').write_text(json.dumps({'script_sha256': sha256_file(__file__),
         'statistics_helper_sha256': sha256_file('scripts/report_doc_dynamic_reconstruction_v1.py'),
         'summary_sha256': sha256_file(out / 'summary.csv'), 'audit_sha256': sha256_file(out / 'independent_audit.json'),
         'evaluation_repair_sha256': sha256_file(root / 'verification_environment_repair.json'),
         'evaluation_source_receipt_sha256': sha256_file(root / 'evaluation_source_receipt.json'),
         'report': str(report), 'report_sha256': sha256_file(report),
         'outputs': {name: sha256_file(out / name) for name in outputs},
         'model_selection_modified': False}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'temporal_huc4_clusters': int(time.cluster.nunique()),
                      'report': str(report), 'changes': changes}))


if __name__ == '__main__':
    main()
