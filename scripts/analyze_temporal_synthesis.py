"""Reproduce T8 matched-budget synthesis from audited temporal products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path('experiments/phase4_transfer/temporal_h2x_v1')
SOURCES = ['', 't3_formal', 't4_ablation', 't5_current_only',
           't6_window_curve', 't7_snapshot_completion']
WINDOWS = {'h2x_t': 12, 'h2x_t_current_only': 1, 'h2x_t_lb3': 3,
           'h2x_t_lb6': 6, 'h2x_t_no_history': 12, 'h2x_t_hydro_only': 12}


def paired_bootstrap(errors: np.ndarray, clusters: np.ndarray, *,
                     repeats: int = 2000, seed: int = 20260927) -> dict:
    """errors is [unique query cells, 2 arms], already seed-averaged."""
    _, inverse = np.unique(clusters, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=errors[:, k])
                     for k in range(2)], axis=1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(count), size=(repeats, len(count)))
    sampled = sums[indices].sum(axis=1) / count[indices].sum(axis=1)[:, None]
    delta = sampled[:, 1] - sampled[:, 0]
    reduction = 100 * (1 - sampled[:, 1] / sampled[:, 0])
    return {'clusters': len(count), 'delta_lo': np.quantile(delta, .025),
                'delta_hi': np.quantile(delta, .975),
                'reduction_lo': np.quantile(reduction, .025),
                'reduction_hi': np.quantile(reduction, .975)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    out = root / 't8_synthesis'
    out.mkdir(exist_ok=True)
    rows, ledger, inputs = [], [], {}
    for source in SOURCES:
        directory = root / source
        # Each source directory was audited and frozen at its completion. Do
        # not silently re-audit old sidecars with a later hash schema: that
        # would turn a historical compatibility issue into a fake run failure.
        verdict = json.loads((directory / 'audit_verdict.json').read_text())
        if verdict.get('status') != 'pass':
            raise RuntimeError(f'{source}: frozen audit verdict is not pass')
        frame = pd.read_csv(directory / 'audit_metrics.csv')
        frame['source'] = source
        rows.append(frame)
        for run in frame.run:
            d = directory / 'runs' / run
            meta = json.loads((d / 'meta.json').read_text())
            conf = meta['config']
            for name in ['meta.json', 'full_grid.parquet', 'metrics.json']:
                inputs[str(d / name)] = sha256_file(d / name)
            kind = meta['model_name']
            effective_lb = WINDOWS.get(kind) if kind.startswith('h2x_t') else None
            effective_hidden = 64 if kind.startswith('h2x_t') else None
            issues = []
            if effective_lb is not None and conf.get('lookback') != effective_lb:
                issues.append('declared lookback differs from runner semantics')
            if conf.get('temporal_hidden') != effective_hidden:
                issues.append('declared temporal_hidden differs from runner semantics')
            ledger.append({'run': run, 'source': source, 'model': kind,
                               'declared_lookback': conf.get('lookback'),
                               'runtime_lookback_inferred': effective_lb,
                               'declared_temporal_hidden': conf.get('temporal_hidden'),
                               'runtime_temporal_hidden_inferred': effective_hidden,
                               'epochs_run': meta['training']['epochs_run'],
                               'max_epochs': conf['max_epochs'], 'patience': conf['patience'],
                               'runtime_hash': meta['runtime_code_snapshot_sha256'],
                               'issue': '; '.join(issues)})
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(out / 'training_metadata_ledger.csv', index=False)
    all_runs = pd.concat(rows, ignore_index=True)
    # Use T3 for the temporal arm, not duplicated pilot temporal fits.
    # The pilot contains the matched snapshot seeds 42--44, while T3 is the
    # authoritative five-seed temporal product. Avoid duplicate temporal rows.
    used = all_runs[~((all_runs.source == '') &
                      (all_runs.model_name == 'h2x_t'))]
    used.to_csv(out / 'audited_runs.csv', index=False)
    main_runs = used[used.model_name.isin(['h2x', 'h2x_t'])]
    summary, intervals = [], []
    for (analyte, mask), group in main_runs.groupby(['analyte', 'mask']):
        error_seeds, keys, truth = [], None, None
        for seed in range(42, 47):
            pair, metas = [], []
            for kind in ['h2x', 'h2x_t']:
                row = group[(group.seed == seed) & (group.model_name == kind)]
                assert len(row) == 1
                row = row.iloc[0]
                directory = root / row.source / 'runs' / row.run
                meta = json.loads((directory / 'meta.json').read_text())
                metas.append(meta)
                f = pd.read_parquet(directory / 'full_grid.parquet')
                f = f[f.split == 'test'].sort_values(['station', 'month'])
                this_keys = f[['station', 'month']].reset_index(drop=True)
                y = f.y_true.to_numpy()
                if keys is None:
                    keys, truth = this_keys, y
                assert keys.equals(this_keys) and np.array_equal(truth, y)
                pair.append(np.abs(y - f.y_pred.to_numpy()))
            for field in ['dataset_sha256', 'mask_sha256', 'max_epochs',
                          'patience', 'training_protocol', 'target_transform']:
                assert metas[0]['config'][field] == metas[1]['config'][field], field
            error_seeds.append(np.stack(pair, axis=1))
        errors = np.mean(error_seeds, axis=0)
        mean = errors.mean(axis=0)
        entry = {'analyte': analyte, 'mask': mask, 'query_cells': len(errors), 'seeds': 5,
                     'snapshot_mae': mean[0], 'temporal_mae': mean[1],
                     'delta_mae': mean[1]-mean[0],
                     'reduction_pct': 100*(1-mean[1]/mean[0]),
                     'better_seeds': sum(e[:, 1].mean() < e[:, 0].mean()
                                      for e in error_seeds)}
        for kind, prefix in [('h2x', 'snapshot'), ('h2x_t', 'temporal')]:
            g = group[group.model_name == kind]
            for metric in ['rmse', 'r2', 'log_mae', 'q90_mae']:
                entry[f'{prefix}_{metric}'] = g[metric].mean()
        entry['q90_n'] = int(group.q90_n.iloc[0])
        entry['q90_unstable'] = entry['q90_n'] < 20
        summary.append(entry)
        for cluster in ['station', 'month']:
            intervals.append(dict(analyte=analyte, mask=mask, cluster=cluster,
                                  **paired_bootstrap(errors, keys[cluster].to_numpy())))
    summary = pd.DataFrame(summary)
    intervals = pd.DataFrame(intervals)
    summary.to_csv(out / 'main_summary.csv', index=False)
    intervals.to_csv(out / 'paired_cluster_ci.csv', index=False)
    e2 = summary[summary['mask'].isin(['e2a_strict', 'e2b_partial'])]
    e2.groupby('analyte').reduction_pct.mean().to_csv(out / 'e2_equal_family_summary.csv')
    diag = used[used.seed.isin([42, 43, 44]) & (used.model_name != 'h2x')].copy()
    diag['lookback'] = diag.model_name.map(WINDOWS)
    diag.groupby(['analyte', 'mask', 'model_name', 'lookback'], as_index=False).agg(
        mae=('mae', 'mean'), mae_sd_seed=('mae', 'std'),
        rmse=('rmse', 'mean'), r2=('r2', 'mean'), q90_mae=('q90_mae', 'mean'),
        seeds=('seed', 'count')).to_csv(out / 'diagnostic_summary.csv', index=False)
    make_figures(summary, diag, out)
    issues = int(ledger.issue.ne('').sum())
    cap = int((ledger.epochs_run == ledger.max_epochs).sum())
    report = ['# T8 matched-budget synthesis', '',
              f'All {len(ledger)} source products passed the existing artifact-integrity audit.',
              (f'{issues} products have declarative temporal metadata discrepancies; '
              'see training_metadata_ledger.csv. Historical sidecars are unchanged.'),
              (f'{cap}/{len(ledger)} runs reached their maximum epoch count. '
              'These experiments establish a budget-specific comparison, not convergence.'), '',
              '## Five-seed temporal versus snapshot', '',
              ('| Analyte | Mask | Snapshot MAE | Temporal MAE | Reduction | '
              'Station-cluster delta CI | Month-cluster delta CI |'),
              '|---|---|---:|---:|---:|---|---|']
    for row in summary.itertuples():
        cis = intervals[(intervals.analyte == row.analyte) & (intervals['mask'] == row.mask)]
        formatted = []
        for cluster in ['station', 'month']:
            ci = cis[cis.cluster == cluster].iloc[0]
            formatted.append(f'[{ci.delta_lo:.4g}, {ci.delta_hi:.4g}]')
        report.append(f'| {row.analyte} | {row.mask} | {row.snapshot_mae:.4g} | '
                      f'{row.temporal_mae:.4g} | {row.reduction_pct:.2f}% | '
                      f'{formatted[0]} | {formatted[1]} |')
    report += ['', 'Delta = temporal minus snapshot; negative favors temporal.', '',
               '## Interpretation and next decision', '',
               ('- Keep lookback=12 as the working configuration. The 1/3/6/12 curve '
               'shows most of the pH and conductance improvement by six months.'),
               ('- Window length changes both available history and GRU computation. '
               'Repeated-current-month input over 12 steps has not been tested, so '
               'historical information alone is not identified by this comparison.'),
               ('- Reversed history preserves a fixed, learnable order; it does not '
               'prove time ordering irrelevant. Hydro-only removes current and past '
               'target channels and cannot isolate the ecological contribution.'),
               ('- Both cluster intervals are conditional on the chosen mask and fixed '
               'seed set. They do not estimate joint spatial-temporal dependence.'),
               ('- Repeated inspection of these test sets makes this a descriptive '
               'synthesis. Do not use it as an untouched confirmatory test.'),
               ('- Before a major training expansion, inspect validation learning curves '
               'and optimizer-update counts. Additional seeds do not fix undertraining.'),
               ('- Seed dispersion is not a calibrated predictive interval; this stage '
               'makes no new uncertainty or active-sampling claim.'), '',
               ('Reproduce: `.venv/bin/python scripts/analyze_temporal_synthesis.py` '
               '(or `uv run python scripts/analyze_temporal_synthesis.py` where uv is available).')]
    (out / 'report.md').write_text('\n'.join(report) + '\n')
    outputs = {str(f): sha256_file(f) for f in out.iterdir()
               if f.is_file() and f.name != 'manifest.json'}
    (out / 'manifest.json').write_text(json.dumps({
        'analysis_script_sha256': sha256_file(__file__), 'inputs': inputs, 'outputs': outputs,
        'repeats': 2000, 'rng_seed': 20260927, 'post_result_analysis': True,
        'metadata_discrepancies': issues, 'epochs_at_cap': cap}, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(f'Metadata discrepancies: {issues}; epochs at cap: {cap}/{len(ledger)}')


def make_figures(summary, diag, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (analyte, group) in zip(axes, summary.groupby('analyte')):
        x = np.arange(len(group))
        ax.bar(x - .18, group.snapshot_mae, .36, label='Snapshot')
        ax.bar(x + .18, group.temporal_mae, .36, label='H2X-T')
        ax.set_xticks(x, ['E1', 'E2a', 'E2b', 'E3'])
        ax.set_title(analyte)
        ax.set_ylabel('MAE (native units)')
    axes[0].legend()
    fig.suptitle('Five seeds; 10-epoch budget')
    fig.tight_layout()
    fig.savefig(out / 'main_comparison.png', dpi=180)
    plt.close(fig)
    windows = diag[diag.model_name.isin(['h2x_t', 'h2x_t_lb3', 'h2x_t_lb6',
                                       'h2x_t_current_only'])]
    windows = windows[windows['mask'].isin(['e2a_strict', 'e2b_partial'])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (analyte, group) in zip(axes, windows.groupby('analyte')):
        for mask, g in group.groupby('mask'):
            means = g.groupby('lookback').mae.mean()
            ax.plot(means.index, means.values, 'o-', label=mask)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xlabel('Window (months)')
        ax.set_ylabel('MAE (native units)')
        ax.set_title(analyte)
    axes[0].legend()
    fig.suptitle('Window diagnostics; matched three seeds')
    fig.tight_layout()
    fig.savefig(out / 'window_curve.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
