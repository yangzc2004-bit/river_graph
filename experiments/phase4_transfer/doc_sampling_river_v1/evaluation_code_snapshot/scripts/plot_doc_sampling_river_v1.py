"""Figures contrasting monthly pooling, actual sample age and measured phase."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_sampling_river_v1 import BASE, ROOT

LABELS = {BASE: 'Complete current model', 'plain_upstream': 'Ordinary upstream attention',
    'sampling_age': 'Actual source dates', 'sampling_hydro': 'Dates + measured flow phase',
    'shuffled_sampling_hydro': 'Shuffled source-flow correspondence',
    'expanded_upstream': 'Previous environmental-state branch',
    'matched_sampling_upstream': 'Matched real upstream',
    'matched_sampling_nonupstream': 'Matched non-upstream'}
COLORS = ['#335C81', '#8BACBE', '#A9C3B7', '#CE7D48', '#D8BBA4', '#6C937E', '#9D89B1', '#C3B4D4']


def contrast_rows(effects, region):
    rows = []
    for candidate, reference, label in (
        ('sampling_hydro', 'plain_upstream', 'Full sampling support vs ordinary attention'),
        ('sampling_hydro', 'sampling_age', 'Extra measured-flow phase information'),
        ('sampling_hydro', 'shuffled_sampling_hydro', 'Actual vs shuffled source-flow correspondence'),
        ('sampling_hydro', 'expanded_upstream', 'Versus previous environmental-state branch'),
        ('sampling_hydro', BASE, 'Versus complete current model'),
        ('matched_sampling_upstream', 'matched_sampling_nonupstream', 'Real vs matched non-upstream identity')):
        row = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference) & effects.region.eq(region)].iloc[0].copy()
        row['contrast'] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(ROOT/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(ROOT/'analysis/paired_effects.csv')
    fig, axes = plt.subplots(2, 2, figsize=(16., 9.5))
    ax = axes[0, 0]
    for i, (name, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        value = summary.loc[name, 'mae']
        ax.scatter(value, i, s=40, color=color)
        ax.annotate(f'{value:.4f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(len(LABELS)), list(LABELS.values()))
    ax.invert_yaxis()
    lo, hi = summary.loc[list(LABELS), 'mae'].min(), summary.loc[list(LABELS), 'mae'].max()
    span = max(hi-lo, .004)
    ax.set_xlim(lo-.15*span, hi+.75*span)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('DOC MAE (mg/L; expanded scale)')
    ax.set_title('a  Same monthly targets and frozen local predictor', loc='left')
    for ax, region, title in ((axes[0, 1], 'overall', 'b  Overall DOC reconstruction'),
        (axes[1, 0], 'q90', 'c  Source-Q90 high-DOC reconstruction')):
        rows = contrast_rows(effects, region)
        forest(ax, rows, 'contrast')
        ax.set_xlabel('Paired MAE reduction for labelled comparison (%)')
        ax.set_title(title, loc='left')
    parts = pd.read_csv(ROOT/'analysis/partition_metrics.csv')
    primary = parts[parts.model_name.eq('sampling_hydro')].set_index('split_seed')
    ax = axes[1, 1]
    for reference, label, color in (('plain_upstream', 'Versus ordinary attention', '#335C81'),
        ('sampling_age', 'Extra flow phase', '#CE7D48'),
        ('shuffled_sampling_hydro', 'Actual vs shuffled correspondence', '#9D89B1')):
        base = parts[parts.model_name.eq(reference)].set_index('split_seed')
        ax.plot(range(3), 100*(1-primary.mae/base.mae), marker='o', label=label, color=color)
    ax.axhline(0, color='#62686C', lw=.8, ls='--')
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction (%)')
    ax.set_title('d  Three-partition stability', loc='left')
    # Positive headroom keeps the legend clear of the tiny negative effects.
    ax.set_ylim(ax.get_ylim()[0], .005)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .012, 'ST357 source-development roles · 3 seeds · 95% paired whole-station intervals,5,000 draws · positive gain means lower error',
        fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1), w_pad=2, h_pad=2)
    save(fig, ROOT, 'sampling_river_comparison')

    support = pd.read_csv(ROOT/'analysis/source_support.csv').groupby('split_seed').mean(numeric_only=True)
    detail = pd.read_parquet(ROOT/'analysis/query_diagnostics.parquet')
    unique = detail.drop_duplicates(['split_seed', 'cell'])
    supported = unique[unique.river_support.eq(True)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(3)
    total = support.total_cells.to_numpy()
    for offset, value, label, color in (
        (-.18, support.supported_cells, 'Any upstream DOC', '#335C81'),
        (0., support.source_and_receiver_phase_cells, 'Both measured flow phases', '#CE7D48'),
        (.18, support.multiple_sample_day_cells, 'Multi-day monthly DOC', '#6C937E')):
        axes[0].bar(x+offset, value.to_numpy()/total*100, width=.18, label=label, color=color)
    axes[0].set_xticks(x, ['142', '143', '144'])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Receiving query cells (%)')
    axes[0].set_title('a  Actual input coverage', loc='left')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    for i, partition in enumerate((142, 143, 144)):
        values = np.sort(supported[supported.split_seed.eq(partition)].youngest_source_age_days.to_numpy())
        axes[1].plot(values, np.arange(1, len(values)+1)/len(values), label=str(partition), color=COLORS[i*2])
    axes[1].set_xlabel('Youngest source-mean age (days; capped at396)')
    axes[1].set_ylabel('Cumulative fraction of supported cells')
    axes[1].set_title('b  Source sampling age at month-end', loc='left')
    axes[1].legend(title='Partition', frameon=False, fontsize=8)
    rows = pd.read_csv(ROOT/'analysis/stratified_effects.csv')
    rows = rows[rows.reference.eq('plain_upstream') & rows.grouping.eq('phase_group')]
    labels = {'no_observed_upstream': 'No observed upstream DOC', 'no_source_phase': 'Source phase missing',
        'source_and_receiver_phase': 'Both source and receiver phase', 'source_phase_only': 'Only source phase'}
    forest(axes[2], rows, 'stratum', labels)
    axes[2].set_xlabel('Full sampling-support MAE reduction (%)')
    axes[2].set_title('c  Measured-phase strata (descriptive)', loc='left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=2)
    save(fig, ROOT, 'sampling_source_support')


if __name__ == '__main__':
    main()
