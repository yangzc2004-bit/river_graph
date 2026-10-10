"""Scientific figures for explicit river mixing, buffering and source support."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_confluence_storage_v1 import BASE, ROOT

LABELS = {BASE: 'Complete current model', 'dynamic_lagged': 'Previous observed-DOC branch',
    'expanded_upstream': 'Previous environmental-state branch', 'plain_upstream': 'Plain upstream attention',
    'confluence_mixing': 'Branch-frontier mixing', 'confluence_storage': 'Mixing + storage buffering',
    'matched_storage_upstream': 'Matched real upstream', 'matched_storage_nonupstream': 'Matched non-upstream'}
COLORS = ['#335C81', '#8BACBE', '#6C937E', '#A9C3B7', '#CE7D48', '#A56035', '#9D89B1', '#C3B4D4']


def contrasts(effects, region):
    rows = []
    for candidate, reference, label in (
        ('confluence_storage', 'plain_upstream', 'Versus ordinary upstream attention'),
        ('confluence_storage', 'confluence_mixing', 'Extra effect of storage buffering'),
        ('confluence_storage', 'expanded_upstream', 'Versus previous environmental branch'),
        ('confluence_storage', BASE, 'Versus complete current model'),
        ('matched_storage_upstream', 'matched_storage_nonupstream', 'Real versus matched non-upstream')):
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
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.))
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
    ax.set_title('a  Same receiving cells and frozen local model', loc='left')
    for ax, region, title in ((axes[0, 1], 'overall', 'b  Overall DOC reconstruction'),
        (axes[1, 0], 'q90', 'c  Source-Q90 high-DOC reconstruction')):
        rows = contrasts(effects, region)
        forest(ax, rows, 'contrast', {name: name for name in rows.contrast})
        ax.set_xlabel('Paired MAE reduction for labelled comparison (%)')
        ax.set_title(title, loc='left')
    parts = pd.read_csv(ROOT/'analysis/partition_metrics.csv')
    candidate = parts[parts.model_name.eq('confluence_storage')].set_index('split_seed')
    ax = axes[1, 1]
    for reference, label, color in (('plain_upstream', 'Versus ordinary attention', '#335C81'),
        ('confluence_mixing', 'Extra buffering effect', '#CE7D48'),
        ('expanded_upstream', 'Versus previous river version', '#6C937E')):
        baseline = parts[parts.model_name.eq(reference)].set_index('split_seed')
        ax.plot(range(3), 100*(1-candidate.mae/baseline.mae), marker='o', label=label, color=color)
    ax.axhline(0, color='#62686C', lw=.8, ls='--')
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction (%)')
    ax.set_title('d  Three-partition stability', loc='left')
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .012, 'ST357 source-development roles · 3 seeds · 95% paired whole-station intervals,5,000 draws · positive gain means lower error',
        fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1), w_pad=2, h_pad=2)
    save(fig, ROOT, 'confluence_storage_comparison')

    support = pd.read_csv(ROOT/'analysis/source_support.csv').groupby('split_seed').mean(numeric_only=True)
    operators = pd.read_csv(ROOT/'analysis/operator_summary.csv').set_index('model_name')
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 6.4))
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x-.17, 100*support.supported_fraction, .32, color='#335C81', label='Any observed upstream DOC')
    ax.bar(x+.17, 100*support.multiple_branches_cell_fraction, .32, color='#CE7D48', label='Multiple represented branches')
    ax.set_xticks(x, ['142', '143', '144'])
    ax.set_ylabel('Receiving query cells (%)')
    ax.set_xlabel('Source partition')
    ax.set_ylim(0, 100)
    ax.set_title('a  What the observed river bank covers', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.22))
    ax = axes[1]
    measured = 100*support.measured_flow_slots/support.frontier_lag_slots
    ax.bar(x, measured, color='#335C81', label='Measured monthly discharge')
    ax.bar(x, 100-measured, bottom=measured, color='#D8B58C', label='Drainage-area proxy')
    ax.set_xticks(x, ['142', '143', '144'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Retained frontier source–lag slots (%)')
    ax.set_xlabel('Source partition')
    ax.set_title('b  Flow information behind mixing weights', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.22))
    ax = axes[2]
    names = ['plain_upstream', 'confluence_mixing', 'confluence_storage']
    labels = ['Plain', 'Mixing', 'Mixing +\nstorage']
    bottom = np.zeros(3)
    for lag, color in ((0, '#335C81'), (1, '#8BACBE'), (3, '#D8B58C')):
        values = 100*operators.loc[names, f'river_lag_mass_{lag}'].to_numpy()
        ax.bar(x, values, bottom=bottom, color=color, label=f'{lag}-month memory')
        bottom += values
    values = 100*operators.loc[names, 'river_prior_mass'].to_numpy()
    ax.bar(x, values, bottom=bottom, color='#E4E6E8', label='Zero-message alternative')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Mean attention allocation (%)')
    ax.set_title('c  Allocation on supported query cells', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.22))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .01, 'Three-seed / three-partition means · observed innovations, not all river nodes · monthly memory is not physical travel time',
        fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .075, 1, 1), w_pad=2)
    save(fig, ROOT, 'confluence_source_support')
    print('Saved two PNG/PDF/SVG figures.')


if __name__ == '__main__':
    main()
