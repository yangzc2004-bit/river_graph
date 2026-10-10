"""Publication figures for protected dynamic river-message experiments."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_dynamic_river_v1 import BASE, ROOT

LABELS = {BASE: 'Complete current model', 'station_hidden_trees': 'Matched environmental trees',
    'static_upstream': 'Static upstream pooling', 'dynamic_same_month': 'Dynamic same-month messages',
    'dynamic_lagged': 'Dynamic upstream + history', 'matched_upstream': 'Support-matched upstream',
    'matched_nonupstream': 'Support-matched non-upstream'}
COLORS = {BASE: '#335C81', 'station_hidden_trees': '#A1A8AB', 'static_upstream': '#81A4AE',
    'dynamic_same_month': '#71A48B', 'dynamic_lagged': '#CE7D48',
    'matched_upstream': '#D5AD68', 'matched_nonupstream': '#9D89B1'}


def save(fig, root, name):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(root/'figures'/f'{name}.{suffix}', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def forest(ax, rows, label_column, labels=None):
    for i, row in enumerate(rows.itertuples()):
        ax.plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color='#335C81', lw=1.6)
        ax.scatter(row.relative_gain_pct, i, color='#CE7D48', s=35, zorder=3)
    ticks = rows[label_column].astype(str).tolist()
    ax.set_yticks(np.arange(len(rows)), [labels.get(x, x) for x in ticks] if labels else ticks)
    ax.invert_yaxis()
    ax.axvline(0, color='#60676C', lw=.8, ls='--')
    ax.set_xlabel('MAE reduction versus complete model (%)')
    ax.grid(axis='x', color='#E9EDEF', lw=.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    args = parser.parse_args()
    root = args.root
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(root/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(root/'analysis/paired_effects.csv')
    parts = pd.read_csv(root/'analysis/partition_metrics.csv')
    order = [BASE, 'static_upstream', 'dynamic_same_month', 'dynamic_lagged', 'matched_upstream', 'matched_nonupstream']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), gridspec_kw={'width_ratios': [1, 1.05]})
    ax = axes[0, 0]
    for i, name in enumerate(order):
        ax.scatter(summary.loc[name, 'mae'], i, color=COLORS[name], s=42)
        ax.annotate(f"{summary.loc[name, 'mae']:.4f}", (summary.loc[name, 'mae'], i),
                    xytext=(8, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(len(order)), [LABELS[name] for name in order])
    ax.invert_yaxis()
    limits = summary.loc[order, 'mae']
    span = max(float(limits.max()-limits.min()), .004)
    ax.set_xlim(float(limits.min())-.2*span, float(limits.max())+.8*span)
    ax.set_xlabel('DOC MAE (mg/L; expanded scale)')
    ax.set_title('a  Complete prediction and river extensions', loc='left')
    selected = effects[effects.reference.eq(BASE) & effects.region.eq('overall')].set_index('candidate').loc[order[1:]].reset_index()
    forest(axes[0, 1], selected, 'candidate', LABELS)
    axes[0, 1].set_title('b  Whole-station bootstrap, 5,000 draws', loc='left')
    ax = axes[1, 0]
    for name in ('static_upstream', 'dynamic_same_month', 'dynamic_lagged'):
        candidate = parts[parts.model_name.eq(name)].set_index('split_seed')
        base = parts[parts.model_name.eq(BASE)].set_index('split_seed')
        gain = 100*(1-candidate.mae/base.mae)
        ax.plot(range(3), gain, marker='o', color=COLORS[name], lw=1.5, label=LABELS[name])
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.axhline(0, color='#60676C', lw=.8, ls='--')
    ax.set_ylabel('DOC MAE reduction (%)')
    ax.set_title('c  Mean of three seeds within each partition', loc='left')
    ax.legend(frameon=True, framealpha=1., edgecolor='white', fontsize=8, loc='best')
    diagnostics = pd.read_parquet(root/'analysis/query_diagnostics.parquet')
    supported = diagnostics[diagnostics.river_support.eq(True)]
    weights = []
    for column in ('river_lag_mass_0', 'river_lag_mass_1', 'river_lag_mass_3', 'river_prior_mass'):
        weights.append(supported.groupby(['split_seed', 'seed'])[column].mean().groupby('split_seed').mean().mean())
    ax = axes[1, 1]
    ax.bar(range(4), weights, color=['#CE7D48', '#DFA775', '#ECD0AD', '#B6BEC2'], width=.65)
    ax.set_xticks(range(4), ['Current\nmonth', 'Previous\nmonth', '3 months\nearlier', 'Zero\nmessage'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean attention mass')
    ax.set_title('d  Allocation where upstream observations exist', loc='left')
    for i, value in enumerate(weights):
        ax.text(i, value+.025, f'{value:.2f}', ha='center', fontsize=9)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .01, 'ST357 source-role development · K0 receiving water quality absent · retained complete base frozen',
             fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=2, w_pad=2)
    save(fig, root, 'dynamic_river_comparison')

    strata = pd.read_csv(root/'analysis/stratified_effects.csv').set_index('stratum')
    groups = ['observed_upstream', 'no_observed_upstream', 'path_0_50_km', 'path_50_200_km',
              'path_over_200_km', 'age_0_1', 'age_2_6', 'age_7_12']
    labels = {'observed_upstream': 'Usable upstream observation', 'no_observed_upstream': 'No usable upstream observation',
        'path_0_50_km': 'Nearest path: ≤50 km', 'path_50_200_km': 'Nearest path: 50–200 km',
        'path_over_200_km': 'Nearest path: >200 km', 'age_0_1': 'Youngest observation: 0–1 month',
        'age_2_6': 'Youngest observation: 2–6 months', 'age_7_12': 'Youngest observation: 7–12 months'}
    rows = strata.loc[[name for name in groups if name in strata.index]].reset_index()
    labels = {row.stratum: f'{labels[row.stratum]}  (n={row.n_stations_unique} stations)' for row in rows.itertuples()}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    forest(ax, rows, 'stratum', labels)
    ax.set_title('Dynamic upstream messages: where DOC reconstruction changes', loc='left', pad=14)
    fig.text(.02, .01, 'Predefined descriptive groups; overlapping stations · selected development sample · 95% whole-station intervals',
             fontsize=8.5, color='#596166')
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, root, 'dynamic_river_strata')
    print('Saved two inspected-ready figures in PNG/PDF/SVG.')


if __name__ == '__main__':
    main()
