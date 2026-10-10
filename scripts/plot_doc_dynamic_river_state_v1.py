"""Scientific comparison of observed DOC and upstream environmental states."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_dynamic_river_state_v1 import BASE, OBSERVED, ROOT

LABELS = {BASE: 'Complete current model', OBSERVED: 'Observed upstream DOC',
    'state_upstream': 'Upstream state only', 'observed_state': 'Observed DOC + upstream state',
    'uniform_state': 'Uniform upstream state', 'matched_state_upstream': 'Hydro-matched upstream state',
    'matched_state_nonupstream': 'Hydro-matched non-upstream state'}
COLORS = {BASE: '#335C81', OBSERVED: '#9CBABD', 'state_upstream': '#5A977F',
    'observed_state': '#CE7D48', 'uniform_state': '#AAC4A0',
    'matched_state_upstream': '#DBB375', 'matched_state_nonupstream': '#9D89B1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    args = parser.parse_args()
    root = args.root
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(root/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(root/'analysis/paired_effects.csv')
    parts = pd.read_csv(root/'analysis/partition_metrics.csv')
    coverage = pd.read_csv(root/'analysis/source_support.csv').groupby('split_seed').mean(numeric_only=True)
    order = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax = axes[0, 0]
    for i, name in enumerate(order):
        x = summary.loc[name, 'mae']
        ax.scatter(x, i, color=COLORS[name], s=40)
        ax.annotate(f'{x:.4f}', (x, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(len(order)), [LABELS[name] for name in order])
    ax.invert_yaxis()
    minimum, maximum = summary.loc[order, 'mae'].min(), summary.loc[order, 'mae'].max()
    span = max(maximum-minimum, .004)
    ax.set_xlim(minimum-.15*span, maximum+.8*span)
    ax.set_xlabel('DOC MAE (mg/L; expanded scale)')
    ax.set_title('a  Fixed base and state-message extensions', loc='left')
    selected = effects[effects.reference.eq(BASE) & effects.region.eq('overall')].set_index('candidate').loc[order[2:]].reset_index()
    forest(axes[0, 1], selected, 'candidate', LABELS)
    axes[0, 1].set_title('b  Paired station uncertainty (5,000 draws)', loc='left')
    ax = axes[1, 0]
    for name in (OBSERVED, 'state_upstream', 'observed_state'):
        candidate = parts[parts.model_name.eq(name)].set_index('split_seed')
        reference = parts[parts.model_name.eq(BASE)].set_index('split_seed')
        ax.plot(range(3), 100*(1-candidate.mae/reference.mae), marker='o', color=COLORS[name], label=LABELS[name])
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction versus complete model (%)')
    ax.axhline(0, color='#60676C', lw=.8, ls='--')
    ax.set_title('c  Three-seed means within each partition', loc='left')
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    both = coverage.observed_fraction-(1-coverage.state_fraction-coverage.neither_fraction)
    observed_only = coverage.observed_fraction-both
    state_only = coverage.state_without_DOC_fraction
    bottom = np.zeros(3)
    for values, label, color in ((both, 'Both channels', '#5A977F'),
        (observed_only, 'Observed DOC only', '#9CBABD'),
        (state_only, 'State without DOC', '#CE7D48'),
        (coverage.neither_fraction, 'Neither channel', '#DEE4E7')):
        ax.bar(range(3), values, bottom=bottom, label=label, color=color, width=.62)
        bottom += values.to_numpy()
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of DOC query cells')
    ax.set_title('d  What upstream information is available?', loc='left')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=True, edgecolor='white')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .01, 'ST357 source-role development · K0 receiving water quality absent · frozen complete/observed anchors',
             fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=2., w_pad=2.)
    save(fig, root, 'upstream_state_comparison')

    strata = pd.read_csv(root/'analysis/stratified_effects.csv')
    selected = strata[strata.reference.eq(OBSERVED)].set_index('stratum')
    groups = ['observed_and_state', 'state_without_DOC', 'observed_without_state', 'neither_channel',
              'path_0_50_km', 'path_50_200_km', 'path_over_200_km']
    label = {'observed_and_state': 'Observed DOC and environmental state',
        'state_without_DOC': 'Environmental state, no observed DOC',
        'observed_without_state': 'Observed DOC, no usable state', 'neither_channel': 'Neither channel usable',
        'path_0_50_km': 'Nearest path ≤50 km', 'path_50_200_km': 'Nearest path 50–200 km',
        'path_over_200_km': 'Nearest path >200 km'}
    selected = selected.loc[[name for name in groups if name in selected.index]].reset_index()
    label = {r.stratum: f'{label[r.stratum]} (n={r.n_stations_unique} stations)' for r in selected.itertuples()}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    forest(ax, selected, 'stratum', label)
    ax.set_xlabel('Additional MAE reduction versus observed-DOC branch (%)')
    ax.set_title('Where does the upstream environmental-state branch add value?', loc='left', pad=15)
    fig.text(.02, .01, 'Fixed descriptive groups; overlapping stations · selected development sample · 95% station intervals',
             fontsize=8.5, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, root, 'upstream_state_strata')
    print('Saved two PNG/PDF/SVG scientific figures.')


if __name__ == '__main__':
    main()
