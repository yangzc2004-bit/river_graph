"""Render absolute-state versus contrast-value DOC message comparisons."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_river_state_contrast_v1 import BASE, OBSERVED, ROOT

LABELS = {BASE: 'Complete current model', OBSERVED: 'Observed upstream DOC',
    'expanded_upstream': 'Absolute upstream state', 'expanded_uniform': 'Absolute uniform state',
    'contrast_upstream': 'Upstream-to-local contrast', 'contrast_uniform': 'Uniform contrast',
    'contrast_matched_upstream': 'Matched upstream contrast',
    'contrast_matched_nonupstream': 'Matched non-upstream contrast'}
COLORS = ['#335C81', '#9CBABD', '#5A977F', '#AAC4A0', '#CE7D48', '#DBB375', '#9D89B1', '#B6A9CB']


def contrasts(effects, region):
    rows = []
    comparisons = [('contrast_upstream', BASE, 'Versus complete model'),
        ('contrast_upstream', OBSERVED, 'Versus observed-DOC branch'),
        ('contrast_upstream', 'expanded_upstream', 'Contrast versus absolute state'),
        ('contrast_upstream', 'contrast_uniform', 'Dynamic versus uniform contrast'),
        ('contrast_matched_upstream', 'contrast_matched_nonupstream', 'Real versus matched non-upstream')]
    for candidate, reference, label in comparisons:
        row = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference)
                      & effects.region.eq(region)].iloc[0].copy()
        row['contrast'] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    root = ROOT
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(root/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(root/'analysis/paired_effects.csv')
    fig, axes = plt.subplots(2, 2, figsize=(13.7, 8.9))
    ax = axes[0, 0]
    for i, (name, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        value = summary.loc[name, 'mae']
        ax.scatter(value, i, s=42, color=color)
        ax.annotate(f'{value:.4f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(len(LABELS)), list(LABELS.values()))
    ax.invert_yaxis()
    low, high = summary.loc[list(LABELS), 'mae'].min(), summary.loc[list(LABELS), 'mae'].max()
    span = max(high-low, .004)
    ax.set_xlim(low-.15*span, high+.8*span)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('DOC MAE (mg/L; expanded scale)')
    ax.set_title('a  Only the environmental message value changes', loc='left')
    for ax, region, title in ((axes[0, 1], 'overall', 'b  Overall DOC error'),
                               (axes[1, 0], 'q90', 'c  Source-Q90 high-DOC error')):
        rows = contrasts(effects, region)
        forest(ax, rows, 'contrast', {name: name for name in rows.contrast})
        ax.set_xlabel('Paired MAE reduction for labelled contrast (%)')
        ax.set_title(title, loc='left')
    parts = pd.read_csv(root/'analysis/partition_metrics.csv')
    ax = axes[1, 1]
    for reference, label, color in ((BASE, 'Versus complete model', '#335C81'),
        ('expanded_upstream', 'Additional contrast-value gain', '#CE7D48')):
        candidate = parts[parts.model_name.eq('contrast_upstream')].set_index('split_seed')
        comparison = parts[parts.model_name.eq(reference)].set_index('split_seed')
        ax.plot(range(3), 100*(1-candidate.mae/comparison.mae), marker='o', color=color, label=label)
    ax.axhline(0, color='#60676C', lw=.8, ls='--')
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction (%)')
    ax.set_title('d  Separate accumulated and incremental gains', loc='left')
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .01, 'ST357 source-role development · identical upstream coverage · 95% paired station intervals (5,000 draws)',
             fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=2, w_pad=2)
    save(fig, root, 'river_state_contrast_comparison')

    strata = pd.read_csv(root/'analysis/stratified_effects.csv')
    rows = strata[strata.reference.eq('expanded_upstream')].set_index('stratum')
    groups = ['usable_state', 'observed_and_state', 'state_without_DOC', 'neither_channel',
              'path_0_50_km', 'path_50_200_km', 'path_over_200_km']
    labels = {'usable_state': 'Any usable upstream environmental state',
        'observed_and_state': 'Upstream DOC and environmental state',
        'state_without_DOC': 'Environmental state, no upstream DOC', 'neither_channel': 'Neither river channel',
        'path_0_50_km': 'Nearest path ≤50 km', 'path_50_200_km': 'Nearest path 50–200 km',
        'path_over_200_km': 'Nearest path >200 km'}
    rows = rows.loc[[name for name in groups if name in rows.index]].reset_index()
    labels = {row.stratum: f'{labels[row.stratum]} (n={row.n_stations_unique} stations)' for row in rows.itertuples()}
    rows.loc[rows.n_stations_unique < 20, ['gain_ci_low_pct', 'gain_ci_high_pct']] = float('nan')
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    forest(ax, rows, 'stratum', labels)
    ax.set_xlabel('Additional MAE reduction versus absolute upstream states (%)')
    ax.set_title('Does an upstream-to-local contrast improve the existing message?', loc='left', pad=15)
    fig.text(.02, .01, 'Descriptive, overlapping strata · <20-station intervals omitted (retained in tables)',
             fontsize=8.5, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, root, 'river_state_contrast_strata')
    print('Saved two PNG/PDF/SVG scientific figures.')


if __name__ == '__main__':
    main()
