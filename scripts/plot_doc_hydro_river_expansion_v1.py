"""Render controlled DOC comparisons and the actual added hydro coverage."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_hydro_river_expansion_v1 import BASE, OBSERVED, ROOT

LABELS = {BASE: 'Complete current model', OBSERVED: 'Observed upstream DOC',
    'observed_state': 'Previous upstream state', 'expanded_upstream': 'Expanded upstream state',
    'expanded_uniform': 'Expanded uniform state',
    'expanded_matched_upstream': 'Matched real upstream',
    'expanded_matched_nonupstream': 'Matched non-upstream'}
COLORS = ['#335C81', '#9CBABD', '#5A977F', '#CE7D48', '#AAC4A0', '#DBB375', '#9D89B1']


def main():
    root = ROOT
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(root/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(root/'analysis/paired_effects.csv')
    coverage = pd.read_csv(root/'analysis/source_support.csv').groupby('split_seed').mean(numeric_only=True)
    order = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    ax = axes[0, 0]
    for i, (name, color) in enumerate(zip(order, COLORS, strict=True)):
        value = summary.loc[name, 'mae']
        ax.scatter(value, i, s=42, color=color)
        ax.annotate(f'{value:.4f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(range(len(order)), [LABELS[name] for name in order])
    ax.invert_yaxis()
    low, high = summary.loc[order, 'mae'].min(), summary.loc[order, 'mae'].max()
    span = max(high-low, .004)
    ax.set_xlim(low-.15*span, high+.8*span)
    ax.set_xlabel('DOC MAE (mg/L; expanded scale)')
    ax.set_title('a  Same frozen model, expanded covariate pool', loc='left')
    rows = []
    contrasts = [('expanded_upstream', BASE, 'Versus complete model'),
        ('expanded_upstream', OBSERVED, 'Versus observed-DOC branch'),
        ('expanded_upstream', 'observed_state', 'Versus previous state branch'),
        ('expanded_upstream', 'expanded_uniform', 'Dynamic versus uniform'),
        ('expanded_matched_upstream', 'expanded_matched_nonupstream', 'Real versus non-upstream')]
    for candidate, reference, label in contrasts:
        row = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference) & effects.region.eq('overall')].iloc[0].copy()
        row['contrast'] = label
        rows.append(row)
    rows = pd.DataFrame(rows)
    forest(axes[0, 1], rows, 'contrast', {name: name for name in rows.contrast})
    axes[0, 1].set_xlabel('Paired MAE reduction for labelled contrast (%)')
    axes[0, 1].set_title('b  Paired station intervals (5,000 draws)', loc='left')
    ax = axes[1, 0]
    for offset, field, label, color in ((-.18, 'previous_state_fraction', 'Original source pool', '#5A977F'),
        (.18, 'state_fraction', 'Expanded hydro-only pool', '#CE7D48')):
        values = coverage[field]*100
        ax.bar([i+offset for i in range(3)], values, width=.34, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(i+offset, value+1, f'{value:.1f}', ha='center', fontsize=9)
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylim(0, 80)
    ax.set_ylabel('Usable upstream state / DOC query cells (%)')
    ax.set_title('c  Hydro availability limits the added coverage', loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    parts = pd.read_csv(root/'analysis/partition_metrics.csv')
    ax = axes[1, 1]
    for reference, label, color in ((BASE, 'Versus complete model', '#335C81'),
        ('observed_state', 'Additional pool-expansion gain', '#CE7D48')):
        candidate = parts[parts.model_name.eq('expanded_upstream')].set_index('split_seed')
        comparison = parts[parts.model_name.eq(reference)].set_index('split_seed')
        ax.plot(range(3), 100*(1-candidate.mae/comparison.mae), marker='o', color=color, label=label)
    ax.axhline(0, color='#60676C', lw=.8, ls='--')
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction (%)')
    ax.set_title('d  Separate accumulated and incremental gains', loc='left')
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .01, 'ST357 source-role development · known-atlas hydro covariates · receiving water quality absent',
             fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=2, w_pad=2)
    save(fig, root, 'hydro_expansion_comparison')

    strata = pd.read_csv(root/'analysis/stratified_effects.csv')
    rows = strata[strata.reference.eq('observed_state')].set_index('stratum')
    groups = ['newly_supported', 'previously_supported', 'still_unsupported', 'state_without_DOC',
              'path_0_50_km', 'path_50_200_km', 'path_over_200_km']
    labels = {'newly_supported': 'Newly supported by hydro-only nodes',
        'previously_supported': 'Previously supported', 'still_unsupported': 'Still unsupported',
        'state_without_DOC': 'Usable state, no upstream DOC', 'path_0_50_km': 'Nearest path ≤50 km',
        'path_50_200_km': 'Nearest path 50–200 km', 'path_over_200_km': 'Nearest path >200 km'}
    rows = rows.loc[[name for name in groups if name in rows.index]].reset_index()
    labels = {row.stratum: f'{labels[row.stratum]} (n={row.n_stations_unique} stations)' for row in rows.itertuples()}
    # A four-station band needs many empty-partition redraws; its narrow
    # conditional bootstrap interval must not appear as strong field evidence.
    rows.loc[rows.n_stations_unique < 20, ['gain_ci_low_pct', 'gain_ci_high_pct']] = float('nan')
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    forest(ax, rows, 'stratum', labels)
    ax.set_xlabel('Additional MAE reduction versus previous state branch (%)')
    ax.set_title('Where do additional upstream hydro nodes change DOC error?', loc='left', pad=15)
    fig.text(.02, .01, 'Descriptive, overlapping strata · 95% station intervals; <20-station intervals omitted (retained in tables)',
             fontsize=8.5, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, root, 'hydro_expansion_strata')
    print('Saved two PNG/PDF/SVG scientific figures.')


if __name__ == '__main__':
    main()
