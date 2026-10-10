"""Plot conditional residual learning and measured upstream prediction gains."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from plot_doc_dynamic_river_v1 import forest, save
from run_doc_river_readout_crossfit_v2 import BASE, OBSERVED, ROOT

LABELS = {BASE: 'Complete current model', OBSERVED: 'Observed upstream DOC',
    'expanded_upstream': 'Previous upstream-state branch', 'refitted_state': 'Fitted-readout training control',
    'crossfit_state': 'Station-held readout training', 'crossfit_uniform': 'Station-held, uniform pooling',
    'crossfit_matched_upstream': 'Station-held, matched upstream',
    'crossfit_matched_nonupstream': 'Station-held, matched non-upstream'}
COLORS = ['#335C81', '#9CBABD', '#5A977F', '#AAC4A0', '#CE7D48', '#DBB375', '#9D89B1', '#B6A9CB']


def contrasts(effects, region):
    records = []
    for candidate, reference, label in (
        ('crossfit_state', 'refitted_state', 'Held versus fitted readout training'),
        ('crossfit_state', 'expanded_upstream', 'Versus previous upstream-state branch'),
        ('crossfit_state', BASE, 'Versus complete current model'),
        ('crossfit_state', 'crossfit_uniform', 'Learned versus uniform pooling'),
        ('crossfit_matched_upstream', 'crossfit_matched_nonupstream', 'Real versus matched non-upstream')):
        row = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference) & effects.region.eq(region)].iloc[0].copy()
        row['contrast'] = label
        records.append(row)
    return pd.DataFrame(records)


def main():
    root = ROOT
    (root/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary = pd.read_csv(root/'analysis/summary.csv').set_index('model_name')
    effects = pd.read_csv(root/'analysis/paired_effects.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14.3, 9.))
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
    ax.set_title('a  Identical final inference anchor and river inputs', loc='left')
    for ax, region, title in ((axes[0, 1], 'overall', 'b  Overall DOC reconstruction'),
                               (axes[1, 0], 'q90', 'c  Source-Q90 high-DOC reconstruction')):
        rows = contrasts(effects, region)
        forest(ax, rows, 'contrast', {name: name for name in rows.contrast})
        ax.set_xlabel('Paired MAE reduction for labelled comparison (%)')
        ax.set_title(title, loc='left')
    parts = pd.read_csv(root/'analysis/partition_metrics.csv')
    ax = axes[1, 1]
    for reference, label, color in (('refitted_state', 'Held versus fitted training', '#335C81'),
        ('expanded_upstream', 'Versus previous river version', '#CE7D48')):
        candidate = parts[parts.model_name.eq('crossfit_state')].set_index('split_seed')
        comparison = parts[parts.model_name.eq(reference)].set_index('split_seed')
        ax.plot(range(3), 100*(1-candidate.mae/comparison.mae), marker='o', color=color, label=label)
    ax.axhline(0, color='#60676C', lw=.8, ls='--')
    ax.set_xticks(range(3), ['Partition 142', 'Partition 143', 'Partition 144'])
    ax.set_ylabel('MAE reduction (%)')
    ax.set_title('d  Training-target and actual version gains differ', loc='left')
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.02, .012, 'Source-role development · conditional scalar-readout holdout, not complete-model OOF · 95% station intervals, 5,000 draws',
        fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .04, 1, 1), h_pad=2, w_pad=2)
    save(fig, root, 'river_conditional_readout_comparison')

    anchors = pd.read_csv(root/'analysis/training_anchor_diagnostics.csv')
    means = anchors.groupby(['split_seed', 'anchor'], as_index=False).mean(numeric_only=True)
    labels = {'original_fitted': 'Original fitted anchor', 'refitted_fitted': 'Refitted in-sample readout',
        'conditional_crossfit': 'Station-held readout', 'validation_anchor': 'Receiving validation anchor'}
    colors = ['#5A977F', '#AAC4A0', '#CE7D48', '#335C81']
    fig, axes = plt.subplots(1, 3, figsize=(14., 4.8))
    for ax, field, title, unit in ((axes[0], 'mae', 'a  Overall anchor errors', 'DOC MAE (mg/L)'),
        (axes[1], 'q90_mae', 'b  High-DOC anchor errors', 'Source-Q90 MAE (mg/L)'),
        (axes[2], 'wasserstein_to_validation_log_residual', 'c  Error-distribution distance', 'Wasserstein distance (log1p residual)')):
        for i, (name, color) in enumerate(zip(labels, colors, strict=True)):
            rows = means[means.anchor.eq(name)].sort_values('split_seed')
            if field.startswith('wasserstein') and name == 'validation_anchor':
                continue
            ax.bar(np.arange(3)+(i-1.5)*.19, rows[field], width=.18, color=color, label=labels[name])
        ax.set_xticks(range(3), ['142', '143', '144'])
        ax.set_xlabel('Source partition')
        ax.set_ylabel(unit)
        ax.set_title(title, loc='left')
        ax.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.02), fontsize=9)
    fig.text(.02, .01, 'Three-seed means · source/receiving populations differ · distance is descriptive, not a test of a causal explanation',
        fontsize=9, color='#596166')
    fig.tight_layout(rect=(0, .06, 1, .88), w_pad=2.5)
    save(fig, root, 'river_conditional_anchor_diagnostics')
    print('Saved two PNG/PDF/SVG scientific figures.')


if __name__ == '__main__':
    main()
