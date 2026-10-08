"""Paper figures for geographical and nested station-held river information."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_river_connected_comparison_v1 import ROOT as CONNECTED
from run_doc_river_frozen_comparison_v1 import ARMS, ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

LABELS = ['Current complete', 'Simple upstream', 'Structure upstream', 'Matched non-upstream']
COLORS = ['#737e86', '#c49856', '#287f7c', '#9178a2']


def save(fig, out, name):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'{name}.{extension}', dpi=230, facecolor='white')
    plt.close(fig)


def bars(ax, summary, title):
    values = summary.set_index('model_name').loc[list(ARMS), 'mae'].to_numpy()
    for i, (value, color) in enumerate(zip(values, COLORS, strict=True)):
        ax.barh(i, value, height=.57, color=color)
        ax.text(value+values.max()*.018, i, f'{value:.5f}', va='center', fontsize=8)
    ax.set(yticks=np.arange(4), yticklabels=LABELS, ylim=(3.6, -.6), xlim=(0, values.max()*1.24),
        xlabel='MAE (mg L$^{-1}$)', title=title)


def main():
    analysis, connected = ROOT/'analysis', CONNECTED/'analysis'
    out = ROOT/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), layout='constrained')
    bars(axes[0, 0], pd.read_csv(analysis/'primary_summary.csv'), 'Whole-region withholding: all four return the fixed base')
    coverage = pd.read_csv(analysis/'availability_by_region.csv', dtype={'target_huc4': str})
    coverage = coverage.set_index('target_huc4').loc[['1013', '1019', '0708', '1030', '1101']]
    y = np.arange(5)
    raw = 100*coverage.raw_supported_cells/coverage.test_cells
    common = 100*coverage.matched_supported_cells/coverage.test_cells
    axes[0, 1].barh(y, raw, height=.57, color='#d6e4e5', label='Raw real upstream availability')
    axes[0, 1].barh(y, common, height=.57, color='#287f7c', label='Availability matched to control')
    for i, (_, row) in enumerate(coverage.iterrows()):
        count = int(row.matched_supported_stations)
        axes[0, 1].text(raw.iloc[i]+.45, i, f'{count} supported station'+('' if count == 1 else 's'),
            va='center', fontsize=8)
    axes[0, 1].set(yticks=y, yticklabels=coverage.index, ylim=(4.6, -.6), xlim=(0, 38),
        xlabel='Test station-months with upstream DOC (%)', title='Whole-region tasks leave little observed upstream information')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(.48, -.17), fontsize=8, frameon=False)
    bars(axes[1, 0], pd.read_csv(connected/'summary.csv'), 'Nested station withholding: four source-development procedures')
    effects = pd.read_csv(connected/'paired_effects.csv')
    selected = effects[(effects.reference == 'current_complete') & (effects.subset == 'all')].set_index('candidate')
    for i, arm in enumerate(ARMS[1:]):
        row = selected.loc[arm]
        axes[1, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color=COLORS[i+1], lw=1.7)
        axes[1, 1].plot(row.relative_gain_pct, i, 'o', color=COLORS[i+1], ms=6)
        axes[1, 1].annotate(f'{row.relative_gain_pct:+.3f}%', (row.relative_gain_pct, i),
            xytext=(-6, 8), textcoords='offset points', ha='right', fontsize=8)
    axes[1, 1].axvline(0, lw=.8, color='#808080')
    axes[1, 1].set(yticks=np.arange(3), yticklabels=LABELS[1:], ylim=(2.7, -.7),
        xlim=(min(-.03, selected.gain_ci_low_pct.min()-.04), max(.07, selected.gain_ci_high_pct.max()+.04)),
        xlabel='MAE reduction versus fixed base (%)', title='Paired effects with 95% station bootstrap intervals')
    for letter, ax in zip('abcd', axes.ravel(), strict=True):
        ax.text(-.10, 1.07, letter, transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.grid(axis='x', color='#eeeeee', lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Does real river structure add DOC reconstruction information?', fontsize=14)
    fig.supxlabel('Added readouts may select zero. Identical output gives zero paired difference, not architecture equivalence.\n'
                  'ST357 retrospective analyses · Receiving water-quality inputs hidden · Source-development effects require confirmation',
                  fontsize=8)
    save(fig, out, 'four_procedure_comparison')
    diagnostic, ax = plt.subplots(1, 2, figsize=(12.5, 5.7), layout='constrained')
    scores = pd.read_csv(connected/'station_cv_scores.csv')
    scores = scores[scores.alpha.notna()].copy()
    scores['degradation'] = 100*(scores.validation_mae/scores.zero_reference_mae-1)
    best = scores.sort_values('validation_mae').drop_duplicates(['split_seed', 'seed', 'model_name', 'outer_fold'])
    rng = np.random.default_rng(42)
    for i, arm in enumerate(ARMS[1:]):
        values = best[best.model_name.eq(arm)].degradation.to_numpy()
        ax[0].scatter(values, i+rng.uniform(-.12, .12, len(values)), s=18, color=COLORS[i+1], alpha=.65)
        ax[0].plot(np.median(values), i, '|', ms=17, mew=2, color='#333333')
    ax[0].axvline(0, color='#666666', lw=.8)
    ax[0].set(yticks=np.arange(3), yticklabels=LABELS[1:], ylim=(2.6, -.6),
        xlabel='Best nonzero readout: inner-CV MAE change (%)', title='Calibration performance before outer-query evaluation')
    forms = pd.read_csv(connected/'form_population.csv').groupby('form').sum(numeric_only=True)
    names = ['form_1', 'form_2', 'form_3', 'unclassified']
    form_labels = ['Elongated, tributary-rich', 'Mainstem dominated', 'Broad, tributary-rich', 'Unclassified geometry']
    for i, name in enumerate(names):
        row = forms.loc[name]
        ax[1].barh(i, row.n_cells, height=.55, color='#d6e4e5')
        ax[1].barh(i, row.supported_cells, height=.55, color='#287f7c')
        ax[1].text(row.n_cells+60, i, f'{int(row.supported_cells)}/{int(row.n_cells)}', va='center', fontsize=8)
    ax[1].set(yticks=np.arange(4), yticklabels=form_labels, ylim=(3.6, -.6),
        xlim=(0, forms.n_cells.max()*1.29), xlabel='Unique receiving station-months',
        title='Upstream observations are concentrated in broad networks')
    ax[1].text(.5, -.16, 'Dark: matched support in ≥1 split; light: all unique queries.\nDifferent form populations cannot isolate a morphological cause.',
        transform=ax[1].transAxes, ha='center', fontsize=8)
    for i, axis in enumerate(ax):
        axis.text(-.1, 1.07, 'ab'[i], transform=axis.transAxes, fontweight='bold', fontsize=12)
        axis.grid(axis='x', color='#eeeeee', lw=.6)
        axis.set_axisbelow(True)
    diagnostic.suptitle('What limits transferable river corrections?', fontsize=13)
    save(diagnostic, out, 'river_calibration_and_form_support')
    paths = [analysis/'primary_summary.csv', analysis/'availability_by_region.csv',
             connected/'summary.csv', connected/'paired_effects.csv', connected/'station_cv_scores.csv', connected/'form_population.csv']
    write_json(out/'sources.json', {'plotter_sha256': sha256_file(__file__),
        'inputs': {str(path): sha256_file(path) for path in paths}, 'interval': '5000 paired station bootstrap draws'})


if __name__ == '__main__':
    main()
