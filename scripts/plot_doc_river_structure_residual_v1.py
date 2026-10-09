"""Render DOC error, real-link contrasts, structure groups and lag allocation."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_doc_river_structure_residual_v1 import ROOT
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file

LABELS = {'station_hidden_trees': 'Strong environmental trees', 'available_real_integrated': 'Complete current model',
          'river_none': 'Matched current refit', 'river_simple': 'Simple upstream messages',
          'river_structure': 'Structure conditioned messages', 'river_rewired': 'Rewired source control'}
COLORS = {'station_hidden_trees': '#879aa9', 'available_real_integrated': '#555d65', 'river_none': '#aab0b3',
          'river_simple': '#bf9953', 'river_structure': '#287c78', 'river_rewired': '#8a759e'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=type(ROOT), default=ROOT)
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    analysis = args.root/('partial_analysis' if args.available_only else 'analysis')
    summary = pd.read_csv(analysis/'summary.csv').set_index('model_name')
    effects = pd.read_csv(analysis/'paired_effects.csv')
    strata = pd.read_csv(analysis/'structure_support_effects.csv')
    diagnostics = pd.read_csv(analysis/'river_diagnostics.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.25, right=.97, bottom=.20, top=.90, wspace=.98, hspace=.55)
    names = list(LABELS)
    largest = float(summary.loc[names, 'mae'].max())
    for i, name in enumerate(names):
        value = float(summary.loc[name, 'mae'])
        axes[0, 0].barh(i, value, color=COLORS[name], height=.58)
        axes[0, 0].text(value+.02*largest, i, f'{value:.3f}', va='center', fontsize=8)
    axes[0, 0].set(yticks=np.arange(len(names)), yticklabels=[LABELS[n] for n in names],
        ylim=(len(names)-.5, -.5), xlim=(0, largest*1.23), xlabel='MAE (mg L$^{-1}$)', title='Unmonitored station DOC reconstruction')
    references = ['available_real_integrated', 'river_none', 'river_simple', 'river_rewired']
    for i, reference in enumerate(references):
        row = effects[(effects.candidate == 'river_structure') & (effects.reference == reference) & (effects.region == 'overall')].iloc[0]
        axes[0, 1].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color='#287c78', lw=1.8)
        axes[0, 1].plot(row.relative_gain_pct, i, 'o', color='#287c78', ms=5)
    axes[0, 1].axvline(0, lw=.8, color='#555555')
    axes[0, 1].set(yticks=np.arange(len(references)), yticklabels=[LABELS[n].replace(' model', '\nmodel').replace(' messages', '\nmessages') for n in references],
        ylim=(len(references)-.5, -.5), xlabel='MAE reduction (%)', title='Structure messages versus each comparator')
    selected = strata[(strata.candidate == 'river_structure') & (strata.reference == 'river_none')].copy()
    order = ['low_order', 'chain', 'confluence', 'mainstem', 'storage', 'observed_upstream', 'no_observed_upstream']
    labels = {'low_order': 'Low order tributaries', 'chain': 'Chain settings', 'confluence': 'Major confluence vicinity',
              'mainstem': 'Integrated mainstems', 'storage': 'Lake or reservoir paths',
              'observed_upstream': 'Upstream observation available', 'no_observed_upstream': 'No upstream observation'}
    selected = selected.set_index('stratum').reindex([name for name in order if name in set(selected.stratum)])
    for i, (_, row) in enumerate(selected.iterrows()):
        axes[1, 0].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i, i], color='#287c78', lw=1.7)
        axes[1, 0].plot(row.relative_gain_pct, i, 's', color='#287c78', ms=4)
    axes[1, 0].axvline(0, lw=.8, color='#555555')
    axes[1, 0].set(yticks=np.arange(len(selected)),
        yticklabels=[f'{labels[name]} (n={int(row.n_stations_unique)})' for name, row in selected.iterrows()],
        ylim=(len(selected)-.5, -.5), xlabel='MAE reduction versus matched refit (%)', title='Physical structure and source support')
    lag_models = ['river_simple', 'river_structure', 'river_rewired']
    means = diagnostics.groupby(['split_seed', 'model_name']).mean(numeric_only=True).groupby('model_name').mean()
    bottom = np.zeros(len(lag_models))
    lag_colors = ['#2b6f78', '#77a1a8', '#c8d9da', '#dddddd']
    for column, label, color in zip(['river_lag_mass_0', 'river_lag_mass_1', 'river_lag_mass_3', 'river_prior_mass'],
                                   ['Current month', 'One month earlier', 'Three months earlier', 'Zero message prior'], lag_colors, strict=True):
        values = means.loc[lag_models, column].to_numpy(float)
        axes[1, 1].bar(np.arange(3), values, bottom=bottom, color=color, label=label, width=.55)
        bottom += values
    np.testing.assert_allclose(bottom, 1., rtol=0, atol=2e-6)
    axes[1, 1].set(xticks=np.arange(3), xticklabels=['Simple\nupstream', 'Structure\nconditioned', 'Rewired\ncontrol'],
        ylim=(0, 1), ylabel='Mean allocated attention mass', title='Predictive lag allocation across all query cells')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(.5, -.19), frameon=False, ncol=2, fontsize=8)
    for letter, ax in zip('abcd', axes.ravel(), strict=True):
        ax.text(-.17, 1.10, letter, transform=ax.transAxes, fontweight='bold', fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis='x' if ax != axes[1, 1] else 'y', color='#eeeeee', lw=.6)
    counts = pd.read_csv(analysis/'run_metrics.csv')
    packages = counts[['split_seed', 'seed']].drop_duplicates().shape[0]
    draws = json.loads((analysis/'sources.json').read_text())['bootstrap_draws']
    fig.suptitle('Real river information in the current DOC reconstruction model', fontsize=13, y=.96)
    scope = 'Partial source development' if args.available_only else 'Source validation development'
    fig.text(.55, .065, f'{scope} · {packages} split–seed packages · Receiving water quality hidden · {draws:,} paired station draws', ha='center', fontsize=8)
    fig.text(.55, .040, 'Structure profiles overlap; n counts unique stations. Lag weights describe information allocation, not travel time.', ha='center', fontsize=8, color='#666666')
    out = args.root/('partial_figures' if args.available_only else 'figures')
    out.mkdir(exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'river_structure_comparison.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    distance_fig, distance_axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    distance_fig.subplots_adjust(left=.22, right=.97, bottom=.26, top=.82, wspace=1.10)
    bands = ['path_0_to_50_km', 'path_50_to_200_km', 'path_over_200_km', 'no_eligible_path']
    band_labels = ['Nearest upstream path ≤50 km', 'Nearest path 50–200 km',
                   'Nearest path >200 km', 'No eligible upstream path']
    for offset, reference, color, marker_style, legend in [
            (-.12, 'river_none', '#287c78', 'o', 'Versus matched refit'),
            (.12, 'river_rewired', '#8a759e', 's', 'Versus rewired sources')]:
        group = strata[strata.reference.eq(reference) & strata.candidate.eq('river_structure')].set_index('stratum')
        for i, name in enumerate(bands):
            row = group.loc[name]
            distance_axes[0].plot([row.gain_ci_low_pct, row.gain_ci_high_pct], [i+offset, i+offset], lw=1.7, color=color)
            distance_axes[0].plot(row.relative_gain_pct, i+offset, marker_style, color=color, ms=5,
                                  label=legend if i == 0 else None)
    distance_axes[0].axvline(0, color='#555555', lw=.8)
    distance_axes[0].set(yticks=np.arange(4), yticklabels=band_labels, ylim=(3.5, -.5),
        xlabel='MAE reduction (%) with 95% station interval', title='River gain by source path distance')
    distance_axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=1, frameon=False, fontsize=8)
    original = float(summary.loc['available_real_integrated', 'mae'])
    silent = float(summary.loc['river_structure_zero_river', 'mae'])
    complete = float(summary.loc['river_structure', 'mae'])
    increments = np.array([silent-original, complete-silent, complete-original])*100/original
    for i, (value, color) in enumerate(zip(increments, ['#bf9953', '#287c78', '#7b858a'], strict=True)):
        distance_axes[1].barh(i, value, color=color, height=.48)
        distance_axes[1].text(value+(.025 if value >= 0 else -.025), i, f'{value:+.3f}%',
                              ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    distance_axes[1].axvline(0, color='#555555', lw=.8)
    span = max(abs(increments))*1.55
    distance_axes[1].set(yticks=np.arange(3), yticklabels=['Refitted nonriver paths', 'Direct river correction', 'Combined model change'],
        ylim=(2.6, -.6), xlim=(-span, span), xlabel='Change in MAE (% of current model error)', title='Error changes within the joint model')
    distance_axes[1].text(.5, -.23, 'Point estimates on the same query cells.\nPositive: higher error; negative: lower error.',
                           transform=distance_axes[1].transAxes, ha='center', fontsize=8, color='#666666')
    for letter, ax in zip('ab', distance_axes, strict=True):
        ax.text(-.19, 1.10, letter, transform=ax.transAxes, fontweight='bold', fontsize=12)
        ax.grid(axis='x', color='#eeeeee', lw=.6)
        ax.set_axisbelow(True)
    distance_fig.suptitle('Where real river messages add information', fontsize=13, y=.95)
    distance_fig.text(.56, .05, f'{scope} · {packages} split–seed packages · Receiving water quality hidden · No external evaluation',
                      ha='center', fontsize=8, color='#666666')
    for extension in ('png', 'pdf', 'svg'):
        distance_fig.savefig(out/f'river_distance_and_error_components.{extension}', dpi=220, facecolor='white')
    plt.close(distance_fig)
    write_json(out/'sources.json', {'plotter_sha256': sha256_file(__file__), 'evaluation_role': scope,
        'csvs': {name: sha256_file(analysis/name) for name in ('summary.csv', 'paired_effects.csv',
            'structure_support_effects.csv', 'river_diagnostics.csv', 'run_metrics.csv')}})


if __name__ == '__main__':
    main()
