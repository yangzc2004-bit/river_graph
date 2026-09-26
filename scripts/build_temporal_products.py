"""Build final H2X-T ensemble products and publication figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path('experiments/phase4_transfer/temporal_h2x_v1')
OUT = ROOT / 't9_products'
ANALYTES = ['doc', 'ph', 'spec_conductance']
MASKS = ['e1_r20_seed42', 'e2a_strict', 'e2b_partial', 'e3_spatial_seed42']
DATASETS = {
    'doc': 'data/processed/mississippi_graph_graphfix_st357.pt',
    'ph': 'data/processed/mississippi_graph_ph_st357.pt',
    'spec_conductance': 'data/processed/mississippi_graph_spec_conductance_st357.pt',
}


def load_ensemble(analyte: str, mask: str) -> pd.DataFrame:
    frames = []
    for seed in range(42, 47):
        path = ROOT / 't3_formal' / 'runs' / f'h2x_t__{analyte}__{mask}__seed{seed}' / 'full_grid.parquet'
        f = pd.read_parquet(path)
        f = f.sort_values(['station', 'month']).reset_index(drop=True)
        frames.append(f[['station', 'month', 'month_index', 'y_true', 'observed', 'split', 'y_pred']].rename(columns={'y_pred': f'y_pred_seed{seed}'}))
    out = frames[0].copy()
    for index, f in enumerate(frames[1:], start=43):
        keys = ['station', 'month', 'month_index', 'y_true', 'observed', 'split']
        if not out[keys].equals(f[keys]):
            raise ValueError(f'grid mismatch for {analyte}/{mask}')
        out[f'y_pred_seed{index}'] = f[f'y_pred_seed{index}'].to_numpy()
    pred_cols = [f'y_pred_seed{s}' for s in range(42, 47)]
    values = out[pred_cols].to_numpy(dtype=float)
    out['analyte'] = analyte
    out['mask'] = mask
    out['model_name'] = 'H2X-T-ensemble-5'
    out['y_pred'] = np.median(values, axis=1)
    out['y_pred_mean'] = values.mean(axis=1)
    out['seed_spread_sd'] = values.std(axis=1, ddof=1)
    out['seed_spread_iqr'] = np.quantile(values, .75, axis=1) - np.quantile(values, .25, axis=1)
    out['source_seeds'] = '42,43,44,45,46'
    return out[['analyte', 'mask', 'station', 'month', 'month_index', 'y_true',
                'y_pred', 'y_pred_mean', 'seed_spread_sd', 'seed_spread_iqr',
                'observed', 'split', 'model_name', 'source_seeds']]


def station_coordinates() -> pd.DataFrame:
    nodes = pd.read_csv('data/processed/graph_nodes_graphfix_st357.csv', dtype={'site_no': str})
    return nodes[['site_no', 'dec_lat_va', 'dec_long_va', 'huc_cd']].rename(columns={'site_no': 'station'})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    products = []
    source_hashes = {}
    for analyte in ANALYTES:
        for mask in MASKS:
            f = load_ensemble(analyte, mask)
            products.append(f)
            for seed in range(42, 47):
                source = ROOT / 't3_formal' / 'runs' / f'h2x_t__{analyte}__{mask}__seed{seed}' / 'full_grid.parquet'
                source_hashes[str(source)] = sha256_file(source)
    product = pd.concat(products, ignore_index=True)
    product.to_parquet(OUT / 'h2x_t_ensemble5_full_grid.parquet', index=False)
    product[product.observed].to_parquet(OUT / 'h2x_t_ensemble5_observed_grid.parquet', index=False)
    test = product[product['split'].eq('test')].copy()
    test['abs_error'] = (test.y_pred - test.y_true).abs()
    test.to_parquet(OUT / 'h2x_t_ensemble5_test_cells.parquet', index=False)
    station_error = test.groupby(['analyte', 'mask', 'station'], as_index=False).agg(
        test_cells=('abs_error', 'size'), mae=('abs_error', 'mean'),
        seed_spread_sd=('seed_spread_sd', 'mean'))
    station_error = station_error.merge(station_coordinates(), on='station', how='left')
    station_error.to_csv(OUT / 'test_station_error.csv', index=False)
    test.groupby(['analyte', 'mask'], as_index=False).agg(
        test_cells=('abs_error', 'size'), mae=('abs_error', 'mean'),
        median_seed_spread=('seed_spread_sd', 'median'),
        mean_seed_spread=('seed_spread_sd', 'mean')).to_csv(OUT / 'ensemble_test_summary.csv', index=False)
    make_figures(product, station_error)
    manifest = {
        'version': 'temporal_h2x_v1_t9_products',
        'model': 'H2X-T-ensemble-5', 'seeds': [42, 43, 44, 45, 46],
        'analytes': ANALYTES, 'masks': MASKS,
        'rows_full_grid': len(product), 'rows_test': len(test),
        'source_hashes': source_hashes,
        'notes': [
            'seed_spread_sd and seed_spread_iqr are ensemble dispersion diagnostics, not calibrated prediction intervals',
            'Products use the five audited T3 H2X-T runs per analyte and mask',
            'Figures are descriptive and do not select a test result for a primary claim',
        ],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['version', 'rows_full_grid', 'rows_test']}))


def make_figures(product: pd.DataFrame, station_error: pd.DataFrame) -> None:
    summary = pd.read_csv(ROOT / 't8_synthesis' / 'main_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        data = summary[summary.analyte.eq(analyte)].copy()
        pivot = data.set_index('mask')['reduction_pct'].reindex(MASKS)
        matrix = pivot.to_numpy()[None, :]
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-5, vmax=60, aspect='auto')
        ax.set_xticks(range(4), ['E1', 'E2a', 'E2b', 'E3'])
        ax.set_yticks([])
        ax.set_title(analyte.replace('_', ' '))
        for j, val in enumerate(pivot):
            ax.text(j, 0, f'{val:.1f}%', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=axes, label='MAE reduction (%)')
    fig.suptitle('H2X-T versus snapshot: five-seed reduction')
    fig.savefig(OUT / 'figure_main_reduction_heatmap.png', dpi=220)
    plt.close(fig)

    # Fixed descriptive station: lexicographically first station with test cells
    # in every analyte for E2a. This is a display choice, not model selection.
    e2 = product[(product['mask'] == 'e2a_strict') & product['split'].eq('test')]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        candidates = e2[e2.analyte.eq(analyte)].groupby('station').size()
        station = candidates.sort_values(ascending=False).index.sort_values()[0]
        d = product[(product['analyte'] == analyte) & (product['mask'] == 'e2a_strict') & (product['station'] == station)].copy()
        d['month'] = pd.to_datetime(d.month)
        observed = d[d['observed']]
        ax.plot(d.month, d.y_pred, color='#1f77b4', lw=1.2, label='H2X-T median')
        ax.fill_between(d.month, d.y_pred - d.seed_spread_sd, d.y_pred + d.seed_spread_sd,
                        color='#1f77b4', alpha=.15, label='seed ±1 SD')
        ax.scatter(observed.month, observed.y_true, s=6, color='#222222', alpha=.45, label='observed')
        ax.set_ylabel(analyte.replace('_', '\n'))
    axes[0].set_title('E2a descriptive station traces (station fixed per analyte)')
    axes[-1].set_xlabel('Month')
    axes[0].legend(ncol=3, fontsize=8)
    fig.savefig(OUT / 'figure_station_trace_e2a.png', dpi=220)
    plt.close(fig)

    # E2a spatial error maps, one panel per analyte.
    d = station_error[station_error['mask'].eq('e2a_strict')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, analyte in zip(axes, ANALYTES):
        x = d[d.analyte.eq(analyte)]
        sc = ax.scatter(x.dec_long_va, x.dec_lat_va, c=x.mae, s=14, cmap='magma', alpha=.85)
        ax.set_title(analyte.replace('_', ' ')); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        fig.colorbar(sc, ax=ax, label='Test MAE')
    fig.suptitle('E2a test-cell error by station: H2X-T ensemble median')
    fig.savefig(OUT / 'figure_spatial_error_e2a.png', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
