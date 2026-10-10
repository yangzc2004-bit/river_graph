"""Independent arithmetic checks of the upstream state primary comparison."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from run_doc_dynamic_river_state_v1 import BASE, OBSERVED, ROOT
from run_unified_doc_spatial import write_json


def check(reference):
    packages, points, records = [], {}, []
    for path in sorted((ROOT/'runs').glob('*/complete.json')):
        config = json.loads((path.parent/'config.json').read_text())
        frame = pd.read_parquet(path.parent/'predictions.parquet')
        columns = ['cell', 'station', 'y_true', 'y_pred']
        base = frame[frame.model_name.eq(reference)][columns].set_index('cell')
        model = frame[frame.model_name.eq('observed_state')][columns].set_index('cell')
        np.testing.assert_array_equal(base.index, model.index)
        np.testing.assert_array_equal(base.y_true, model.y_true)
        packages.append(pd.DataFrame({'split': config['split_seed'], 'seed': config['seed'],
            'cell': base.index, 'station': base.station.to_numpy(), 'y_true': base.y_true.to_numpy(),
            'base_error': abs(base.y_pred-base.y_true).to_numpy(),
            'model_error': abs(model.y_pred-model.y_true).to_numpy(),
            'base_log_error': abs(np.log1p(base.y_pred)-np.log1p(base.y_true)).to_numpy(),
            'model_log_error': abs(np.log1p(model.y_pred)-np.log1p(model.y_true)).to_numpy(),
            'tail': base.y_true.to_numpy() >= config['q90_threshold_train']}))
    panel = pd.concat(packages, ignore_index=True)
    effects = pd.read_csv(ROOT/'analysis/paired_effects.csv')
    primary = effects[effects.candidate.eq('observed_state') & effects.reference.eq(reference)]
    for region in ('overall', 'q90', 'log1p'):
        part = panel[panel['tail']] if region == 'q90' else panel
        names = ('base_log_error', 'model_log_error') if region == 'log1p' else ('base_error', 'model_error')
        # Recompute from each run directly, independent of the report's helpers.
        run_mean = part.groupby(['split', 'seed'])[list(names)].mean()
        average = run_mean.groupby('split').mean().mean()
        gain = 100*(1-average[names[1]]/average[names[0]])
        saved = primary[primary.region.eq(region)].iloc[0]
        np.testing.assert_allclose(gain, saved.relative_gain_pct, rtol=0, atol=1e-10)
        np.testing.assert_allclose(average[names[0]], saved.reference_mae, rtol=0, atol=1e-12)
        np.testing.assert_allclose(average[names[1]], saved.candidate_mae, rtol=0, atol=1e-12)
        points[region] = {'reference_error': float(average[names[0]]),
                          'candidate_error': float(average[names[1]]), 'gain_pct': float(gain)}
    # Independent station-quantity implementation, same deterministic draws.
    observed = panel.groupby(['split', 'station', 'cell'])[['base_error', 'model_error']].mean().reset_index()
    station = observed.groupby(['split', 'station']).agg(n=('cell', 'size'),
                a=('model_error', 'sum'), b=('base_error', 'sum')).reset_index()
    ids = sorted(station.station.unique())
    splits = sorted(station.split.unique())
    matrices = []
    for split in splits:
        rows = station[station.split.eq(split)].set_index('station').reindex(ids).fillna(0)
        matrices.append(rows[['n', 'a', 'b']].to_numpy())
    weights = np.random.default_rng(42).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=5000)
    means = []
    for values in matrices:
        totals = weights @ values
        assert (totals[:, 0] > 0).all()
        means.append(totals[:, 1:]/totals[:, :1])
    means = np.mean(means, axis=0)
    gains = 100*(1-means[:, 0]/means[:, 1])
    ci = np.quantile(gains, [.025, .975])
    saved = primary[primary.region.eq('overall')].iloc[0]
    np.testing.assert_allclose(ci, [saved.gain_ci_low_pct, saved.gain_ci_high_pct], rtol=0, atol=1e-10)
    for split, group in observed.groupby('split'):
        station_mean = group.groupby('station')[['base_error', 'model_error']].mean().mean()
        records.append({'split': int(split), 'station_equal_base': float(station_mean.base_error),
                        'station_equal_candidate': float(station_mean.model_error)})
    station_base = np.mean([row['station_equal_base'] for row in records])
    station_candidate = np.mean([row['station_equal_candidate'] for row in records])
    return {
        'points': points, 'primary_station_bootstrap_95ci_pct': ci.tolist(),
        'bootstrap_draws': 5000, 'station_equal_gain_pct': 100*(1-station_candidate/station_base),
        'unique_stations': len(ids), 'unique_station_months': panel[['station', 'cell']].drop_duplicates().shape[0],
        'split_cell_occurrences': len(observed), 'seed_repetitions_are_not_ecological_samples': True,
        'all_primary_arithmetic_and_ci_match': True}


def main():
    write_json(ROOT/'verification/independent_calculations.json',
        {reference: check(reference) for reference in (BASE, OBSERVED)})
    print('Both anchors: independent primary MAE, Q90, log1p and 5,000-draw station CIs match.')


if __name__ == '__main__':
    main()
