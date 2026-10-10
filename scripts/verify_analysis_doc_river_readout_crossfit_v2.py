"""Independently recalculate cell estimands and four primary station intervals."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from run_doc_river_readout_crossfit_v2 import ROOT
from run_unified_doc_spatial import write_json


def main():
    frames, thresholds = [], {}
    for path in sorted((ROOT/'runs').glob('*/complete.json')):
        run = path.parent
        config = json.loads((run/'config.json').read_text())
        frames.append(pd.read_parquet(run/'predictions.parquet'))
        thresholds[(config['split_seed'], config['seed'])] = config['q90_threshold_train']
    panel = pd.concat(frames, ignore_index=True)
    summary = pd.read_csv(ROOT/'analysis/summary.csv').set_index('model_name')
    for name, frame in panel.groupby('model_name'):
        per = []
        for (_, _), group in frame.groupby(['split_seed', 'seed']):
            q = thresholds[(int(group.split_seed.iloc[0]), int(group.seed.iloc[0]))]
            tail = group.y_true >= q
            per.append([float(abs(group.y_true-group.y_pred).mean()),
                float(abs(np.log1p(group.y_true)-np.log1p(group.y_pred)).mean()),
                float(abs(group.loc[tail, 'y_true']-group.loc[tail, 'y_pred']).mean())])
        np.testing.assert_allclose(np.mean(per, axis=0), summary.loc[name, ['mae', 'log_mae', 'q90_mae']].to_numpy(float), rtol=2e-12, atol=2e-12)
    effects = pd.read_csv(ROOT/'analysis/paired_effects.csv')
    checks = []
    comparisons = [('crossfit_state', 'refitted_state'), ('crossfit_state', 'expanded_upstream'),
        ('crossfit_state', 'crossfit_uniform'), ('crossfit_matched_upstream', 'crossfit_matched_nonupstream')]
    for candidate, reference in comparisons:
        keys = ['split_seed', 'seed', 'station', 'cell']
        a = panel[panel.model_name.eq(candidate)][[*keys, 'y_true', 'y_pred']]
        b = panel[panel.model_name.eq(reference)][[*keys, 'y_true', 'y_pred']]
        merged = a.merge(b, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
        for region in ('overall', 'q90', 'log1p'):
            group = merged.copy()
            if region == 'q90':
                threshold = np.array([thresholds[(s, seed)] for s, seed in zip(group.split_seed, group.seed, strict=True)])
                group = group[group.y_true_a >= threshold]
            group['a'] = abs(group.y_pred_a-group.y_true_a)
            group['b'] = abs(group.y_pred_b-group.y_true_b)
            if region == 'log1p':
                group['a'] = abs(np.log1p(group.y_pred_a)-np.log1p(group.y_true_a))
                group['b'] = abs(np.log1p(group.y_pred_b)-np.log1p(group.y_true_b))
            cells = group.groupby(['split_seed', 'station', 'cell'], sort=True)[['a', 'b']].mean().reset_index()
            names, parts = sorted(cells.station.unique()), sorted(cells.split_seed.unique())
            index = {name: i for i, name in enumerate(names)}
            matrices = np.zeros((3, len(names), len(parts)))
            for j, partition in enumerate(parts):
                for row in cells[cells.split_seed.eq(partition)].itertuples():
                    i = index[row.station]
                    matrices[0, i, j] += 1
                    matrices[1, i, j] += row.a
                    matrices[2, i, j] += row.b
            candidate_mae = np.mean(matrices[1].sum(0)/matrices[0].sum(0))
            reference_mae = np.mean(matrices[2].sum(0)/matrices[0].sum(0))
            rng = np.random.default_rng(42)
            gains, deltas = [], []
            rejected = 0
            while len(gains) < 5000:
                weights = rng.multinomial(len(names), np.full(len(names), 1/len(names)), size=min(512, 5000-len(gains)))
                counts = weights@matrices[0]
                for k in range(len(weights)):
                    if not (counts[k] > 0).all():
                        rejected += 1
                        continue
                    ca = np.mean((weights[k]@matrices[1])/counts[k])
                    re = np.mean((weights[k]@matrices[2])/counts[k])
                    gains.append(100*(re-ca)/re)
                    deltas.append(ca-re)
            result = effects[effects.candidate.eq(candidate) & effects.reference.eq(reference) & effects.region.eq(region)].iloc[0]
            np.testing.assert_allclose([candidate_mae, reference_mae], [result.candidate_mae, result.reference_mae], rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(np.quantile(gains, [.025, .975]), [result.gain_ci_low_pct, result.gain_ci_high_pct], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(np.quantile(deltas, [.025, .975]), [result.delta_ci_low, result.delta_ci_high], rtol=2e-10, atol=2e-10)
            assert rejected == result.bootstrap_empty_split_redraws
            checks.append({'candidate': candidate, 'reference': reference, 'region': region, 'draws': 5000,
                'station_ci_recalculated': True, 'n_stations': len(names), 'redraws': rejected})
    write_json(ROOT/'verification/analysis.json', {'all_model_mae_log_q90': 'independently recalculated', 'comparisons': checks})
    print('All model metrics and twelve primary paired station intervals independently verified.')


if __name__ == '__main__':
    main()
