"""Compare historical DOC RF predictions with the temporal paper query sets.

Read-only with respect to experiments: writes a manuscript context table and
source hashes, without promoting historical pilot results to confirmatory runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file


def main():
    root = Path('experiments/phase4_transfer/temporal_h2x_v1')
    old = Path('experiments/phase2_ablation_stcore_v1/predictions')
    out = Path('docs/paper/latex/supplementary')
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(root / 't8_synthesis/main_summary.csv')
    sources, rows = {}, []
    for row in summary[summary.analyte == 'doc'].itertuples():
        target = root / f'target_masks/doc__{row.mask}.npz'
        parent = Path(f'experiments/masks_stcore_v1/{row.mask}.npz')
        with np.load(target) as a, np.load(parent) as b:
            for role in ('train', 'val', 'test', 'context'):
                av = a[role] if role in a else np.array([], dtype=int)
                bv = b[role] if role in b else np.array([], dtype=int)
                np.testing.assert_array_equal(np.sort(av), np.sort(bv))
        maes = []
        for seed in range(42, 47):
            rf = old / f'P2X_eco_RF_s{seed}__{row.mask}.parquet'
            meta = rf.with_suffix('.meta.json')
            current = root / 't3_formal/runs' / (
                f'h2x_t__doc__{row.mask}__seed{seed}')
            temporal = current / 'full_grid.parquet'
            rm = json.loads(meta.read_text())
            tm = json.loads((current / 'meta.json').read_text())
            assert rm['dataset']['sha256'] == tm['dataset']['sha256']
            left, right = pd.read_parquet(rf), pd.read_parquet(temporal)
            left = left[left['split'] == 'test'].copy()
            role = 'split' if 'split' in right else 'visibility_role'
            right = right[right[role] == 'test'].copy()
            for frame in (left, right):
                frame['station'] = frame.station.astype(str)
                frame['month'] = pd.to_datetime(frame.month)
            joined = left.merge(right, on=['station', 'month'],
                                validate='one_to_one', suffixes=('_rf', '_t'))
            assert len(joined) == len(left) == len(right) == row.query_cells
            np.testing.assert_allclose(joined.y_true_rf, joined.y_true_t)
            mae = np.mean(np.abs(joined.y_pred_rf - joined.y_true_rf))
            np.testing.assert_allclose(mae, rm['metrics']['mae'], rtol=1e-6)
            maes.append(mae)
            for p in (rf, meta, temporal, current / 'meta.json'):
                sources[str(p)] = sha256_file(p)
        rows.append({'mask': row.mask, 'query_cells': row.query_cells, 'seeds': 5,
                     'historical_rf_mae': np.mean(maes),
                     'temporal_mae': row.temporal_mae,
                     'temporal_minus_rf': row.temporal_mae - np.mean(maes)})
        for p in (target, parent):
            sources[str(p)] = sha256_file(p)
    pd.DataFrame(rows).to_csv(out / 'historical_doc_rf_context.csv', index=False)
    sources[str(root / 't8_synthesis/main_summary.csv')] = sha256_file(
        root / 't8_synthesis/main_summary.csv')
    (out / 'historical_doc_rf_manifest.json').write_text(json.dumps({
        'role': 'Historical pilot context; not a new confirmatory comparison',
        'checks': ['dataset hash equality', 'all split role arrays equal',
                   'unique station-month query equality', 'label equality',
                   'RF MAE recomputed from predictions'],
        'source_sha256': sources,
    }, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
