"""Recover actual source-date support for frozen ST357 monthly DOC observations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import bind_files, verify_files, write_json

from river_graph.analysis.river_sampling_resolution import load_station_daily_flow
from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file
from river_graph.models.sampling_aware_river import monthly_sampling_metadata

ROOT = Path('experiments/phase4_transfer/doc_sampling_river_v1')
DATASET = Path('data/processed/mississippi_graph_graphfix_st357.pt')


def prepare_metadata(root=ROOT):
    if (root/'metadata_complete.json').exists():
        definition = json.loads((root/'metadata_definition.json').read_text())
        verify_files(root, 'metadata_complete.json', definition)
        for path, checksum in definition['sources'].items():
            if sha256_file(path) != checksum:
                raise ValueError('sampling raw data changed')
        return
    root.mkdir(parents=True, exist_ok=True)
    data = torch.load(DATASET, weights_only=False, map_location='cpu')
    sites, months = np.asarray(data['site_no'], str), pd.DatetimeIndex(data['months'])
    rows, sources = [], {str(DATASET): sha256_file(DATASET)}
    for site in sites:
        path = Path('data/raw/wqp_results')/f'{site}.csv'
        raw = load_station_results(path)
        obs = extract_doc_obs(raw)
        rows.append(obs[['site_no', 'date', 'doc']])
        sources[str(path)] = sha256_file(path)
    samples = pd.concat(rows, ignore_index=True)
    samples['month'] = samples.date.dt.to_period('M')
    station_index = {s: i for i, s in enumerate(sites)}
    month_index = {m: i for i, m in enumerate(months.to_period('M'))}
    samples = samples[samples.month.isin(month_index)].copy()
    samples['cell'] = samples.site_no.map(station_index).astype(np.int64)*len(months)+samples.month.map(month_index).astype(np.int64)
    monthly = samples.groupby('cell', as_index=False).agg(raw_doc=('doc', 'mean'), n_results=('doc', 'size'),
        n_sample_days=('date', 'nunique'), first_date=('date', 'min'), last_date=('date', 'max'))
    observed = np.flatnonzero(np.asarray(data['y_mask']).ravel())
    selected = monthly.set_index('cell').reindex(observed)
    if selected.raw_doc.isna().any():
        raise ValueError('frozen DOC cells missing raw sampling dates')
    selected['frozen_doc'] = np.asarray(data['y']).ravel()[observed]
    np.testing.assert_allclose(selected.raw_doc, selected.frozen_doc, atol=2e-5, rtol=1e-6)
    selected.reset_index().to_csv(root/'monthly_reconciliation.csv', index=False)
    print(f'Reconciled {len(observed)} frozen DOC cells across {len(sites)} stations.', flush=True)
    daily, inventory = load_station_daily_flow('data/raw/nwis_dv', set(sites))
    sources.update({row['path']: row['sha256'] for row in inventory['files']})
    meta, receiver, ends = monthly_sampling_metadata(samples, sites, months, np.asarray(data['x'])[..., 1],
        np.asarray(data['x_mask'])[..., 1], daily)
    np.savez_compressed(root/'sampling_metadata.npz', metadata=meta, receiver_phase=receiver, month_ends=ends)
    definition = {'sources': sources, 'monthly_doc': 'unchanged result-row arithmetic mean',
        'prediction_reference': 'fixed calendar month-end; receiving sampling dates never used',
        'source_date': 'result-row weighted mean and latest date, span and support counts',
        'flow': 'reconciled NWIS daily cfs; no gap filling; negative/missing flow unavailable for phase ratios',
        'phase': 'sample and fixed month-end relative to exactly seven days earlier',
        'source_visibility': 'per-run training cells only; whole receiving fold excluded by bank owners',
        'n_stations': len(sites), 'n_observed_months': len(observed),
        'max_doc_reconstruction_difference': float(abs(selected.raw_doc-selected.frozen_doc).max()),
        'n_sample_rows': len(samples), 'daily_qc': inventory['quality_summary'],
        'known_source_flow_months': int((meta[..., 5] > 0).sum()),
        'known_source_phase_months': int((meta[..., 8] > 0).sum()),
        'known_receiver_phase_months': int(receiver[..., 1].sum())}
    write_json(root/'metadata_definition.json', definition)
    bind_files(root, 'metadata_complete.json', [root/name for name in
        ('sampling_metadata.npz', 'metadata_definition.json', 'monthly_reconciliation.csv')], definition)
    print(json.dumps({k: v for k, v in definition.items() if k not in ('sources', 'daily_qc')}), flush=True)


if __name__ == '__main__':
    prepare_metadata()
