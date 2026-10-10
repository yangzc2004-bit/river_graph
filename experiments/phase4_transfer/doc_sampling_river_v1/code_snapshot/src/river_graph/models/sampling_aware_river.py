"""Sampling-time support for monthly, causal upstream DOC innovations.

Dates describe the result-row weighted monthly observation, never a fictional
instantaneous concentration. Receiving sampling dates and labels are not inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from river_graph.models.dynamic_river_residual import DynamicRiverResidual
from river_graph.models.river_structure_residual import LAGS

TIMING_NAMES = ('mean_age_days', 'youngest_age_days', 'sample_span_days',
    'sample_day_count', 'result_row_count', 'sample_flow_fraction',
    'sample_vs_month_flow', 'sample_flow_phase', 'receiver_flow_phase',
    'source_receiver_phase_contrast', 'sample_phase_fraction', 'receiver_phase_valid')
TIMING_DIM = len(TIMING_NAMES)


def monthly_sampling_metadata(rows, sites, months, monthly_flow, flow_mask, daily):
    """Summarize dates/flow with the same row weights as frozen monthly DOC.

    This function never reads a DOC value. Flow pairs use exactly seven days
    before the actual source sample or the fixed receiving month-end. Missing,
    conflicting (already reconciled), and reversing flows remain unavailable.
    """
    sites = np.asarray(sites, str)
    months = pd.DatetimeIndex(months).to_period('M')
    n, t = len(sites), len(months)
    if np.shape(monthly_flow) != (n, t) or np.shape(flow_mask) != (n, t):
        raise ValueError('monthly hydro must align with station/month grid')
    if daily.duplicated(['site_no', 'date']).any():
        raise ValueError('reconciled unique station-day flow required')
    lookup = daily.set_index(['site_no', 'date']).discharge_cfs
    ends = months.to_timestamp(how='end').normalize().to_numpy(dtype='datetime64[D]').astype(np.int64)
    site_index, month_index = {s: i for i, s in enumerate(sites)}, {m: i for i, m in enumerate(months)}
    f = rows[['site_no', 'date']].copy()
    f['site_no'] = f.site_no.astype(str)
    f['date'] = pd.to_datetime(f.date).dt.normalize()
    f['month'] = f.date.dt.to_period('M')
    f = f[f.site_no.isin(site_index) & f.month.isin(month_index)].copy()
    f['cell'] = f.site_no.map(site_index).astype(np.int64)*t+f.month.map(month_index).astype(np.int64)
    f['day'] = f.date.to_numpy(dtype='datetime64[D]').astype(np.int64)
    f['q'] = lookup.reindex(pd.MultiIndex.from_arrays([f.site_no, f.date])).to_numpy()
    f['q7'] = lookup.reindex(pd.MultiIndex.from_arrays([f.site_no, f.date-pd.Timedelta(days=7)])).to_numpy()
    row, month = f.cell.to_numpy()//t, f.cell.to_numpy() % t
    mean_q = np.asarray(monthly_flow)[row, month]
    available = np.asarray(flow_mask, bool)[row, month]
    ok = available & np.isfinite(f.q) & (f.q >= 0) & np.isfinite(mean_q) & (mean_q >= 0)
    pair = ok & np.isfinite(f.q7) & (f.q7 >= 0)
    f['flow_valid'], f['phase_valid'] = ok, pair
    f['flow_change'] = np.where(ok, (f.q-mean_q)/(abs(f.q)+abs(mean_q)+1e-12), 0.)
    f['phase'] = np.where(pair, (f.q-f.q7)/(abs(f.q)+abs(f.q7)+1e-12), 0.)
    # Slots: mean day, latest day, span, unique days, result count, flow coverage,
    # row-weighted available flow ratio/phase, phase coverage, date valid.
    meta = np.zeros((n, t, 10), np.float64)
    if len(f):
        g = f.groupby('cell', sort=True).agg(mean_day=('day', 'mean'), latest=('day', 'max'),
            first=('day', 'min'), days=('day', 'nunique'), count=('day', 'size'),
            flow_fraction=('flow_valid', 'mean'), change=('flow_change', 'sum'),
            phase=('phase', 'sum'), phase_fraction=('phase_valid', 'mean'),
            flow_n=('flow_valid', 'sum'), phase_n=('phase_valid', 'sum'))
        cells = g.index.to_numpy()
        meta.reshape(-1, 10)[cells] = np.column_stack([g.mean_day, g.latest, g.latest-g['first'],
            g.days, g['count'], g.flow_fraction, g.change/g.flow_n.clip(lower=1),
            g.phase/g.phase_n.clip(lower=1), g.phase_fraction, np.ones(len(g))])
    receiving = np.zeros((n, t, 2), np.float32)
    for i, site in enumerate(sites):
        dates = pd.to_datetime(ends, unit='D')
        q = lookup.reindex(pd.MultiIndex.from_arrays([[site]*t, dates])).to_numpy()
        q7 = lookup.reindex(pd.MultiIndex.from_arrays([[site]*t, dates-pd.Timedelta(days=7)])).to_numpy()
        valid = np.asarray(flow_mask, bool)[i] & np.isfinite(q) & np.isfinite(q7) & (q >= 0) & (q7 >= 0)
        receiving[i, :, 0] = np.where(valid, (q-q7)/(abs(q)+abs(q7)+1e-12), 0.)
        receiving[i, :, 1] = valid
    if not np.isfinite(meta).all() or not np.isfinite(receiving).all():
        raise ValueError('sampling metadata must be finite')
    return meta, receiving, ends


def sampling_cell_features(messages, cells, metadata, receiver_phase, ends, visible):
    """Read only permitted donor months at each causal slot; no receiver dates.

    `visible` is source training visibility. Query-fold exclusion is separately
    enforced by the owners in the nested bank. True donor observation months are
    recomputed: the matched bank's deliberately common age is not its timestamp.
    """
    metadata = np.asarray(metadata)
    visible = np.asarray(visible, bool)
    n, t = np.shape(receiver_phase)[:2]
    cells = np.asarray(cells, np.int64)
    if metadata.shape != (*visible.shape, 10) or visible.shape[1] != t or len(ends) != t:
        raise ValueError('aligned sampling metadata and source visibility required')
    if (cells < 0).any() or (cells >= n*t).any():
        raise ValueError('receiving cells outside grid')
    rows, months = cells//t, cells % t
    owners = np.asarray(messages['river_owner'])[rows]
    valid = np.asarray(messages['river_valid'])[rows, months]
    if (owners < -1).any() or (owners >= len(metadata)).any():
        raise ValueError('donor owner outside source bank')
    last = np.maximum.accumulate(np.where(visible, np.arange(t)[None], -1), axis=1)
    slots = months[:, None, None]-np.asarray(LAGS)[None, None]
    observed = last[owners.clip(min=0)[..., None], slots.clip(min=0)]
    if (valid & ((owners < 0)[..., None] | (slots < 0) | (observed < 0))).any():
        raise ValueError('visible river value without a permitted source observation')
    selected = metadata[owners.clip(min=0)[..., None], observed.clip(min=0)]
    if (valid & (selected[..., 9] != 1)).any():
        raise ValueError('source observation has no reconciled sampling dates')
    cutoff = np.asarray(ends)[slots.clip(min=0)]
    if (valid & (selected[..., 1] > cutoff)).any():
        raise ValueError('future source sampling date')
    age = np.asarray(ends)[months, None, None]-selected[..., 0]
    young = np.asarray(ends)[months, None, None]-selected[..., 1]
    phase = np.asarray(receiver_phase)[rows, months]
    shape = valid.shape
    result = np.stack([np.clip(age/396., 0, 1), np.clip(young/396., 0, 1), selected[..., 2]/31.,
        np.minimum(1., np.log1p(selected[..., 3])/np.log(32)),
        np.minimum(1., np.log1p(selected[..., 4])/np.log(32)), selected[..., 5], selected[..., 6],
        selected[..., 7], np.broadcast_to(phase[:, None, None, 0], shape),
        (selected[..., 7]-phase[:, None, None, 0])/2, selected[..., 8],
        np.broadcast_to(phase[:, None, None, 1], shape)], -1)
    # A contrast is meaningful only where BOTH measured phase summaries exist.
    result[..., 9] *= (selected[..., 8] > 0) & (phase[:, None, None, 1] > 0)
    return np.where(valid[..., None], result, 0.).astype(np.float32)


def shuffle_hydro_timing(timing, valid, months, folds, seed):
    """Permute source flow metadata within cutoff-month/lag/query-fold pools.

    Age, concentration, receiver hydro and date support stay fixed. Every pool
    uses observations already available by the same cutoff; hidden query folds
    are never mixed. This tests correspondence, not physical source rewiring.
    """
    result = np.asarray(timing).copy()
    groups = np.column_stack([months, folds])
    rng = np.random.default_rng(seed+37191)
    for group in np.unique(groups, axis=0):
        rows = np.flatnonzero(np.all(groups == group, axis=1))
        for lag in range(len(LAGS)):
            r, c = np.where(valid[rows, :, lag])
            positions = rows[r]
            order = rng.permutation(len(r))
            for feature in (5, 6, 7, 10):
                result[positions, c, lag, feature] = timing[positions[order], c[order], lag, feature]
    result[..., 9] = (result[..., 7]-result[..., 8])/2
    result[..., 9] *= (result[..., 10] > 0) & (result[..., 11] > 0)
    return np.where(np.asarray(valid)[..., None], result, 0.).astype(np.float32)


class SamplingAwareRiverResidual(DynamicRiverResidual):
    """Existing sparse two-head GNN with an additive sampling-support encoder."""

    def __init__(self, query_dim, feature_dim, *, sampling_mode='hydro', timing_dim=TIMING_DIM, **kwargs):
        if sampling_mode not in ('none', 'age', 'hydro') or timing_dim != TIMING_DIM:
            raise ValueError('invalid sampling mode or timing dimension')
        super().__init__(query_dim, feature_dim, **kwargs)
        with torch.random.fork_rng():
            torch.manual_seed(self.config['seed']+37191)
            self.timing_encoder = nn.Linear(timing_dim, self.config['dimensions'], bias=False)
        nn.init.zeros_(self.timing_encoder.weight)
        self.config.update(sampling_mode=sampling_mode, timing_dim=timing_dim)

    @staticmethod
    def _arrays(arrays):
        result = DynamicRiverResidual._arrays(arrays)
        result['timing'] = torch.as_tensor(arrays['timing'], dtype=torch.float32)
        return result

    def forward(self, query, features, values, valid, timing):
        if timing.shape != (*valid.shape, TIMING_DIM) or not torch.isfinite(timing).all():
            raise ValueError('aligned finite sampling features required')
        if (query.ndim != 2 or features.ndim != 4 or features.shape[:3] != values.shape
                or values.shape != valid.shape or valid.dtype != torch.bool
                or features.shape[-1] != self.config['feature_dim']
                or query.shape != (len(values), self.config['query_dim'])):
            raise ValueError('aligned query and river features required')
        if not all(torch.isfinite(x).all() for x in (query, features, values)):
            raise ValueError('nonfinite river inputs')
        if torch.count_nonzero(values[~valid]) or torch.count_nonzero(timing[~valid]):
            raise ValueError('hidden river inputs must be zero')
        if self.config['mode'] != 'dynamic_lagged':
            valid = valid.clone()
            valid[..., 1:] = False
        selected = timing.clone()
        if self.config['sampling_mode'] == 'none':
            selected[:] = 0
        elif self.config['sampling_mode'] == 'age':
            selected[..., 5:] = 0
        b, c, l = values.shape
        h, d = self.config['heads'], self.config['dimensions']
        encoded = self.encoder(features)+self.timing_encoder(selected)
        keys = self.key(encoded).reshape(b, c*l, h, d)
        queries = self.query(query).reshape(b, h, d)
        scores = torch.einsum('bhd,bchd->bhc', queries, keys)/np.sqrt(d)
        if self.config['mode'] == 'static_same_month':
            scores = scores*0
        scores = scores.masked_fill(~valid.reshape(b, 1, c*l), -torch.inf)
        weights = torch.softmax(torch.cat([scores, scores.new_zeros((b, h, 1))], -1), -1)
        allocated = weights[..., :-1].reshape(b, h, c, l)
        gate = torch.sigmoid(self.reliability(encoded)).permute(0, 3, 1, 2)
        if self.config['mode'] == 'static_same_month':
            gate = gate*0+1.
        basis = torch.stack([values, values.clamp(min=0), values.clamp(max=0)], -1)
        state = (allocated[..., None]*gate[..., None]*basis[:, None]).sum((2, 3))
        delta = self.output(state.flatten(1)).squeeze(-1)
        diagnostics = {'prior_mass': weights[..., -1].mean(-1),
            'entropy': -(weights*weights.clamp(min=1e-12).log()).sum(-1).mean(-1),
            'lag_mass': allocated.sum((1, 2))/h, 'support': valid.any((1, 2)), 'delta_log': delta,
            'sample_age_mass_days': (allocated*timing[:, None, ..., 0]).sum((1, 2, 3))*396/h,
            'sample_phase_mass': (allocated*timing[:, None, ..., 10]).sum((1, 2, 3))/h,
            'sample_flow_mass': (allocated*timing[:, None, ..., 5]).sum((1, 2, 3))/h}
        return delta, diagnostics
