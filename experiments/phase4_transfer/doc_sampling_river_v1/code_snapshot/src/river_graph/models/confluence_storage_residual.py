"""Sparse confluence mixing and storage priors on upstream DOC innovations.

These are information-mixing operators in log1p residual space, not a DOC
mass balance. Monthly lag slots represent memory rather than travel time.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from river_graph.models.dynamic_river_residual import DynamicRiverResidual
from river_graph.models.river_structure_residual import LAGS, upstream_paths


def confluence_inputs(messages, cells, bank_names, edges, physical, flow, flow_valid):
    """Keep the nearest valid donor along each represented upstream branch.

    A nested ancestor is suppressed only if a closer downstream donor has a
    usable DOC innovation at that lag. This avoids counting nested catchments
    as independent tributary contributions. Use measured nonnegative monthly
    discharge for a whole candidate group when all its flows are present and
    their sum is positive; otherwise use drainage area for the whole group.
    Signed reversing discharge triggers the area fallback, never abs(flow).
    """
    names, cells = np.asarray(bank_names, str), np.asarray(cells)
    flow, flow_valid = np.asarray(flow, float), np.asarray(flow_valid)
    if (flow.ndim != 2 or flow.shape != flow_valid.shape or flow.shape[0] != len(names)
            or flow_valid.dtype != np.bool_ or cells.ndim != 1 or cells.dtype.kind not in 'iu'):
        raise ValueError('aligned source flow/mask and integer receiver cells required')
    if not np.isfinite(flow[flow_valid]).all():
        raise ValueError('visible discharge must be finite')
    t = flow.shape[1]
    owners = np.asarray(messages['river_owner'])[cells//t]
    valid = np.asarray(messages['river_valid'])[cells//t, cells % t]
    if (owners < -1).any() or (owners >= len(names)).any():
        raise ValueError('source owner outside flow bank')
    if (valid & (owners < 0)[..., None]).any():
        raise ValueError('valid slot without a source station')
    ancestors = upstream_paths(edges, physical, names, names,
        candidates=max(1, len(names)), max_km=np.inf)
    ancestor_sets = [{s for s, *_ in row} for row in ancestors]
    # Relation[a,b] means a is upstream of b; it is independent of DOC values.
    relation = np.zeros((len(names), len(names)), bool)
    for b, group in enumerate(ancestor_sets):
        relation[:, b] = np.isin(names, list(group))
    nested = relation[owners.clip(min=0)[:, :, None], owners.clip(min=0)[:, None, :]]
    nested &= (owners >= 0)[:, :, None] & (owners >= 0)[:, None, :]
    suppressed = (nested[..., None] & valid[:, None]).any(axis=2)
    frontier = valid & ~suppressed
    dates = cells[:, None, None] % t-np.asarray(LAGS)[None, None]
    if (frontier & (dates < 0)).any():
        raise ValueError('pre-calendar source cannot be a valid candidate')
    donor_flow = flow[owners.clip(min=0)[..., None], dates.clip(min=0)]
    measured = flow_valid[owners.clip(min=0)[..., None], dates.clip(min=0)] & (donor_flow >= 0)
    use_flow = ((~frontier | measured).all(axis=1)
        & (np.where(frontier, donor_flow, 0.).sum(axis=1) > 0) & frontier.any(axis=1))
    area = np.expm1(np.asarray(messages['river_path'])[cells//t, :, 7]*np.log1p(1e6))
    weight = np.where(use_flow[:, None], donor_flow, area[..., None])
    weight = np.where(frontier, np.maximum(weight, 1e-12), 0.)
    prior = weight/np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return {'frontier': frontier, 'mix_prior': prior.astype(np.float32),
        'measured_flow': np.broadcast_to(use_flow[:, None], valid.shape).copy() & frontier,
        'nested_suppressed': valid & ~frontier}


class ConfluenceStorageResidual(DynamicRiverResidual):
    """Same local anchor, with optional explicit mixing/buffering operators.

    Path attributes remain available to the ordinary attention encoder in all
    arms. Ablations test their explicit operator use, not total availability.
    """

    def __init__(self, query_dim, feature_dim, *, operator='storage', **kwargs):
        if operator not in ('plain', 'mixing', 'storage'):
            raise ValueError('operator must be plain, mixing or storage')
        super().__init__(query_dim, feature_dim, **kwargs)
        self.config['operator'] = operator
        # Identical allocated capacity in all arms; disabled priors use zero.
        self.attenuation_raw = nn.Parameter(torch.full((self.config['heads'], 3), -2.))
        self.delay_raw = nn.Parameter(torch.full((self.config['heads'],), -2.))

    def forward(self, query, features, values, valid, frontier, mix_prior,
                measured_flow, nested_suppressed):
        if (frontier.shape != valid.shape or frontier.dtype != torch.bool
                or mix_prior.shape != valid.shape or measured_flow.shape != valid.shape
                or nested_suppressed.shape != valid.shape or (frontier & ~valid).any()
                or not torch.isfinite(mix_prior).all() or (mix_prior < 0).any()
                or torch.count_nonzero(mix_prior[~frontier])):
            raise ValueError('finite aligned branch-frontier mixing priors required')
        if (query.ndim != 2 or features.shape[:3] != values.shape or values.shape != valid.shape
                or features.shape[-1] != self.config['feature_dim'] or valid.dtype != torch.bool
                or query.shape != (len(values), self.config['query_dim'])
                or values.shape[-1] != len(LAGS)):
            raise ValueError('aligned sparse upstream tensors required')
        if not all(torch.isfinite(x).all() for x in (query, features, values)) or torch.count_nonzero(values[~valid]):
            raise ValueError('visible finite inputs and zero hidden source values required')
        operator = self.config['operator']
        allowed = valid if operator == 'plain' else frontier
        b, c, l = values.shape
        h, d = self.config['heads'], self.config['dimensions']
        encoded = self.encoder(features)
        keys = self.key(encoded).reshape(b, c, l, h, d)
        queries = self.query(query).reshape(b, h, d)
        scores = torch.einsum('bhd,bclhd->bhcl', queries, keys)/np.sqrt(d)
        if operator != 'plain':
            scores = scores+mix_prior.clamp_min(1e-12).log()[:, None]
        # Distance / mapped storage fraction / mapped confluences.
        physical = torch.stack([features[..., 0], features[..., 2]+features[..., 6], features[..., 1]], -1)
        attenuation = torch.exp(-torch.einsum('hp,bclp->bhcl', F.softplus(self.attenuation_raw), physical))
        lag = values.new_tensor(LAGS)/max(LAGS)
        if operator == 'storage':
            scores = scores+F.softplus(self.delay_raw)[None, :, None, None]*(features[..., 2]+features[..., 6])[:, None]*lag
        else:
            attenuation = attenuation*0+1.
        scores = scores.reshape(b, h, c*l).masked_fill(~allowed.reshape(b, 1, c*l), -torch.inf)
        weights = torch.softmax(torch.cat([scores, scores.new_zeros((b, h, 1))], -1), -1)
        allocated = weights[..., :-1].reshape(b, h, c, l)
        gate = torch.sigmoid(self.reliability(encoded)).permute(0, 3, 1, 2)
        basis = torch.stack([values, values.clamp_min(0), values.clamp_max(0)], -1)
        state = (allocated[..., None]*gate[..., None]*attenuation[..., None]*basis[:, None]).sum((2, 3))
        delta = self.output(state.flatten(1)).squeeze(-1)
        mass = allocated.sum((1, 2, 3))
        diagnostics = {'prior_mass': weights[..., -1].mean(-1),
            'entropy': -(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1),
            'lag_mass': allocated.sum((1, 2))/h, 'support': allowed.any((1, 2)), 'delta_log': delta,
            'frontier_count': frontier.sum(1).float().mean(-1),
            'nested_removed': nested_suppressed.sum((1, 2)).float(),
            'measured_flow_mass': (allocated*measured_flow[:, None]).sum((1, 2, 3))/mass.clamp_min(1e-12),
            'attenuation_mean': (allocated*attenuation).sum((1, 2, 3))/mass.clamp_min(1e-12),
            'storage_mass': (allocated*(features[..., 2]+features[..., 6])[:, None]).sum((1, 2, 3))/mass.clamp_min(1e-12)}
        return delta, diagnostics

    @staticmethod
    def _arrays(arrays):
        result = DynamicRiverResidual._arrays(arrays)
        for key in ('frontier', 'mix_prior', 'measured_flow', 'nested_suppressed'):
            result[key] = torch.as_tensor(arrays[key], dtype=torch.float32 if key == 'mix_prior' else torch.bool)
        return result
