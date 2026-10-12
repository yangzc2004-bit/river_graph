"""Label-free upstream environmental states on a frozen DOC predictor."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from river_graph.models.dynamic_river_residual import DynamicRiverResidual
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.river_structure_residual import (
    LAGS,
    path_candidates,
    upstream_paths,
)


def environmental_view(inputs):
    """Remove every DOC value, visibility, age and network-support channel.

    Monthly temperature/flow, their masks, calendar, coordinates and ecology
    retain the original source preprocessing. Extra readout features are not
    passed into the hidden encoder; no chemistry target cache is consumed.
    """
    view = {key: np.asarray(value).copy() for key, value in inputs.items()}
    raw = view['raw']
    if raw.ndim != 3 or raw.shape[-1] != 23:
        raise ValueError('this retained M1 encoder requires 14 base + 9 observation channels')
    raw[..., 8:10] = 0.
    raw[..., 14:] = 0.
    age = np.log1p(np.arange(raw.shape[1])+1)/np.log1p(12)
    raw[..., 15] = age[None, :]
    view['age'] = np.broadcast_to(age, raw.shape[:2]).copy()
    view['support'] = np.zeros((*raw.shape[:2], 3), dtype=raw.dtype)
    # Extra contains contextual DOC predictions. The hidden encoder ignores it
    # in this retained model, but zero it explicitly rather than rely on that.
    view['extra'] = np.zeros_like(view['extra'])
    return view


def hydro_history_valid(raw, daily):
    """At least one measured hydro input within the causal 12-month window."""
    raw, daily = np.asarray(raw), np.asarray(daily)
    observed = (raw[..., 1] > 0) | (raw[..., 3] > 0) | (daily[..., 5:8] > 0).any(-1)
    cumulative = np.concatenate([np.zeros((len(raw), 1), int), observed.cumsum(1)], 1)
    right = np.arange(raw.shape[1])+1
    return cumulative[:, right]-cumulative[:, np.maximum(0, right-12)] > 0


def environmental_states(model, inputs, *, batch=512, progress=None):
    """Reuse the frozen self/ecology encoder and causal GRU, without DOC."""
    if model.hydro_projection is not None or model.reference_projection is not None:
        raise ValueError('unexpected historical projection: define its visibility before use')
    view = environmental_view(inputs)
    # Bypass the downstream retrieval readout's donor-array requirements. This
    # calls the existing local encoder preparation, not a fabricated library.
    prepared = EncoderNativeResidual._prepare_inputs(model, view)
    n, t = view['age'].shape
    state = np.empty((n*t, model.hidden_size), np.float32)
    with torch.no_grad():
        for start in range(0, n*t, batch):
            cells = np.arange(start, min(start+batch, n*t))
            state[cells] = model._hidden_cells(prepared, cells).numpy()
            if progress is not None and start % (batch*40) == 0:
                progress({'encoded_cells': int(start+len(cells)), 'total_cells': n*t})
    daily = np.asarray(inputs['extra'])[..., 30:38]
    previous = np.concatenate([np.zeros_like(daily[:, :1]), daily[:, :-1]], 1)
    values = np.concatenate([state.reshape(n, t, -1),
        np.broadcast_to(view['env'][:, None], (n, t, view['env'].shape[-1])),
        daily, daily-previous, view['raw'][..., :4]], -1)
    return values.astype(np.float32), hydro_history_valid(view['raw'], daily)


def state_paths(edges, physical, receivers, bank_names, available, *, candidates=20):
    """Real paths and non-ancestor controls matched on hydro availability.

    This matcher reads no DOC. Complete ancestor and same-COMID exclusion
    includes paths beyond the 3000-km primary message cap.
    """
    names, receivers = np.asarray(bank_names, str), np.asarray(receivers, str)
    available = np.asarray(available, bool)
    real = path_candidates(edges, physical, receivers, names,
        np.full(available.shape, np.nan), candidates=candidates)
    owners = real['river_owner'].copy()
    attributes = physical.set_index('station')
    attributes.index = attributes.index.astype(str)
    area = np.log1p(attributes.loc[names, 'drainage_area_km2'].to_numpy(float))
    comids = attributes.loc[names, 'comid'].to_numpy()
    ancestors = upstream_paths(edges, physical, receivers, names, candidates=len(names), max_km=np.inf)
    records = []
    for row, receiver in enumerate(receivers):
        excluded = {receiver, *[x[0] for x in ancestors[row]]}
        aliases = set(attributes.loc[list(excluded), 'comid'])
        pool = [i for i, name in enumerate(names) if name not in excluded and comids[i] not in aliases]
        for slot, original in enumerate(real['river_owner'][row]):
            if original < 0:
                continue
            if not pool:
                raise ValueError('insufficient matched non-ancestor state donors')
            score = [abs(area[i]-area[original])+2*(1-(available[i] & available[original]).sum()
                        /max(1, (available[i] | available[original]).sum())) for i in pool]
            donor = pool.pop(int(np.argmin(score)))
            owners[row, slot] = donor
            records.append({'receiver': receiver, 'slot': slot, 'real_source': names[original],
                'control_source': names[donor], 'source_is_nonancestor': True,
                'source_log_area_difference': float(abs(area[donor]-area[original]))})
    return real, {**real, 'river_owner': owners}, records


def state_cell_features(paths, cells, receiver_hydro, bank_hydro, states, available):
    """Gather environmental values at allowed upstream dates; never future."""
    cells = np.asarray(cells, np.int64)
    n, t, _ = np.shape(receiver_hydro)
    if cells.ndim != 1 or (cells < 0).any() or (cells >= n*t).any():
        raise ValueError('receiving cells outside state grid')
    row, month = cells//t, cells % t
    owners = np.asarray(paths['river_owner'])[row]
    if (owners < -1).any() or (owners >= len(states)).any():
        raise ValueError('state source outside environmental bank')
    dates = month[:, None, None]-np.asarray(LAGS)[None, None]
    donor = owners.clip(min=0)[..., None]
    valid = (owners[..., None] >= 0) & (dates >= 0) & np.asarray(available)[donor, dates.clip(min=0)]
    hydro = np.asarray(bank_hydro)[donor, dates.clip(min=0)]
    previous = np.asarray(bank_hydro)[donor, (dates-1).clip(min=0)]
    previous = np.where((dates >= 1)[..., None], previous, 0.)
    current = np.asarray(receiver_hydro)[row, month]
    path = np.broadcast_to(paths['river_path'][row, :, None], (*valid.shape, 8))
    lag = np.broadcast_to(np.eye(len(LAGS))[None, None], (*valid.shape, len(LAGS)))
    features = np.concatenate([path, hydro, hydro-previous, hydro-current[:, None, None],
        np.zeros((*valid.shape, 1)), lag], -1)
    values = np.asarray(states)[donor, dates.clip(min=0)]
    return {'features': np.where(valid[..., None], features, 0.).astype(np.float32),
        'states': np.where(valid[..., None], values, 0.).astype(np.float32), 'valid': valid}


class DynamicRiverState(DynamicRiverResidual):
    """Sparse upstream latent-state messages, distinct from observed DOC."""

    def __init__(self, query_dim, feature_dim, state_dim, *, allocation='dynamic', **kwargs):
        if allocation not in ('dynamic', 'uniform'):
            raise ValueError('dynamic or uniform state allocation required')
        super().__init__(query_dim, feature_dim, **kwargs)
        self.config.update({'state_dim': state_dim, 'allocation': allocation})
        h, d = self.config['heads'], self.config['dimensions']
        with torch.random.fork_rng():
            torch.manual_seed(self.config['seed']+9321)
            self.value = nn.Sequential(nn.Linear(state_dim, d, bias=False), nn.Tanh())
            self.output = nn.Linear(h*d, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _arrays(arrays):
        return {key: torch.as_tensor(arrays[key], dtype=torch.bool if key == 'valid'
                else torch.float32) for key in ('query', 'features', 'states', 'valid')}

    def forward(self, query, features, states, valid):
        if (features.ndim != 4 or states.shape[:3] != features.shape[:3]
                or valid.shape != features.shape[:3] or valid.dtype != torch.bool
                or query.shape != (len(valid), self.config['query_dim'])
                or states.shape[-1] != self.config['state_dim']
                or features.shape[-1] != self.config['feature_dim'] or valid.shape[-1] != len(LAGS)):
            raise ValueError('aligned state, query, path and boolean validity required')
        if not all(torch.isfinite(x).all() for x in (query, features, states)):
            raise ValueError('nonfinite environmental state')
        if torch.count_nonzero(states[~valid]) or torch.count_nonzero(features[~valid]):
            raise ValueError('invalid environmental candidates must be zero')
        b, c, l = valid.shape
        h, d = self.config['heads'], self.config['dimensions']
        encoded = self.encoder(features)
        keys = self.key(encoded).reshape(b, c*l, h, d)
        queries = self.query(query).reshape(b, h, d)
        scores = torch.einsum('bhd,bchd->bhc', queries, keys)/np.sqrt(d)
        if self.config['allocation'] == 'uniform':
            scores = scores*0
        scores = scores.masked_fill(~valid.reshape(b, 1, c*l), -torch.inf)
        weights = torch.softmax(torch.cat([scores, scores.new_zeros((b, h, 1))], -1), -1)
        allocated = weights[..., :-1].reshape(b, h, c, l)
        reliability = torch.sigmoid(self.reliability(encoded)).permute(0, 3, 1, 2)
        if self.config['allocation'] == 'uniform':
            reliability = reliability*0+1.
        message = (allocated[..., None]*reliability[..., None]*self.value(states)[:, None]).sum((2, 3))
        delta = self.output(message.flatten(1)).squeeze(-1)
        return delta, {'prior_mass': weights[..., -1].mean(-1),
            'entropy': -(weights*weights.clamp(min=1e-12).log()).sum(-1).mean(-1),
            'lag_mass': allocated.sum((1, 2))/h, 'support': valid.any((1, 2)), 'delta_log': delta}
