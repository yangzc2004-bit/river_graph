"""A protected dynamic river correction on the retained DOC predictor.

The ecology/GRU/similarity model stays frozen. Its queried recurrent state
conditions sparse upstream attention; observed donor departures enter through
causal 0/1/3-month slots. No source observation means exactly the old prediction.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from river_graph.models.river_structure_residual import LAGS, PATH_DIM


def river_cell_features(messages, cells, receiver_hydro, bank_hydro):
    """Gather source hydro at t-lag and t-lag-1, without any future reads."""
    cells = np.asarray(cells)
    n, t, channels = np.shape(receiver_hydro)
    if channels != 8 or cells.ndim != 1 or cells.dtype.kind not in 'iu':
        raise ValueError('integer cells and eight daily-hydro channels required')
    if (cells < 0).any() or (cells >= n*t).any():
        raise ValueError('river cells outside receiver grid')
    row, month = cells//t, cells % t
    owners = np.asarray(messages['river_owner'])[row]
    dates = month[:, None, None]-np.asarray(LAGS)[None, None, :]
    if (owners < -1).any() or (owners >= len(bank_hydro)).any():
        raise ValueError('river owner outside bank')
    valid = np.asarray(messages['river_valid'])[row, month].copy()
    if (valid & ((owners < 0)[..., None] | (dates < 0))).any():
        raise ValueError('missing owner or pre-calendar lag cannot be visible')
    hydro = np.asarray(bank_hydro)[owners.clip(min=0)[..., None], dates.clip(min=0)]
    previous = np.asarray(bank_hydro)[owners.clip(min=0)[..., None], (dates-1).clip(min=0)]
    previous = np.where((dates >= 1)[..., None], previous, 0.)
    current = np.asarray(receiver_hydro)[row, month]
    receiver_previous = np.asarray(receiver_hydro)[row, (month-1).clip(min=0)]
    receiver_previous = np.where((month >= 1)[:, None], receiver_previous, 0.)
    path = np.broadcast_to(np.asarray(messages['river_path'])[row, :, None], (*valid.shape, PATH_DIM))
    age = np.asarray(messages['river_age'])[row, month]/13.
    lag = np.broadcast_to(np.eye(len(LAGS))[None, None], (*valid.shape, len(LAGS)))
    features = np.concatenate([path, hydro, hydro-previous,
        hydro-current[:, None, None], age[..., None], lag], -1)
    features = np.where(valid[..., None], features, 0.)
    values = np.asarray(messages['river_values'])[row, month]
    if np.any(values[~valid] != 0.) or not np.isfinite(features).all():
        raise ValueError('invalid sources must contain zero finite values')
    return {'features': features.astype(np.float32), 'values': values.astype(np.float32),
            'valid': valid, 'query_hydro': np.concatenate([current, current-receiver_previous], -1)}


class DynamicRiverResidual(nn.Module):
    """Two-head sparse river operator attached to a frozen complete model.

    fit uses source labels, while early stopping uses source-validation labels.
    These are development results. The retained complete base is fitted on
    source labels; its training predictions are not complete-model OOF results.
    Donor departures use separately verified double-held environmental OOF.
    """

    def __init__(self, query_dim, feature_dim, *, mode='dynamic_lagged', seed=42,
                 heads=2, dimensions=32, epochs=30, patience=5, batch_size=512,
                 learning_rate=1e-3, tail_weight=2.):
        super().__init__()
        if mode not in ('static_same_month', 'dynamic_same_month', 'dynamic_lagged'):
            raise ValueError('invalid river allocation mode')
        self.config = {'query_dim': query_dim, 'feature_dim': feature_dim, 'mode': mode,
            'seed': seed, 'heads': heads, 'dimensions': dimensions, 'epochs': epochs, 'patience': patience,
            'batch_size': batch_size, 'learning_rate': learning_rate, 'tail_weight': tail_weight}
        with torch.random.fork_rng():
            torch.manual_seed(seed+8217)
            self.query = nn.Linear(query_dim, heads*dimensions, bias=False)
            self.encoder = nn.Sequential(nn.Linear(feature_dim, dimensions), nn.Tanh())
            self.key = nn.Linear(dimensions, heads*dimensions, bias=False)
            self.reliability = nn.Linear(dimensions, heads)
            self.output = nn.Linear(heads*3, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, features, values, valid):
        if (query.ndim != 2 or features.ndim != 4 or features.shape[:3] != values.shape
                or values.shape != valid.shape or valid.dtype != torch.bool
                or features.shape[-1] != self.config['feature_dim']
                or query.shape != (len(values), self.config['query_dim'])
                or values.shape[-1] != len(LAGS)):
            raise ValueError('aligned finite query, edge/lag features and boolean validity required')
        if not all(torch.isfinite(x).all() for x in (query, features, values)):
            raise ValueError('nonfinite river inputs')
        if torch.count_nonzero(values[~valid]):
            raise ValueError('hidden source values must be zero')
        valid = valid.clone()
        if self.config['mode'] != 'dynamic_lagged':
            valid[..., 1:] = False
        b, c, l = values.shape
        h, d = self.config['heads'], self.config['dimensions']
        encoded = self.encoder(features)
        keys = self.key(encoded).reshape(b, c*l, h, d)
        queries = self.query(query).reshape(b, h, d)
        scores = torch.einsum('bhd,bchd->bhc', queries, keys)/np.sqrt(d)
        if self.config['mode'] == 'static_same_month':
            scores = scores*0
        scores = scores.masked_fill(~valid.reshape(b, 1, c*l), -torch.inf)
        # Explicit zero-message alternative keeps empty neighbourhoods finite.
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
            'lag_mass': allocated.sum((1, 2))/h, 'support': valid.any((1, 2)),
            'delta_log': delta}
        return delta, diagnostics

    @staticmethod
    def _arrays(arrays):
        required = ('query', 'features', 'values', 'valid')
        result = {key: torch.as_tensor(arrays[key], dtype=torch.bool if key == 'valid'
                                      else torch.float32) for key in required}
        return result

    def predict(self, arrays, base, *, diagnostics=False):
        arrays = self._arrays(arrays)
        base = np.asarray(base, float)
        if base.shape != (len(arrays['query']),) or not np.isfinite(base).all() or (base < 0).any():
            raise ValueError('nonnegative finite complete-base predictions required')
        chunks = {}
        with torch.no_grad():
            for start in range(0, len(base), self.config['batch_size']):
                _, diag = self(**{k: v[start:start+self.config['batch_size']] for k, v in arrays.items()})
                for key, values in diag.items():
                    chunks.setdefault(key, []).append(values.numpy())
        diag = {k: np.concatenate(v) for k, v in chunks.items()}
        prediction = np.maximum(0., np.expm1(np.log1p(base)+diag['delta_log'].astype(float)))
        # Preserve exact bits for every zero correction, including unsupported cells.
        prediction[diag['delta_log'] == 0] = base[diag['delta_log'] == 0]
        if not np.isfinite(prediction).all():
            raise FloatingPointError('nonfinite dynamic river prediction')
        return (prediction, diag) if diagnostics else prediction

    def fit(self, source, source_base, source_y, validation, validation_base, validation_y,
            *, tail_threshold, progress=None):
        arrays = self._arrays(source)
        y, base = np.asarray(source_y, float), np.asarray(source_base, float)
        vy, vb = np.asarray(validation_y, float), np.asarray(validation_base, float)
        if (y.shape != base.shape or vy.shape != vb.shape or len(y) != len(arrays['query'])
                or not all(np.isfinite(x).all() and (x >= 0).all() for x in (y, base, vy, vb))):
            raise ValueError('finite aligned source/validation labels and bases required')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        weights = torch.tensor(np.where(y >= tail_threshold, self.config['tail_weight'], 1.), dtype=torch.float64)
        truth, anchor = torch.tensor(y), torch.tensor(np.log1p(base))
        best, state, stale = np.inf, None, 0
        self.trace_, self.best_epoch_ = [], 0
        for epoch in range(self.config['epochs']+1):
            loss_sum = 0.
            if epoch:
                order = np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch])).permutation(len(y))
                for start in range(0, len(y), self.config['batch_size']):
                    rows = order[start:start+self.config['batch_size']]
                    optimizer.zero_grad()
                    delta, _ = self(**{k: v[rows] for k, v in arrays.items()})
                    predicted = torch.expm1(anchor[rows]+delta.double()).clamp(min=0)
                    error = weights[rows]*(predicted-truth[rows]).abs()
                    loss = error.mean()/weights.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite river training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(error.detach().sum())
            predicted = self.predict(validation, vb)
            mae = float(np.abs(predicted-vy).mean())
            tail = vy >= tail_threshold
            record = {'epoch': epoch, 'validation_mae': mae,
                'validation_q90_mae': float(np.abs(predicted[tail]-vy[tail]).mean()) if tail.any() else None,
                'training_weighted_mae': loss_sum/float(weights.sum()) if epoch else None}
            self.trace_.append(record)
            if mae < best-1e-12:
                best, state, stale = mae, copy.deepcopy(self.state_dict()), 0
                self.best_epoch_ = epoch
            else:
                stale += 1
            if progress:
                progress(record)
            if stale >= self.config['patience']:
                break
        self.load_state_dict(state)
        self.epochs_run_ = len(self.trace_)-1
        self.tail_threshold_ = float(tail_threshold)
        return self

    def to_dict(self):
        return {'version': 1, 'config': self.config.copy(), 'trace': copy.deepcopy(self.trace_),
                'best_epoch': self.best_epoch_, 'epochs_run': self.epochs_run_,
                'tail_threshold': self.tail_threshold_,
                'trainable_parameters': sum(p.numel() for p in self.parameters()),
                'base_frozen': True, 'complete_base_training_predictions_are_oof': False}

    def to_payload(self):
        return {'summary': self.to_dict(), 'state': copy.deepcopy(self.state_dict())}

    @classmethod
    def from_payload(cls, payload):
        saved = payload['summary']
        if saved.get('version') != 1:
            raise ValueError('unsupported dynamic river checkpoint')
        model = cls(**saved['config'])
        model.load_state_dict(payload['state'])
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite river checkpoint')
        model.trace_ = copy.deepcopy(saved['trace'])
        model.best_epoch_, model.epochs_run_ = saved['best_epoch'], saved['epochs_run']
        model.tail_threshold_ = saved['tail_threshold']
        return model
