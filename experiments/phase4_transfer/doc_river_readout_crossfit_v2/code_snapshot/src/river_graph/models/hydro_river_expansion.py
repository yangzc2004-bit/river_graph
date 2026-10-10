"""Expand real river messages with covariate-only stations, never chemistry.

This reuses the retained encoder's preprocessing: source-cell monthly hydro
statistics and historical all-cohort static/ecology statistics. Statistics are
saved once and subsequently fixed. The donor pool and DOC experience library
are different objects; this module supplies no water-quality values or masks.
"""
from __future__ import annotations

import numpy as np
import torch

from river_graph.models.dynamic_river_state import environmental_view


class FrozenHydroPreprocessing:
    """Reconstruct the existing model's environmental input on every node."""

    def __init__(self, statistics):
        self.statistics = statistics

    @classmethod
    def fit(cls, dataset, source_cells):
        x = torch.as_tensor(dataset['x'], dtype=torch.float32)
        n, t, _ = x.shape
        cells = np.asarray(source_cells, np.int64)
        if cells.ndim != 1 or not len(cells) or (cells < 0).any() or (cells >= n*t).any():
            raise ValueError('nonempty valid source cells required')
        static = torch.as_tensor(dataset['static'], dtype=torch.float32)
        regime = torch.as_tensor(dataset['regime'], dtype=torch.float32)
        statistics = {}
        for name, references in (('hydro', [x[..., c][cells//t, cells % t] for c in range(2)]),
                                 ('static', [static[:, c:c+1].reshape(-1) for c in range(2)])):
            # The historical implementation reduces individual vectors, not
            # columns of a matrix. Preserve float32 reduction order too.
            statistics[name] = {'mean': [v.mean().item() for v in references],
                'sd': [v.std(unbiased=False).clamp_min(1e-8).item() for v in references]}
        statistics['regime'] = {'mean': regime.mean(0).tolist(),
            'sd': regime.std(0, unbiased=False).clamp_min(1e-8).tolist()}
        return cls(statistics)

    def transform(self, dataset, daily):
        x = torch.as_tensor(dataset['x'], dtype=torch.float32)
        masks = torch.as_tensor(dataset['x_mask'], dtype=torch.float32)
        n, t, channels = x.shape
        daily = np.asarray(daily, np.float32)
        if channels != 2 or masks.shape != x.shape or daily.shape != (n, t, 8):
            raise ValueError('aligned two-channel hydro and eight-channel daily input required')

        def normalize(value, name):
            stats = self.statistics[name]
            return (torch.as_tensor(value, dtype=torch.float32)-torch.tensor(stats['mean']))/torch.tensor(stats['sd'])

        hydro = normalize(x, 'hydro')
        static = normalize(dataset['static'], 'static')
        regime = normalize(dataset['regime'], 'regime')
        months = torch.tensor([int(str(m)[5:7]) for m in dataset['months']], dtype=torch.float32)
        angle = 2*torch.pi*(months-1)/12
        raw = torch.zeros((n, t, 23), dtype=torch.float32)
        raw[..., :4] = torch.stack([hydro[..., 0], masks[..., 0], hydro[..., 1], masks[..., 1]], -1)
        raw[..., 4], raw[..., 5] = torch.sin(angle), torch.cos(angle)
        raw[..., 6:8] = static[:, None]
        raw[..., 10:14] = regime[:, None, :4]
        arrays = {'raw': raw.numpy(), 'env': regime[:, 4:].numpy(),
            'age': np.zeros((n, t)), 'support': np.zeros((n, t, 3)),
            'extra': np.zeros((n, t, 41))}
        view = environmental_view(arrays)
        view['extra'][..., 30:38] = daily
        if not all(np.isfinite(value).all() for value in view.values()):
            raise ValueError('nonfinite environmental covariates')
        return view


def eligible_hydro_nodes(n_nodes, receivers):
    """Exclude the complete receiving fold/role, with no label access."""
    receiving = np.asarray(receivers, np.int64)
    if (receiving < 0).any() or (receiving >= n_nodes).any():
        raise ValueError('receiver outside covariate atlas')
    return np.setdiff1d(np.arange(n_nodes), receiving)
