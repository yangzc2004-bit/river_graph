"""Station-held readout errors conditional on an existing frozen DOC model.

This is deliberately NOT complete-model OOF: encoder, GRU, donor allocation,
donor features and ecological coefficients retain their historical source fits.
Only the newly fitted scalar readout excludes the complete receiving fold.
The paired fitted-readout control uses exactly the same fixed features/budget.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def readout_design(model, inputs, cells):
    """Export the actual retained head design, including frozen donor states."""
    arrays = model._prepare_inputs(inputs)
    result = []
    with torch.no_grad():
        for start in range(0, len(cells), model.batch_size):
            at = torch.as_tensor(cells[start:start+model.batch_size], dtype=torch.long)
            hidden = model._hidden_cells(arrays, at)
            t = arrays['age'].shape[1]
            extra = arrays['extra'][at//t, at % t]
            pieces = [hidden, extra]
            if model.interaction_indices:
                pieces.append((hidden[..., None]*extra[:, model.interaction_indices][:, None]).flatten(1))
            state, _ = model._attention_cells(arrays, at, hidden)
            pieces.append(state)
            result.append(torch.cat(pieces, -1).numpy())
    return np.concatenate(result).astype(np.float32, copy=False)


def combine_readout(design, head, base, coefficient):
    """Fixed memory/tree offset plus the scaled native-unit scalar readout."""
    x = torch.as_tensor(design, dtype=torch.float32)
    with torch.no_grad():
        delta = head(x).squeeze(-1).double().numpy()
    prediction = np.maximum(0., np.asarray(base, float)+float(coefficient)*delta)
    if not np.isfinite(prediction).all():
        raise FloatingPointError('nonfinite conditional readout')
    return prediction


def fit_readout(design, base, y, train_rows, *, coefficient=1., seed=42,
                epochs=30, batch_size=512, learning_rate=.001, tail_threshold=10.):
    """Fit a fresh zero readout using only explicitly permitted label rows.

    Epoch count is fixed: held-fold labels and the evaluated validation labels
    are not used to choose its epoch, scale or hyperparameters.
    """
    rows = np.asarray(train_rows)
    design, base, y = np.asarray(design), np.asarray(base), np.asarray(y)
    if (design.ndim != 2 or base.shape != (len(design),) or y.shape != base.shape
            or rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
            or len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(y)).any()
            or not np.isfinite(design).all() or not np.isfinite(base).all()
            or not np.isfinite(y[rows]).all() or (y[rows] < 0).any()
            or not 0 < coefficient <= 1 or epochs < 0):
        raise ValueError('finite fixed features and unique permitted fitting rows required')
    # Select labels BEFORE constructing tensors or computing a loss/weight.
    x = torch.as_tensor(design[rows], dtype=torch.float32)
    target, anchor = torch.tensor(y[rows], dtype=torch.float64), torch.tensor(base[rows], dtype=torch.float64)
    weights = torch.tensor(np.where(y[rows] >= tail_threshold, 2., 1.), dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = nn.Linear(design.shape[1], 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    trace = []
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(rows))
        total = 0.
        for start in range(0, len(rows), batch_size):
            at = order[start:start+batch_size]
            optimizer.zero_grad()
            prediction = (anchor[at]+coefficient*head(x[at]).squeeze(-1).double()).clamp(min=0)
            error = weights[at]*(prediction-target[at]).abs()
            loss = error.mean()/weights.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite conditional fitting loss')
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(error.detach().sum())
        trace.append({'epoch': epoch, 'training_weighted_mae': total/float(weights.sum())})
    return head, trace


def conditional_crossfit(design, base, y, cells, folds, *, n_months, **kwargs):
    """Exclude every station in each fold from the new readout loss."""
    cells = np.asarray(cells)
    ids = np.unique(cells//n_months)
    folds = [np.asarray(fold, dtype=np.int64) for fold in folds]
    if (len(folds) < 2 or not np.array_equal(np.sort(np.concatenate(folds)), ids)
            or len(cells) != len(y)):
        raise ValueError('disjoint station folds must partition the source rows')
    prediction = np.empty(len(cells), dtype=float)
    heads, records = [], []
    for i, held in enumerate(folds):
        query = np.flatnonzero(np.isin(cells//n_months, held))
        train = np.flatnonzero(~np.isin(cells//n_months, held))
        head, trace = fit_readout(design, base, y, train, **kwargs)
        prediction[query] = combine_readout(design[query], head, np.asarray(base)[query], kwargs.get('coefficient', 1.))
        heads.append({k: value.detach().clone() for k, value in head.state_dict().items()})
        records.append({'fold': i, 'held_station_ids': held.tolist(),
            'fitted_station_ids': np.unique(cells[train]//n_months).tolist(),
            'query_rows': query.tolist(), 'train_rows': train.tolist(), 'trace': trace,
            'readout_loss_is_station_held': True, 'complete_model_is_oof': False})
    return prediction, heads, records
