"""Joint source chemistry supervision of the existing DOC ecological GRU."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def normalized_source_targets(target, valid, *, shuffle=False, seed=42):
    """Standardize available source labels; unavailable values are never read."""
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if (target.ndim != 3 or target.shape[-1] != 2 or valid.shape != target.shape
            or not np.isfinite(target[valid]).all()):
        raise ValueError("two aligned finite source auxiliary targets are required")
    counts = valid.sum((0, 1))
    if not (counts > 0).all():
        raise ValueError("both source indicators need available labels")
    mean, scale, normalized = np.zeros(2), np.ones(2), np.zeros_like(target)
    rng = np.random.default_rng(seed)
    for channel in range(2):
        available = valid[..., channel]
        values = target[..., channel][available]
        mean[channel], scale[channel] = values.mean(), max(values.std(), 1e-6)
        values = (values-mean[channel])/scale[channel]
        normalized[..., channel][available] = rng.permutation(values) if shuffle else values
    return normalized, mean, scale, counts


class SourceChemistryRegularizer:
    """Disposable source heads; their targets never become model inputs.

    Uniform sampling from the union of source auxiliary cells is weighted by
    N_union/(2*N_indicator), giving each indicator half the expected loss.
    The ordinary DOC checkpoint criterion and inference class are unchanged.
    """

    def __init__(self, model, target, valid, *, weight=.1, shuffle=False,
                 seed=42, batch_size=512):
        if not np.isfinite(weight) or weight < 0 or batch_size < 1:
            raise ValueError("auxiliary weight must be nonnegative and batch size positive")
        values, mean, scale, counts = normalized_source_targets(
            target, valid, shuffle=shuffle, seed=seed)
        self.labels = torch.as_tensor(values, dtype=model.dtype)
        self.valid = torch.as_tensor(np.asarray(valid, dtype=bool))
        self.cells = np.flatnonzero(np.asarray(valid).any(-1))
        self.channel_weights = torch.as_tensor(len(self.cells)/(2*counts), dtype=model.dtype)
        self.mean, self.scale, self.counts = mean, scale, counts
        self.weight, self.shuffle, self.seed = float(weight), bool(shuffle), int(seed)
        self.batch_size, self.learning_rate = int(batch_size), .001
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.head = nn.Linear(model.hidden_size, 2, dtype=model.dtype)
        self.initial_state = copy.deepcopy(self.head.state_dict())
        self.reset()

    def reset(self):
        self.head.load_state_dict(self.initial_state)
        self._epoch, self._rng = None, None
        self.sampled_cells = 0
        self.initial_mse = None

    def bind(self, model, source):
        if tuple(source["age"].shape) != tuple(self.labels.shape[:2]):
            raise ValueError("auxiliary source rows must match DOC source input rows")
        self.initial_mse = self.diagnostics(model, source)

    def loss(self, model, source, epoch):
        if self._epoch != epoch:
            self._epoch = epoch
            self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 117]))
        cells = self._rng.choice(self.cells, size=self.batch_size, replace=True)
        months = self.labels.shape[1]
        rows, dates = cells//months, cells % months
        prediction = self.head(model._hidden_cells(source, cells))
        errors = (prediction-self.labels[rows, dates]).square()*self.valid[rows, dates]
        self.sampled_cells += len(cells)
        return self.weight*(errors*self.channel_weights).sum()/len(cells)

    def diagnostics(self, model, source):
        total = np.zeros(2)
        months = self.labels.shape[1]
        with torch.no_grad():
            for start in range(0, len(self.cells), self.batch_size):
                cells = self.cells[start:start+self.batch_size]
                rows, dates = cells//months, cells % months
                errors = (self.head(model._hidden_cells(source, cells))-
                          self.labels[rows, dates]).square()*self.valid[rows, dates]
                total += errors.sum(0).double().numpy()
        return (total/self.counts).tolist()

    def summary(self, model, source_inputs):
        selected = self.diagnostics(model, model._prepare_inputs(source_inputs))
        return {"auxiliary_targets": ["ph", "log1p_spec_conductance"],
            "weight": self.weight, "learning_rate": self.learning_rate,
            "source_label_shuffle": self.shuffle, "seed": self.seed,
            "target_mean": self.mean.tolist(), "target_scale": self.scale.tolist(),
            "fit_label_counts": self.counts.tolist(), "union_source_cells": len(self.cells),
            "sampled_source_cells": self.sampled_cells,
            "initial_source_mse": self.initial_mse, "selected_source_mse": selected,
            "doc_best_epoch": model.best_epoch_, "doc_epochs_run": model.epochs_run_,
            "selection_role": "source_validation_doc_mae",
            "receiving_chemistry_used": False,
            "auxiliary_trainable_parameters": sum(p.numel() for p in self.head.parameters())}

    def to_payload(self):
        return {"head": copy.deepcopy(self.head.state_dict()),
                "initial_head": copy.deepcopy(self.initial_state),
                "target_mean": self.mean.tolist(), "target_scale": self.scale.tolist(),
                "weight": self.weight, "source_label_shuffle": self.shuffle}
