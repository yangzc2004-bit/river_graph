"""Native-unit DOC residual over a fixed context forest.

Only copied GRU/decay weights and a new signed scalar head are trained. Spatial
inputs and the context forest stay fixed. Source bases must be station OOF
predictions; validation/test bases come from the corresponding full-source
forest. A validation-selected scale includes the exact context-only candidate.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from river_graph.models.episodic_station_adapter import _integer
from river_graph.models.episodic_temporal_adapter import (
    _parameter_distance,
    _state_copy,
    gathered_rolling_states,
)


class NativeTemporalResidual:
    """Fit a signed mg/L correction with overall validation MAE selection.

    ``predict_delta`` returns the unscaled native-unit head output. ``predict``
    applies the selected scale and a nonnegative floor. Source training uses
    uniformly shuffled observed cells, with optional weight two at or above the
    source-training Q90 threshold. Validation always uses unweighted raw MAE.
    """

    def __init__(self, temporal, decay, lookback=12, epochs=30, patience=5,
                 batch_size=512, seed=42, learning_rate=1e-4,
                 head_learning_rate=1e-3, tail_weight=1.0,
                 scales=(0.0, 0.25, 0.5, 1.0)):
        if (not isinstance(temporal, nn.GRUCell) or not isinstance(decay, nn.Linear)
                or temporal.input_size != temporal.hidden_size + 1
                or decay.in_features != 4 or decay.out_features != temporal.hidden_size):
            raise ValueError("expected GRUCell(H+1,H) and Linear(4,H) decay")
        self.hidden_size = temporal.hidden_size
        self.lookback = _integer(lookback, "lookback")
        self.epochs = _integer(epochs, "epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed", minimum=0)
        for name, value in (("learning_rate", learning_rate),
                            ("head_learning_rate", head_learning_rate)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        if tail_weight not in (1, 2):
            raise ValueError("tail_weight must be 1 or 2")
        self.tail_weight = float(tail_weight)
        candidates = np.asarray(tuple(scales), dtype=float)
        if (candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all()
                or ((candidates < 0) | (candidates > 1)).any() or not (candidates == 0).any()):
            raise ValueError("scales must lie in [0,1] and include exact context scale zero")
        self.scales = tuple(float(value) for value in np.unique(candidates))
        self.temporal, self.decay = copy.deepcopy(temporal).cpu(), copy.deepcopy(decay).cpu()
        self.dtype = next(self.temporal.parameters()).dtype
        if self.dtype not in (torch.float32, torch.float64) or next(self.decay.parameters()).dtype != self.dtype:
            raise ValueError("GRU and decay require the same float32 or float64 dtype")
        self.temporal.requires_grad_(True)
        self.decay.requires_grad_(True)
        self._initial_temporal_state = _state_copy(self.temporal)
        self._initial_decay_state = _state_copy(self.decay)
        # The random constructor state is discarded; do not consume a caller's RNG.
        with torch.random.fork_rng():
            self.head = nn.Linear(self.hidden_size, 1, dtype=self.dtype)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.selected_scale_ = 0.0
        self.trainable_parameter_count_ = sum(
            p.numel() for module in (self.temporal, self.decay, self.head) for p in module.parameters())

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, dict) or not all(
                key in inputs for key in ("encoded", "age", "support")):
            raise ValueError("inputs require encoded, age and support arrays")
        arrays = {key: torch.as_tensor(inputs[key], dtype=self.dtype).detach().cpu()
                  for key in ("encoded", "age", "support")}
        shape = arrays["encoded"].shape
        if (len(shape) != 3 or min(shape) < 1 or shape[-1] != self.hidden_size
                or arrays["age"].shape != shape[:2] or arrays["support"].shape != (*shape[:2], 3)
                or not all(torch.isfinite(value).all() for value in arrays.values())):
            raise ValueError("inputs must be finite aligned station/month tensors")
        return arrays

    @staticmethod
    def _observed_labels(inputs, base, truth, mask):
        shape = tuple(inputs["age"].shape)
        mask = np.asarray(mask)
        base, truth = np.asarray(base, dtype=float), np.asarray(truth, dtype=float)
        if mask.dtype.kind != "b" or mask.shape != shape or base.shape != shape or truth.shape != shape:
            raise ValueError("native base, truth and boolean mask must align with station/month inputs")
        cells = np.flatnonzero(mask)
        if not len(cells):
            raise ValueError("at least one observed/query cell is required")
        base, truth = base.reshape(-1)[cells], truth.reshape(-1)[cells]
        if (not np.isfinite(base).all() or not np.isfinite(truth).all()
                or (base < 0).any() or (truth < 0).any()):
            raise ValueError("selected native base and truth must be finite and nonnegative")
        # Only selected entries become tensors; hidden NaNs never enter a loss.
        return cells, torch.as_tensor(base), torch.as_tensor(truth)

    def _delta_cells(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        months = inputs["age"].shape[1]
        hidden = gathered_rolling_states(self.temporal, self.decay, inputs,
                                        cells // months, cells % months, lookback=self.lookback)
        delta = self.head(hidden).squeeze(-1).double()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite native temporal residual")
        return delta

    @staticmethod
    def _combine(base, delta, scale):
        if scale == 0:
            return base.clone()
        raw = base + scale * delta
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite native residual prediction")
        # Explicitly select derivative one at the zero boundary. The installed
        # clamp_min/ReLU implementation selects zero there, blocking an
        # initially zero residual from learning upward corrections at base=0.
        return torch.where(raw >= 0, raw, torch.zeros_like(raw))

    def fit(self, source_inputs, source_base_native, source_truth, source_mask,
            val_inputs, val_base_native, val_truth, val_query_mask, *, tail_threshold,
            selection_role, progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        if not np.isfinite(tail_threshold) or tail_threshold < 0:
            raise ValueError("tail_threshold must be finite and nonnegative")
        source, validation = self._prepare_inputs(source_inputs), self._prepare_inputs(val_inputs)
        source_cells, source_base, source_y = self._observed_labels(
            source, source_base_native, source_truth, source_mask)
        val_cells, val_base, val_y = self._observed_labels(
            validation, val_base_native, val_truth, val_query_mask)
        self.tail_threshold_ = float(tail_threshold)
        source_tail = source_y >= self.tail_threshold_
        weights = 1.0 + (self.tail_weight - 1.0) * source_tail.double()
        # Uniform cell minibatches estimate the complete weighted objective;
        # one fixed source mean avoids random per-batch tail normalization.
        mean_weight = float(weights.mean())
        self.n_source_cells_, self.n_validation_query_ = len(source_cells), len(val_cells)
        self.n_source_tail_cells_ = int(source_tail.sum())
        self.n_source_stations_ = len(np.unique(source_cells // source["age"].shape[1]))
        self.n_validation_stations_ = len(np.unique(val_cells // validation["age"].shape[1]))
        self.temporal.load_state_dict(self._initial_temporal_state)
        self.decay.load_state_dict(self._initial_decay_state)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        backbone = list(self.temporal.parameters()) + list(self.decay.parameters())
        parameters = backbone + list(self.head.parameters())
        optimizer = torch.optim.Adam([
            {"params": backbone, "lr": self.learning_rate},
            {"params": self.head.parameters(), "lr": self.head_learning_rate},
        ])
        self.trace_, self.best_epoch_, self.selected_scale_ = [], 0, 0.0
        self.optimizer_steps_ = 0
        best, stale, best_state = float("inf"), 0, None

        def evaluate(epoch, train_loss):
            nonlocal best, stale, best_state
            with torch.no_grad():
                delta = torch.cat([
                    self._delta_cells(validation, val_cells[start:start + self.batch_size])
                    for start in range(0, len(val_cells), self.batch_size)])
                scores = [{"scale": scale, "mae": float(
                    (self._combine(val_base, delta, scale) - val_y).abs().mean())}
                    for scale in self.scales]
            chosen = min(scores, key=lambda row: (row["mae"], row["scale"]))
            if not np.isfinite(chosen["mae"]):
                raise FloatingPointError("nonfinite native validation MAE")
            improved = chosen["mae"] < best
            if improved:
                best, stale = chosen["mae"], 0
                self.best_epoch_, self.selected_scale_ = epoch, chosen["scale"]
                best_state = [_state_copy(module) for module in (self.temporal, self.decay, self.head)]
            else:
                stale += 1
            row = {"epoch": epoch, "training_loss": train_loss,
                   "validation_mae": chosen["mae"], "validation_scale": chosen["scale"],
                   "validation_candidates": scores, "is_best": improved, "stale_epochs": stale,
                   "best_epoch": self.best_epoch_, "best_scale": self.selected_scale_,
                   "best_validation_mae": best,
                   "temporal_parameter_distance": _parameter_distance(self.temporal, self._initial_temporal_state),
                   "decay_parameter_distance": _parameter_distance(self.decay, self._initial_decay_state),
                   "head_parameter_norm": float(torch.sqrt(sum(
                       parameter.detach().double().square().sum() for parameter in self.head.parameters())))}
            self.trace_.append(row)
            if improved:
                self.validation_metrics_ = copy.deepcopy(row)
            if progress is not None:
                progress(copy.deepcopy(row))

        evaluate(0, None)
        for epoch in range(1, self.epochs + 1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
            order = rng.permutation(len(source_cells))
            weighted_error = 0.0
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                optimizer.zero_grad()
                delta = self._delta_cells(source, source_cells[rows])
                prediction = self._combine(source_base[rows], delta, 1.0)
                errors = weights[rows] * (prediction - source_y[rows]).abs()
                loss = errors.mean() / mean_weight
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite native training loss")
                loss.backward()
                if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                    raise FloatingPointError("nonfinite native residual gradient")
                nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                self.optimizer_steps_ += 1
                if not all(torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError("nonfinite native residual parameter")
                weighted_error += float(errors.detach().sum())
            evaluate(epoch, weighted_error / float(weights.sum()))
            if stale >= self.patience:
                break
        self.epochs_run_ = len(self.trace_) - 1
        for module, state in zip((self.temporal, self.decay, self.head), best_state):
            module.load_state_dict(state)
        return self

    def predict_delta(self, full_inputs, batch_size=2048):
        """Unscaled signed native correction, without labels or anchor centering."""
        batch_size = _integer(batch_size, "batch_size")
        inputs = self._prepare_inputs(full_inputs)
        n, t = inputs["age"].shape
        result = np.empty(n * t, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(result), batch_size):
                stop = min(start + batch_size, len(result))
                result[start:stop] = self._delta_cells(inputs, np.arange(start, stop)).numpy()
        return result.reshape(n, t)

    def predict(self, full_inputs, base_native):
        base = np.asarray(base_native)
        shape = np.shape(full_inputs["age"])
        if base.shape != shape or not np.isfinite(base).all() or (base < 0).any():
            raise ValueError("base_native must contain finite nonnegative station/month predictions")
        if self.selected_scale_ == 0:
            return base.copy()
        delta = self.predict_delta(full_inputs)
        combined = base.astype(np.float64) + self.selected_scale_ * delta
        if not np.isfinite(combined).all():
            raise FloatingPointError("nonfinite native residual prediction")
        return np.maximum(combined, 0.0)

    def _config(self):
        return {name: getattr(self, name) for name in (
            "lookback", "epochs", "patience", "batch_size", "seed", "learning_rate",
            "head_learning_rate", "tail_weight", "scales")}

    def to_dict(self):
        if not hasattr(self, "epochs_run_"):
            raise RuntimeError("fit the native residual before serialization")
        config = self._config()
        config["scales"] = list(config["scales"])
        return {"version": 1, "config": config, "hidden_size": self.hidden_size,
                "dtype": str(self.dtype), "trainable_parameter_count": self.trainable_parameter_count_,
                "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_,
                "selected_scale": self.selected_scale_, "optimizer_steps": self.optimizer_steps_,
                "trace": copy.deepcopy(self.trace_), "validation_metrics": copy.deepcopy(self.validation_metrics_),
                "tail_threshold": self.tail_threshold_, "n_source_cells": self.n_source_cells_,
                "n_source_tail_cells": self.n_source_tail_cells_,
                "source_tail_fraction": self.n_source_tail_cells_ / self.n_source_cells_,
                "n_validation_query": self.n_validation_query_,
                "n_source_stations": self.n_source_stations_,
                "n_validation_stations": self.n_validation_stations_,
                "temporal_parameter_distance": _parameter_distance(self.temporal, self._initial_temporal_state),
                "decay_parameter_distance": _parameter_distance(self.decay, self._initial_decay_state),
                "head_parameter_norm": float(torch.sqrt(sum(
                    parameter.detach().double().square().sum() for parameter in self.head.parameters()))),
                "protocol": {"base": "fixed context forest; source station OOF predictions",
                             "output": "max(0, base_native + selected_scale * signed_native_delta)",
                             "training_loss": "weighted_cell_raw_mae; fixed global source mean weight",
                             "tail_rule": "source_truth >= source_training_Q90",
                             "validation_loss": "unweighted pooled fixed-query raw MAE",
                             "selection_role": "source_validation", "optimizer": "Adam",
                             "gradient_clip_norm": 1.0, "nonfinite_policy": "fail; no skipped update or upper clipping",
                             "initialization": "provided original expert GRU/decay; zero scalar head"}}

    def to_payload(self):
        return {"version": 1, "config": self._config(), "hidden_size": self.hidden_size,
                "dtype": str(self.dtype), "temporal_bias": self.temporal.bias,
                "decay_bias": self.decay.bias is not None,
                "temporal": _state_copy(self.temporal), "decay": _state_copy(self.decay),
                "head": _state_copy(self.head),
                "initial_temporal": copy.deepcopy(self._initial_temporal_state),
                "initial_decay": copy.deepcopy(self._initial_decay_state), "summary": self.to_dict()}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1:
            raise ValueError("unsupported native residual checkpoint version")
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(payload["dtype"])
        if dtype is None:
            raise ValueError("unsupported native residual dtype")
        h = _integer(payload["hidden_size"], "hidden_size")
        with torch.random.fork_rng():
            temporal = nn.GRUCell(h + 1, h, bias=payload["temporal_bias"]).to(dtype=dtype)
            decay = nn.Linear(4, h, bias=payload["decay_bias"]).to(dtype=dtype)
        temporal.load_state_dict(payload["initial_temporal"])
        decay.load_state_dict(payload["initial_decay"])
        obj = cls(temporal, decay, **payload["config"])
        for name in ("temporal", "decay", "head"):
            getattr(obj, name).load_state_dict(payload[name])
        if not all(torch.isfinite(parameter).all()
                   for module in (obj.temporal, obj.decay, obj.head)
                   for parameter in module.parameters()):
            raise ValueError("checkpoint contains nonfinite residual weights")
        summary = payload["summary"]
        for name in ("best_epoch", "epochs_run", "optimizer_steps", "n_source_cells", "n_source_tail_cells",
                     "n_validation_query", "n_source_stations", "n_validation_stations"):
            setattr(obj, name + "_", int(summary[name]))
        obj.selected_scale_, obj.tail_threshold_ = float(summary["selected_scale"]), float(summary["tail_threshold"])
        if obj.selected_scale_ not in obj.scales or not np.isfinite(obj.tail_threshold_) or obj.tail_threshold_ < 0:
            raise ValueError("invalid serialized scale or tail threshold")
        obj.trace_, obj.validation_metrics_ = copy.deepcopy(summary["trace"]), copy.deepcopy(summary["validation_metrics"])
        return obj
