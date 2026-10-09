"""A native-unit DOC correction on frozen features and explicit availability.

The caller supplies compact source and validation rows. Only this float64
linear head is fitted; source-only standardization and an availability gate
keep the baseline unchanged wherever auxiliary measurements are absent.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from river_graph.models.episodic_station_adapter import _integer

CORRECTION_SCALES = (0., .25, .5, 1.)
FEATURE_STD_FLOOR = 1e-6


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _native_vector(value, rows, name):
    value = _array(value)
    if (value.shape != (rows,) or not np.isfinite(value).all() or (value < 0).any()):
        raise ValueError(f"{name} must be finite nonnegative values aligned with feature rows")
    return torch.as_tensor(value, dtype=torch.float64).detach().cpu()


def _active_vector(value, rows, name):
    value = _array(value)
    if value.dtype.kind != "b" or value.shape != (rows,):
        raise ValueError(f"{name} must be an explicit boolean vector aligned with feature rows")
    return torch.as_tensor(value, dtype=torch.bool).detach().cpu()


class FrozenNativeFeatureHead:
    """Learn an availability-gated signed correction to a fixed native base.

    Source loss weights are two at/above the supplied source-training Q90 and
    one elsewhere. Each shuffled batch uses the same global mean weight in its
    loss denominator. Inactive rows retain their constant baseline error and
    contribute no gradient. Validation selects unweighted native MAE, including
    epoch zero and exact scale-zero fallback, with MAE/scale/epoch tie order.
    """

    def __init__(self, n_features=682, epochs=120, patience=10, batch_size=512,
                 seed=42, learning_rate=.001):
        self.n_features = _integer(n_features, "n_features")
        self.epochs = _integer(epochs, "epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed", minimum=0)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)
        with torch.random.fork_rng():
            self.head = nn.Linear(self.n_features, 1, dtype=torch.float64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.trainable_parameter_count_ = self.n_features + 1
        self.selected_scale_ = 0.
        self.fitted_ = False

    def _features(self, value, *, allow_empty=False):
        value = torch.as_tensor(_array(value), dtype=torch.float64).detach().cpu()
        if (value.ndim != 2 or value.shape[1] != self.n_features
                or (not allow_empty and not len(value)) or not torch.isfinite(value).all()):
            raise ValueError("features must be finite row-aligned [rows,n_features] arrays")
        return value

    @staticmethod
    def _combine(base, delta, scale):
        if scale == 0:
            return base.clone()
        raw = base + scale * delta
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite native correction")
        # At a zero baseline retain a derivative of one for upward learning.
        return torch.where(raw >= 0, raw, torch.zeros_like(raw))

    def _delta(self, standardized, active):
        raw = self.head(standardized).squeeze(-1)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite frozen-feature head output")
        return torch.where(active, raw, torch.zeros_like(raw))

    @torch.no_grad()
    def _evaluate(self, standardized, base, truth, active):
        delta = self._delta(standardized, active)
        scores = [{"scale": scale, "mae": float((self._combine(base, delta, scale)-truth).abs().mean())}
                  for scale in CORRECTION_SCALES]
        chosen = min(scores, key=lambda row: (row["mae"], row["scale"]))
        if not np.isfinite(chosen["mae"]):
            raise FloatingPointError("nonfinite validation MAE")
        return {"validation_mae": chosen["mae"], "selected_scale": chosen["scale"], "scale_scores": scores}

    def fit(self, source_features, source_base, source_truth,
            val_features, val_base, val_truth, *, tail_threshold,
            source_active, validation_active, selection_role="source_validation", progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        if not np.isfinite(tail_threshold) or tail_threshold < 0:
            raise ValueError("tail_threshold must be finite and nonnegative")
        source, validation = self._features(source_features), self._features(val_features)
        source_base = _native_vector(source_base, len(source), "source_base")
        source_y = _native_vector(source_truth, len(source), "source_truth")
        val_base = _native_vector(val_base, len(validation), "val_base")
        val_y = _native_vector(val_truth, len(validation), "val_truth")
        source_active = _active_vector(source_active, len(source), "source_active")
        val_active = _active_vector(validation_active, len(validation), "validation_active")
        self.feature_mean_ = source.mean(0)
        self.feature_raw_std_ = source.std(0, correction=0)
        self.feature_scale_ = torch.where(self.feature_raw_std_ < FEATURE_STD_FLOOR,
                                          torch.ones_like(self.feature_raw_std_), self.feature_raw_std_)
        source = (source-self.feature_mean_) / self.feature_scale_
        validation = (validation-self.feature_mean_) / self.feature_scale_
        tail = source_y >= float(tail_threshold)
        weights = 1. + tail.double()
        mean_weight, sum_weight = float(weights.mean()), float(weights.sum())
        self.tail_threshold_ = float(tail_threshold)
        self.source_weight_mean_, self.source_weight_sum_ = mean_weight, sum_weight
        self.n_source_cells_, self.n_validation_query_ = len(source), len(validation)
        self.n_source_tail_cells_ = int(tail.sum())
        self.n_source_active_, self.n_validation_active_ = int(source_active.sum()), int(val_active.sum())
        self.selection_role_ = selection_role
        self.baseline_validation_mae_ = float((val_base-val_y).abs().mean())
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.train()
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
        self.trace_, self.optimizer_steps_ = [], 0
        self.best_epoch_, self.epochs_run_, self.selected_scale_ = 0, 0, 0.
        best_rank, best_state, stale = (math.inf, math.inf, math.inf), None, 0
        for epoch in range(self.epochs + 1):
            if epoch:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
                order = rng.permutation(len(source))
                accumulated = 0.
                for begin in range(0, len(order), self.batch_size):
                    rows = order[begin:begin+self.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    delta = self._delta(source[rows], source_active[rows])
                    prediction = self._combine(source_base[rows], delta, 1.)
                    errors = weights[rows] * (prediction-source_y[rows]).abs()
                    loss = errors.mean() / mean_weight
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite source native objective")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    if not all(torch.isfinite(value).all() for value in self.head.parameters()):
                        raise FloatingPointError("nonfinite native head parameters")
                    self.optimizer_steps_ += 1
                    accumulated += float(errors.detach().sum())
                train_loss = accumulated / sum_weight
            else:
                train_loss = float((weights * (source_base-source_y).abs()).sum()) / sum_weight
            scores = self._evaluate(validation, val_base, val_y, val_active)
            rank = (scores["validation_mae"], scores["selected_scale"], epoch)
            improved = rank < best_rank
            if improved:
                best_rank = rank
                best_state = {key: value.detach().clone() for key, value in self.head.state_dict().items()}
                self.best_epoch_, self.selected_scale_ = epoch, scores["selected_scale"]
                self.validation_metrics_ = copy.deepcopy(scores)
                stale = 0
            else:
                stale += 1
            self.epochs_run_ = epoch
            record = {"epoch": epoch, "training_loss": train_loss, **scores,
                      "is_best": improved, "stale_epochs": stale}
            self.trace_.append(record)
            if progress is not None:
                progress(copy.deepcopy(record))
            if epoch and stale >= self.patience:
                break
        self.head.load_state_dict(best_state)
        self.head.eval()
        self.fitted_ = True
        with torch.no_grad():
            delta = self._delta(source, source_active)
            selected = self._combine(source_base, delta, self.selected_scale_)
            full_scale = self._combine(source_base, delta, 1.)
            self.selected_source_weighted_mae_ = float((weights*(selected-source_y).abs()).sum()) / sum_weight
            self.selected_source_full_scale_weighted_mae_ = float((weights*(full_scale-source_y).abs()).sum()) / sum_weight
        return self

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("fit or restore the native feature head before prediction")

    @torch.no_grad()
    def predict_delta(self, features, *, active):
        self._require_fitted()
        values = self._features(features, allow_empty=True)
        active = _active_vector(active, len(values), "active")
        standardized = (values-self.feature_mean_) / self.feature_scale_
        return self._delta(standardized, active).numpy()

    @torch.no_grad()
    def predict(self, features, base, *, active):
        self._require_fitted()
        values = self._features(features, allow_empty=True)
        base_tensor = _native_vector(base, len(values), "base")
        active_tensor = _active_vector(active, len(values), "active")
        if self.selected_scale_ == 0 or not active_tensor.any():
            return _array(base).copy()
        delta = self._delta((values-self.feature_mean_) / self.feature_scale_, active_tensor)
        predicted = self._combine(base_tensor, delta, self.selected_scale_)
        return torch.where(active_tensor, predicted, base_tensor).numpy()

    def to_dict(self):
        self._require_fitted()
        return copy.deepcopy({
            "model_class": "FrozenNativeFeatureHead", "schema_version": 1,
            "config": {name: getattr(self, name) for name in (
                "n_features", "epochs", "patience", "batch_size", "seed", "learning_rate")},
            "definition": {"output": "max(0,base+selected_scale*active*signed_native_delta)",
                "objective": "tail2_weighted_source_native_mae", "tail_rule": "source_truth>=source_training_Q90",
                "global_weight_denominator": "mean_source_weight_for_every_batch; all source rows included",
                "availability": "explicit boolean row gate; inactive delta zero and native base copied",
                "inactive_training_rows": "constant baseline loss included; zero head gradient",
                "normalization_role": "all_source_loss_rows", "feature_std_floor": FEATURE_STD_FLOOR,
                "initialization": "all linear weights and bias zero", "correction_scales": list(CORRECTION_SCALES),
                "selection": "unweighted_pooled_source_validation_native_mae",
                "selection_ties": "lower_scale_then_earlier_epoch", "gradient_clip_norm": 1.,
                "source_shuffle": "numpy SeedSequence([seed,epoch]); uniform cells",
                "trace_training_loss": "online_batch_loss; epoch0_full_source",
                "frozen_feature_gradients": "detached; no backbone or forest updates"},
            "selection_role": self.selection_role_, "tail_threshold": self.tail_threshold_,
            "n_source_cells": self.n_source_cells_, "n_source_tail_cells": self.n_source_tail_cells_,
            "n_validation_query": self.n_validation_query_, "n_source_active": self.n_source_active_,
            "n_validation_active": self.n_validation_active_, "source_weight_mean": self.source_weight_mean_,
            "source_weight_sum": self.source_weight_sum_, "trainable_parameter_count": self.trainable_parameter_count_,
            "normalization": {"feature_mean": self.feature_mean_.tolist(), "feature_scale": self.feature_scale_.tolist(),
                "feature_raw_std": self.feature_raw_std_.tolist(),
                "unit_scale_feature_count": int((self.feature_raw_std_ < FEATURE_STD_FLOOR).sum())},
            "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_, "selected_scale": self.selected_scale_,
            "optimizer_steps": self.optimizer_steps_, "baseline_validation_mae": self.baseline_validation_mae_,
            "initial_validation_mae": self.trace_[0]["validation_mae"],
            "selected_source_weighted_mae": self.selected_source_weighted_mae_,
            "selected_source_full_scale_weighted_mae": self.selected_source_full_scale_weighted_mae_,
            "validation_metrics": self.validation_metrics_, "trace": self.trace_})

    def to_payload(self):
        return {"model_class": "FrozenNativeFeatureHead", "schema_version": 1, "summary": self.to_dict(),
                "head_state": {key: value.detach().clone() for key, value in self.head.state_dict().items()}}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("model_class") != "FrozenNativeFeatureHead" or payload.get("schema_version") != 1:
            raise ValueError("unsupported frozen native feature-head checkpoint")
        summary = copy.deepcopy(payload["summary"])
        model = cls(**summary["config"])
        model.head.load_state_dict(payload["head_state"])
        if not all(torch.isfinite(value).all() for value in model.head.parameters()):
            raise ValueError("nonfinite saved native head weights")
        normalizer = summary["normalization"]
        for attribute, key in (("feature_mean_", "feature_mean"), ("feature_scale_", "feature_scale"),
                               ("feature_raw_std_", "feature_raw_std")):
            value = torch.as_tensor(normalizer[key], dtype=torch.float64)
            if value.shape != (model.n_features,) or not torch.isfinite(value).all():
                raise ValueError("invalid source feature normalization")
            setattr(model, attribute, value)
        expected_scale = torch.where(model.feature_raw_std_ < FEATURE_STD_FLOOR,
                                     torch.ones_like(model.feature_raw_std_), model.feature_raw_std_)
        if ((model.feature_raw_std_ < 0).any() or not torch.equal(model.feature_scale_, expected_scale)):
            raise ValueError("invalid source feature standard deviations")
        if (summary["selected_scale"] not in CORRECTION_SCALES or summary["selection_role"] != "source_validation"
                or summary["trainable_parameter_count"] != model.trainable_parameter_count_):
            raise ValueError("invalid saved head selection or parameter count")
        for name in ("selection_role", "tail_threshold", "n_source_cells", "n_source_tail_cells",
                     "n_validation_query", "n_source_active", "n_validation_active", "source_weight_mean",
                     "source_weight_sum", "best_epoch", "epochs_run", "selected_scale", "optimizer_steps",
                     "baseline_validation_mae", "selected_source_weighted_mae",
                     "selected_source_full_scale_weighted_mae", "validation_metrics", "trace"):
            setattr(model, name+"_", summary[name])
        model.head.eval()
        model.fitted_ = True
        return model
