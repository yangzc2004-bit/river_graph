"""Small conditional densities over a fixed model's log1p DOC residual.

The input representation and native-unit base predictions are supplied by the
caller and remain fixed. Training minimizes unweighted source Gaussian NLL;
checkpoint and correction scale selection use pooled source-validation native
MAE. The two-component point prediction is its deterministic median, not its
mean. This module does not assign physical meanings to the mixture components
or assert that a neural source base is out-of-fold.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from river_graph.models.episodic_station_adapter import _integer

SIGMA_FLOOR = 0.03
FEATURE_SCALE_FLOOR = 1e-6
CORRECTION_SCALES = (0.0, 0.25, 0.5, 1.0)


def residual_distribution(raw, components):
    """Decode raw linear outputs; all returned tensors retain autograd.

    C=1: [mu, sigma_raw]. C=2: [mu_low, gap_raw, sigma_low_raw,
    sigma_high_raw, high_mix_logit]. The second mean cannot be below the first.
    Log weights are computed directly from logits, avoiding log(sigmoid(.))
    underflow in the training likelihood.
    """
    if components not in (1, 2) or raw.ndim != 2 or raw.shape[1] != (2 if components == 1 else 5):
        raise ValueError("raw distribution shape must be [rows,2] or [rows,5]")
    if not torch.isfinite(raw).all():
        raise FloatingPointError("nonfinite raw distribution parameters")
    if components == 1:
        means = raw[:, :1]
        scales = SIGMA_FLOOR + F.softplus(raw[:, 1:2])
        log_weights = torch.zeros_like(means)
    else:
        means = torch.stack((raw[:, 0], raw[:, 0] + F.softplus(raw[:, 1])), dim=1)
        scales = SIGMA_FLOOR + F.softplus(raw[:, 2:4])
        log_weights = torch.stack((F.logsigmoid(-raw[:, 4]), F.logsigmoid(raw[:, 4])), dim=1)
    if not torch.isfinite(means).all() or not torch.isfinite(scales).all():
        raise FloatingPointError("nonfinite decoded distribution parameters")
    return {"means": means, "scales": scales,
            "weights": log_weights.exp(), "log_weights": log_weights}


def gaussian_residual_nll(raw, residual, components):
    """Per-row Gaussian/mixture NLL in log1p-residual units, without weights."""
    dist = residual_distribution(raw, components)
    if residual.ndim != 1 or len(residual) != len(raw) or not torch.isfinite(residual).all():
        raise ValueError("residual must be a finite row-aligned vector")
    standardized = (residual[:, None] - dist["means"]) / dist["scales"]
    log_density = (-0.5 * standardized.square() - dist["scales"].log()
                   - 0.5 * math.log(2 * math.pi))
    return -torch.logsumexp(dist["log_weights"] + log_density, dim=1)


def gaussian_mixture_median(means, scales, weights):
    """Median of one/two ordered Gaussians; 64 deterministic bisection steps.

    The two-component CDF comparison is evaluated as a difference of positive
    log-tail sums. This avoids a false CDF=0.5 plateau for widely separated,
    balanced components. Bounds are min(mu-12*sigma), max(mu+12*sigma).
    This routine is an inference operation, not a differentiable quantile loss.
    """
    means, scales, weights = [torch.as_tensor(value, dtype=torch.float64).detach().cpu()
                              for value in (means, scales, weights)]
    if (means.ndim != 2 or means.shape[1] not in (1, 2)
            or scales.shape != means.shape or weights.shape != means.shape
            or not all(torch.isfinite(v).all() for v in (means, scales, weights))
            or (scales <= 0).any() or (weights < 0).any()
            or not torch.allclose(weights.sum(1), torch.ones(len(means), dtype=torch.float64),
                                  atol=1e-12, rtol=1e-12)):
        raise ValueError("finite aligned Gaussian means/scales/normalized weights required")
    if means.shape[1] == 1:
        return means[:, 0].clone()
    if (means[:, 1] < means[:, 0]).any():
        raise ValueError("two-component means must be ordered")
    low = (means - 12 * scales).min(1).values
    high = (means + 12 * scales).max(1).values
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise FloatingPointError("nonfinite median bracket")
    low_weight, high_weight = weights[:, 0], weights[:, 1]
    offset_positive = torch.clamp_min(low_weight - 0.5, 0).log()
    offset_negative = torch.clamp_min(0.5 - low_weight, 0).log()
    log_low, log_high = low_weight.log(), high_weight.log()
    for _ in range(64):
        mid = low + 0.5 * (high - low)
        z_low = (mid - means[:, 0]) / scales[:, 0]
        z_high = (mid - means[:, 1]) / scales[:, 1]
        log_positive = torch.logaddexp(offset_positive, log_high + torch.special.log_ndtr(z_high))
        log_negative = torch.logaddexp(offset_negative, log_low + torch.special.log_ndtr(-z_low))
        below = log_positive < log_negative
        low, high = torch.where(below, mid, low), torch.where(below, high, mid)
    return low + 0.5 * (high - low)


def _inverse_softplus(value):
    return float(value + np.log(-np.expm1(-value)))


def _finite_vector(value, rows, name):
    array = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if (array.ndim != 1 or len(array) != rows or not torch.isfinite(array).all()
            or (array < 0).any()):
        raise ValueError(f"{name} must be finite, nonnegative and row-aligned")
    return array


class DistributionalResidualHead:
    """Linear Gaussian or ordered Gaussian-mixture density on fixed features.

    ``fit`` takes only compact observed source and source-validation rows. The
    caller controls their label isolation. No label is accepted by inference.
    Source feature standard deviations below 1e-6 are replaced by one.
    ``predict_distribution`` reports transformed-residual parameters, before
    the selected point-correction scale. It is not a calibrated DOC interval.
    """

    def __init__(self, n_features=550, components=1, epochs=100, patience=10,
                 batch_size=512, seed=42, learning_rate=0.001):
        self.n_features = _integer(n_features, "n_features")
        self.components = _integer(components, "components")
        if self.components not in (1, 2):
            raise ValueError("components must be one or two")
        self.epochs = _integer(epochs, "epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed", minimum=0)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)
        with torch.random.fork_rng():
            self.head = nn.Linear(self.n_features, 2 if self.components == 1 else 5,
                                  dtype=torch.float64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.selected_scale_ = 0.0
        self.trainable_parameter_count_ = sum(p.numel() for p in self.head.parameters())
        self.fitted_ = False

    def _features(self, features, *, allow_empty=False):
        value = torch.as_tensor(features, dtype=torch.float64).detach().cpu()
        if (value.ndim != 2 or value.shape[1] != self.n_features
                or (not allow_empty and not len(value)) or not torch.isfinite(value).all()):
            raise ValueError("features must be finite [rows,n_features]")
        return value

    def _initialize(self, residual):
        array = residual.numpy()
        sigma = max(float(array.std()), 0.05)
        if self.components == 1:
            means = [float(array.mean())]
            bias = [means[0], _inverse_softplus(sigma - SIGMA_FLOOR)]
        else:
            low, high = np.quantile(array, [0.25, 0.75], method="linear")
            high = max(float(high), float(low) + 0.05)
            means = [float(low), high]
            bias = [means[0], _inverse_softplus(high - low),
                    _inverse_softplus(sigma - SIGMA_FLOOR),
                    _inverse_softplus(sigma - SIGMA_FLOOR), 0.0]
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor(bias, dtype=torch.float64))
        self.initialization_ = {"means": means, "scales": [sigma] * self.components,
                                "weights": [1 / self.components] * self.components,
                                "residual_mean": float(array.mean()),
                                "residual_std": float(array.std()), "raw_bias": bias}

    @staticmethod
    def _combine(base, residual, scale):
        if scale == 0:
            return base.clone()
        native = torch.expm1(torch.log1p(base) + scale * residual)
        if not torch.isfinite(native).all():
            raise FloatingPointError("nonfinite inverse-transformed point prediction")
        return native.clamp_min(0)

    @torch.no_grad()
    def _evaluate(self, features, base, truth, residual):
        raw = self.head(features)
        dist = residual_distribution(raw, self.components)
        median = gaussian_mixture_median(dist["means"], dist["scales"], dist["weights"])
        scores = [{"scale": scale, "mae": float((self._combine(base, median, scale) - truth).abs().mean())}
                  for scale in CORRECTION_SCALES]
        selected = min(scores, key=lambda row: (row["mae"], row["scale"]))
        nll = float(gaussian_residual_nll(raw, residual, self.components).mean())
        if not np.isfinite(nll):
            raise FloatingPointError("nonfinite validation NLL")
        return {"validation_mae": selected["mae"], "selected_scale": selected["scale"],
                "validation_nll": nll, "scale_scores": scores}

    def fit(self, source_features, source_base, source_truth,
            validation_features, validation_base, validation_truth, *,
            selection_role="source_validation", progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint and scale selection require source_validation")
        source, validation = self._features(source_features), self._features(validation_features)
        source_base = _finite_vector(source_base, len(source), "source_base")
        source_y = _finite_vector(source_truth, len(source), "source_truth")
        val_base = _finite_vector(validation_base, len(validation), "validation_base")
        val_y = _finite_vector(validation_truth, len(validation), "validation_truth")
        self.feature_mean_ = source.mean(0)
        self.feature_raw_std_ = source.std(0, correction=0)
        self.feature_scale_ = torch.where(self.feature_raw_std_ < FEATURE_SCALE_FLOOR,
                                           torch.ones_like(self.feature_raw_std_), self.feature_raw_std_)
        source = (source - self.feature_mean_) / self.feature_scale_
        validation = (validation - self.feature_mean_) / self.feature_scale_
        residual = torch.log1p(source_y) - torch.log1p(source_base)
        val_residual = torch.log1p(val_y) - torch.log1p(val_base)
        self._initialize(residual)
        self.n_source_cells_, self.n_validation_query_ = len(source), len(validation)
        self.selection_role_ = selection_role
        self.baseline_validation_mae_ = float((val_base - val_y).abs().mean())
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        self.trace_ = []
        best_rank, best_state, stale = (math.inf, math.inf, math.inf), None, 0
        self.best_epoch_, self.epochs_run_ = 0, 0
        self.head.train()
        for epoch in range(self.epochs + 1):
            if epoch:
                indices = rng.permutation(len(source))
                accumulated = 0.0
                for begin in range(0, len(indices), self.batch_size):
                    rows = indices[begin:begin + self.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = gaussian_residual_nll(self.head(source[rows]), residual[rows], self.components).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite source density objective")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(self.head.parameters(), 1.0, error_if_nonfinite=True)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite density gradient")
                    optimizer.step()
                    accumulated += float(loss.detach()) * len(rows)
                train_nll = accumulated / len(source)
            else:
                with torch.no_grad():
                    train_nll = float(gaussian_residual_nll(self.head(source), residual, self.components).mean())
            validation_scores = self._evaluate(validation, val_base, val_y, val_residual)
            rank = (validation_scores["validation_mae"], validation_scores["selected_scale"], epoch)
            improved = rank < best_rank
            if improved:
                best_rank = rank
                best_state = {key: value.detach().clone() for key, value in self.head.state_dict().items()}
                self.best_epoch_ = epoch
                self.selected_scale_ = validation_scores["selected_scale"]
                self.validation_metrics_ = copy.deepcopy(validation_scores)
                stale = 0
            else:
                stale += 1
            self.epochs_run_ = epoch
            record = {"epoch": epoch, "training_nll": train_nll,
                      **validation_scores, "is_best": improved, "stale_epochs": stale}
            self.trace_.append(record)
            if progress is not None:
                progress(copy.deepcopy(record))
            if epoch and stale >= self.patience:
                break
        self.head.load_state_dict(best_state)
        self.head.eval()
        self.fitted_ = True
        with torch.no_grad():
            self.selected_source_nll_ = float(
                gaussian_residual_nll(self.head(source), residual, self.components).mean())
        self.source_distribution_ = self._distribution_summary(source)
        self.validation_distribution_ = self._distribution_summary(validation)
        return self

    @torch.no_grad()
    def _distribution_summary(self, standardized_features):
        dist = residual_distribution(self.head(standardized_features), self.components)
        def describe(tensor):
            array = tensor.numpy().reshape(-1)
            return {"mean": float(array.mean()), "min": float(array.min()),
                    "p10": float(np.quantile(array, 0.1)), "median": float(np.median(array)),
                    "p90": float(np.quantile(array, 0.9)), "max": float(array.max())}
        output = {"n_rows": len(standardized_features),
                  "means": [describe(dist["means"][:, c]) for c in range(self.components)],
                  "scales": [describe(dist["scales"][:, c]) for c in range(self.components)],
                  "weights": [describe(dist["weights"][:, c]) for c in range(self.components)],
                  "sigma_near_floor_fraction": float((dist["scales"] < SIGMA_FLOOR + 1e-3).double().mean())}
        if self.components == 2:
            gap = dist["means"][:, 1] - dist["means"][:, 0]
            output.update({"mean_gap": describe(gap),
                           "mean_gap_below_0_01_fraction": float((gap < 0.01).double().mean()),
                           "high_weight_below_0_01_fraction": float((dist["weights"][:, 1] < 0.01).double().mean()),
                           "high_weight_above_0_99_fraction": float((dist["weights"][:, 1] > 0.99).double().mean())})
        return output

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("distributional head must be fitted or restored first")

    @torch.no_grad()
    def predict_distribution(self, features):
        self._require_fitted()
        value = self._features(features, allow_empty=True)
        standardized = (value - self.feature_mean_) / self.feature_scale_
        dist = residual_distribution(self.head(standardized), self.components)
        median = gaussian_mixture_median(dist["means"], dist["scales"], dist["weights"])
        return {"means": dist["means"].numpy(), "scales": dist["scales"].numpy(),
                "weights": dist["weights"].numpy(), "median_residual": median.numpy()}

    def predict(self, features, base):
        self._require_fitted()
        value = self._features(features, allow_empty=True)
        base_tensor = _finite_vector(base, len(value), "base")
        if self.selected_scale_ == 0:
            # Preserve the supplied native baseline's values and dtype exactly.
            return (base.detach().cpu().numpy() if isinstance(base, torch.Tensor)
                    else np.asarray(base)).copy()
        dist = self.predict_distribution(value.numpy())
        return self._combine(base_tensor, torch.from_numpy(dist["median_residual"]),
                             self.selected_scale_).numpy()

    def to_dict(self):
        self._require_fitted()
        return copy.deepcopy({
            "model_class": "DistributionalResidualHead", "schema_version": 1,
            "config": {"n_features": self.n_features, "components": self.components,
                       "epochs": self.epochs, "patience": self.patience,
                       "batch_size": self.batch_size, "seed": self.seed,
                       "learning_rate": self.learning_rate},
            "definition": {"residual": "log1p(truth)-log1p(fixed_native_base)",
                           "objective": "unweighted_source_gaussian_residual_nll",
                           "point": "mixture_median", "median_iterations": 64,
                           "median_bracket": "min(mu-12*sigma), max(mu+12*sigma)",
                           "sigma_floor": SIGMA_FLOOR, "sigma_upper_cap": None,
                           "correction_scales": list(CORRECTION_SCALES),
                           "scale_action": "max(0,expm1(log1p(base)+scale*median_residual))",
                           "selection": "pooled_source_validation_native_mae",
                           "selection_ties": "lower_scale_then_earlier_epoch",
                           "trace_training_nll": "online_batch_loss; epoch0_full_source",
                           "normalization_role": "source_training", "feature_std_floor": FEATURE_SCALE_FLOOR},
            "selection_role": self.selection_role_, "n_source_cells": self.n_source_cells_,
            "n_validation_query": self.n_validation_query_,
            "trainable_parameter_count": self.trainable_parameter_count_,
            "initialization": self.initialization_,
            "normalization": {"feature_mean": self.feature_mean_.tolist(),
                              "feature_scale": self.feature_scale_.tolist(),
                              "feature_raw_std": self.feature_raw_std_.tolist(),
                              "unit_scale_feature_count": int((self.feature_raw_std_ < FEATURE_SCALE_FLOOR).sum())},
            "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_,
            "selected_scale": self.selected_scale_,
            "baseline_validation_mae": self.baseline_validation_mae_,
            "selected_source_nll": self.selected_source_nll_,
            "initial_validation_mae": self.trace_[0]["validation_mae"],
            "validation_metrics": self.validation_metrics_, "trace": self.trace_,
            "source_distribution": self.source_distribution_,
            "validation_distribution": self.validation_distribution_,
        })

    def to_payload(self):
        return {"model_class": "DistributionalResidualHead", "schema_version": 1,
                "summary": self.to_dict(),
                "head_state": {key: value.detach().clone() for key, value in self.head.state_dict().items()}}

    @classmethod
    def from_payload(cls, payload):
        if (payload.get("model_class") != "DistributionalResidualHead"
                or payload.get("schema_version") != 1):
            raise ValueError("unsupported distributional residual payload")
        summary = copy.deepcopy(payload["summary"])
        model = cls(**summary["config"])
        model.head.load_state_dict(payload["head_state"])
        if not all(torch.isfinite(p).all() for p in model.head.parameters()):
            raise ValueError("nonfinite saved distribution weights")
        normalizer = summary["normalization"]
        for attribute, key in (("feature_mean_", "feature_mean"), ("feature_scale_", "feature_scale"),
                               ("feature_raw_std_", "feature_raw_std")):
            value = torch.as_tensor(normalizer[key], dtype=torch.float64)
            if value.shape != (model.n_features,) or not torch.isfinite(value).all():
                raise ValueError("invalid saved source normalization")
            setattr(model, attribute, value)
        if (model.feature_scale_ <= 0).any() or (model.feature_raw_std_ < 0).any():
            raise ValueError("invalid saved source standard deviations")
        model.selected_scale_ = float(summary["selected_scale"])
        if model.selected_scale_ not in CORRECTION_SCALES or summary["selection_role"] != "source_validation":
            raise ValueError("invalid saved validation selection")
        for attribute, key in (("initialization_", "initialization"),
                               ("selection_role_", "selection_role"),
                               ("n_source_cells_", "n_source_cells"),
                               ("n_validation_query_", "n_validation_query"),
                               ("best_epoch_", "best_epoch"), ("epochs_run_", "epochs_run"),
                               ("baseline_validation_mae_", "baseline_validation_mae"),
                               ("selected_source_nll_", "selected_source_nll"),
                               ("validation_metrics_", "validation_metrics"), ("trace_", "trace"),
                               ("source_distribution_", "source_distribution"),
                               ("validation_distribution_", "validation_distribution")):
            setattr(model, attribute, summary[key])
        model.head.eval()
        model.fitted_ = True
        return model
