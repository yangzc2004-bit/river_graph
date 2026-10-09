"""A small nonlinear chemistry basis over the frozen DOC representation.

The inherited trainer, native loss, source normalization, availability fallback
and validation selection are unchanged. Only four auxiliary inputs receive a
learned nonlinear encoding. The original 550 features remain a linear context
path, with interactions between the first 64 standardized hidden features and
the eight learned chemistry coordinates.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from river_graph.models.frozen_native_feature_head import (
    CORRECTION_SCALES,
    FEATURE_STD_FLOOR,
    FrozenNativeFeatureHead,
)

ORIGINAL_FEATURES = 550
HIDDEN_FEATURES = 64
AUXILIARY_FEATURES = 4
CHEMISTRY_BASIS = 8
INPUT_FEATURES = ORIGINAL_FEATURES + AUXILIARY_FEATURES
EXPANDED_FEATURES = ORIGINAL_FEATURES + CHEMISTRY_BASIS + HIDDEN_FEATURES * CHEMISTRY_BASIS


class _NonlinearChemistryProjection(nn.Module):
    """Expose final weight/bias for the inherited trainer's zero-reset hook."""

    def __init__(self):
        super().__init__()
        self.phi = nn.Linear(AUXILIARY_FEATURES, CHEMISTRY_BASIS, dtype=torch.float64)
        self.output = nn.Linear(EXPANDED_FEATURES, 1, dtype=torch.float64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def weight(self):
        return self.output.weight

    @property
    def bias(self):
        return self.output.bias

    def expanded_features(self, standardized):
        original = standardized[:, :ORIGINAL_FEATURES]
        chemistry = F.silu(self.phi(standardized[:, ORIGINAL_FEATURES:]))
        interaction = (original[:, :HIDDEN_FEATURES, None] * chemistry[:, None, :]).flatten(1)
        return torch.cat((original, chemistry, interaction), dim=1)

    def forward(self, standardized):
        return self.output(self.expanded_features(standardized))


class NonlinearChemistryHead(FrozenNativeFeatureHead):
    """Train 1,111 parameters on 554 fixed features, with an eight-state basis.

    Feature order is original550 + auxiliary4. The parent's source-only
    standardization is applied to all 554 inputs before this module; expanded
    features are not standardized again. No backbone, forest, loss or sampling
    changes are introduced. ``fit`` resets phi to its isolated seeded initial
    state and then uses the unmodified parent fitting implementation.
    """

    def __init__(self, n_features=INPUT_FEATURES, epochs=120, patience=10,
                 batch_size=512, seed=42, learning_rate=.001):
        super().__init__(n_features=n_features, epochs=epochs, patience=patience,
                         batch_size=batch_size, seed=seed, learning_rate=learning_rate)
        if self.n_features != INPUT_FEATURES:
            raise ValueError("nonlinear chemistry requires original550 + auxiliary4 = 554 inputs")
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.head = _NonlinearChemistryProjection()
        self._initial_phi_state = {key: value.detach().clone()
                                   for key, value in self.head.phi.state_dict().items()}
        self.trainable_parameter_count_ = sum(value.numel() for value in self.head.parameters())

    def fit(self, source_features, source_base, source_truth,
            val_features, val_base, val_truth, *, tail_threshold,
            source_active, validation_active, selection_role="source_validation", progress=None):
        self.head.phi.load_state_dict(self._initial_phi_state)
        return super().fit(source_features, source_base, source_truth,
                           val_features, val_base, val_truth, tail_threshold=tail_threshold,
                           source_active=source_active, validation_active=validation_active,
                           selection_role=selection_role, progress=progress)

    @staticmethod
    def architecture():
        return {"input_features": INPUT_FEATURES,
                "input_layout": "original550, auxiliary4",
                "original_features": ORIGINAL_FEATURES, "hidden_features": HIDDEN_FEATURES,
                "auxiliary_features": AUXILIARY_FEATURES, "chemistry_basis": CHEMISTRY_BASIS,
                "phi": "SiLU(Linear(4,8))", "expanded_features": EXPANDED_FEATURES,
                "expanded_layout": "standardized_original550, phi8, standardized_hidden64 x phi8",
                "interaction_order": "hidden-major; chemistry-coordinate minor",
                "output": "Linear(1070,1), zero initialized",
                "normalization": "source-only554 before phi; no expanded-feature renormalization",
                "dropout": 0.0, "dtype": "float64"}

    def to_dict(self):
        summary = super().to_dict()
        summary["model_class"] = "NonlinearChemistryHead"
        summary["architecture"] = self.architecture()
        summary["definition"]["initialization"] = (
            "phi uses nn.Linear reset_parameters with isolated torch seed; final output weight/bias zero")
        summary["phi_initialization"] = {
            "seed": self.seed, "method": "torch nn.Linear(4,8) default reset_parameters",
            "weight": self._initial_phi_state["weight"].tolist(),
            "bias": self._initial_phi_state["bias"].tolist()}
        distance = sum((value.detach() - self._initial_phi_state[key]).square().sum()
                       for key, value in self.head.phi.state_dict().items()).sqrt()
        summary["phi_parameter_distance"] = float(distance)
        summary["phi_weight_norm"] = float(self.head.phi.weight.detach().norm())
        summary["output_weight_norm"] = float(self.head.output.weight.detach().norm())
        return summary

    def to_payload(self):
        return {"model_class": "NonlinearChemistryHead", "schema_version": 1,
                "summary": self.to_dict(),
                "head_state": {key: value.detach().clone() for key, value in self.head.state_dict().items()},
                "initial_phi_state": {key: value.detach().clone() for key, value in self._initial_phi_state.items()}}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("model_class") != "NonlinearChemistryHead" or payload.get("schema_version") != 1:
            raise ValueError("unsupported nonlinear chemistry head checkpoint")
        summary = copy.deepcopy(payload["summary"])
        if summary.get("model_class") != "NonlinearChemistryHead" or summary.get("architecture") != cls.architecture():
            raise ValueError("checkpoint architecture does not match the nonlinear chemistry head")
        model = cls(**summary["config"])
        model.head.load_state_dict(payload["head_state"])
        if not all(torch.isfinite(value).all() for value in model.head.parameters()):
            raise ValueError("nonfinite saved nonlinear head weights")
        initial = payload["initial_phi_state"]
        if set(initial) != {"weight", "bias"}:
            raise ValueError("invalid initial chemistry state keys")
        for key, shape in (("weight", (CHEMISTRY_BASIS, AUXILIARY_FEATURES)), ("bias", (CHEMISTRY_BASIS,))):
            value = torch.as_tensor(initial[key], dtype=torch.float64).detach().cpu().clone()
            documented = torch.as_tensor(summary["phi_initialization"][key], dtype=torch.float64)
            if value.shape != shape or not torch.isfinite(value).all() or not torch.equal(value, documented):
                raise ValueError("invalid saved initial chemistry basis")
            model._initial_phi_state[key] = value
        if summary["phi_initialization"]["seed"] != model.seed:
            raise ValueError("initial chemistry seed does not match configuration")
        normalizer = summary["normalization"]
        for attribute, key in (("feature_mean_", "feature_mean"), ("feature_scale_", "feature_scale"),
                               ("feature_raw_std_", "feature_raw_std")):
            value = torch.as_tensor(normalizer[key], dtype=torch.float64)
            if value.shape != (model.n_features,) or not torch.isfinite(value).all():
                raise ValueError("invalid saved source feature normalization")
            setattr(model, attribute, value)
        expected_scale = torch.where(model.feature_raw_std_ < FEATURE_STD_FLOOR,
                                     torch.ones_like(model.feature_raw_std_), model.feature_raw_std_)
        if (model.feature_raw_std_ < 0).any() or not torch.equal(model.feature_scale_, expected_scale):
            raise ValueError("invalid saved source feature standard deviations")
        if (summary["selected_scale"] not in CORRECTION_SCALES
                or summary["selection_role"] != "source_validation"
                or summary["trainable_parameter_count"] != model.trainable_parameter_count_):
            raise ValueError("invalid saved nonlinear head selection or parameter count")
        for name in ("selection_role", "tail_threshold", "n_source_cells", "n_source_tail_cells",
                     "n_validation_query", "n_source_active", "n_validation_active", "source_weight_mean",
                     "source_weight_sum", "best_epoch", "epochs_run", "selected_scale", "optimizer_steps",
                     "baseline_validation_mae", "selected_source_weighted_mae",
                     "selected_source_full_scale_weighted_mae", "validation_metrics", "trace"):
            setattr(model, name + "_", summary[name])
        model.head.eval()
        model.fitted_ = True
        return model
