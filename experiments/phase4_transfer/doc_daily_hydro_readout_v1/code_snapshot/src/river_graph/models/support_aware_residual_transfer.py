"""Choose ecological versus temporal residual transfer jointly with support use.

The ecological profile and temporal expert remain frozen. Each candidate mix
uses the existing SupportShapeAdapter without changing its alpha/ridge grid.
Source validation selects the mix separately for each support budget; K=0 is
locked to the previous ecological-memory decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from river_graph.models.support_shape_adapter import (
    K_VALUES,
    RIDGE_STRENGTHS,
    SupportShapeAdapter,
    SupportShapeEpisode,
)


@dataclass(frozen=True)
class SupportAwareTransferEpisode:
    """One source-validation task; memory arrays are signed native residuals."""

    k: int
    query_cells: np.ndarray
    query_values: np.ndarray
    support_cells: np.ndarray
    support_values: np.ndarray
    query_context: np.ndarray
    query_temporal: np.ndarray
    query_memory: np.ndarray
    support_context: np.ndarray
    support_temporal: np.ndarray
    support_memory: np.ndarray
    query_basis: np.ndarray
    support_basis: np.ndarray


def _base(context, temporal, memory, gamma):
    arrays = [np.asarray(value) for value in (context, temporal, memory)]
    if (any(array.ndim != 1 for array in arrays)
            or any(array.shape != arrays[0].shape for array in arrays)
            or any(not np.isfinite(array).all() for array in arrays)
            or (arrays[0] < 0).any() or (arrays[1] < 0).any()):
        raise ValueError("context/temporal predictions and signed memory must be finite aligned vectors; predictions nonnegative")
    if gamma == 0:
        # Preserve the original expert values and dtype without round trips.
        return arrays[1].copy()
    context, temporal, memory = (array.astype(np.float64) for array in arrays)
    prediction = np.maximum(0.0, context + (1 - gamma) * (temporal - context) + gamma * memory)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite ecological-temporal base prediction")
    return prediction


def _shape_episode(episode, gamma):
    return SupportShapeEpisode(
        k=episode.k, query_cells=episode.query_cells, query_values=episode.query_values,
        query_prediction=_base(episode.query_context, episode.query_temporal, episode.query_memory, gamma),
        support_cells=episode.support_cells, support_values=episode.support_values,
        support_prediction=_base(episode.support_context, episode.support_temporal, episode.support_memory, gamma),
        query_basis=episode.query_basis, support_basis=episode.support_basis,
    )


class SupportAwareResidualTransfer:
    """Jointly select mix gamma and the existing support-adapter parameters.

    Selection uses pooled source-validation query MAE. Exact ties prefer zero
    gamma, then the previously chosen K=0 gamma, then the smaller gamma.
    No target query labels enter ``adapt``; its only labels are support values.
    """

    def __init__(self, n_months, gamma_values=(0.0, .25, .5, 1.0),
                 ridge_strengths=RIDGE_STRENGTHS):
        probe = SupportShapeAdapter(n_months=n_months, ridge_strengths=ridge_strengths)
        self.n_months = probe.n_months
        self.ridge_strengths = probe.ridge_strengths
        gamma = np.asarray(tuple(gamma_values), dtype=np.float64)
        if (gamma.ndim != 1 or not len(gamma) or not np.isfinite(gamma).all()
                or (gamma < 0).any() or (gamma > 1).any() or not (gamma == 0).any()):
            raise ValueError("gamma_values must lie in [0,1] and include exact zero")
        self.gamma_values = tuple(float(value) for value in np.unique(gamma))

    def fit(self, episodes, *, gamma_k0, selection_role="source_validation"):
        if selection_role != "source_validation":
            raise ValueError("mix selection requires source_validation episodes")
        if isinstance(gamma_k0, (bool, np.bool_)) or gamma_k0 not in self.gamma_values:
            raise ValueError("locked gamma_k0 must belong to gamma_values")
        tasks = list(episodes)
        if not tasks or any(not isinstance(task, SupportAwareTransferEpisode) for task in tasks):
            raise ValueError("nonempty SupportAwareTransferEpisode tasks are required")
        if not any(task.k == 0 for task in tasks):
            raise ValueError("K=0 episode is required to retain the locked prior result")
        adapters, scores = {}, []
        for gamma in self.gamma_values:
            adapter = SupportShapeAdapter(n_months=self.n_months, ridge_strengths=self.ridge_strengths)
            adapter.fit([_shape_episode(task, gamma) for task in tasks], selection_role=selection_role)
            adapters[gamma] = adapter
            for k in sorted({task.k for task in tasks}):
                chosen = adapter.selection_by_k_[k]
                ridge_json = "infinity" if np.isinf(chosen["ridge_strength"]) else chosen["ridge_strength"]
                matched = [row for row in adapter.selection_scores_ if row["k"] == k and row["valid"]
                           and row["alpha"] == chosen["alpha"] and row["ridge_strength"] == ridge_json]
                if len(matched) != 1:
                    raise RuntimeError("support-adapter selected score is not uniquely recoverable")
                scores.append({**deepcopy(matched[0]), "gamma": gamma})
        choices = {}
        for k in sorted({task.k for task in tasks}):
            candidates = [row for row in scores if row["k"] == k and (k != 0 or row["gamma"] == gamma_k0)]
            selected = min(candidates, key=lambda row: (row["mae"], row["gamma"] != 0,
                                                        row["gamma"] != gamma_k0, row["gamma"]))
            choices[k] = {**deepcopy(selected), "locked": k == 0}
        self.adapters_by_gamma_, self.gamma_scores_ = adapters, scores
        self.selection_by_k_, self.gamma_k0_ = choices, float(gamma_k0)
        self.selection_role_ = selection_role
        return self

    def selected_gamma(self, k):
        if not hasattr(self, "selection_by_k_"):
            raise RuntimeError("fit support-aware transfer before prediction")
        if isinstance(k, (bool, np.bool_)) or k not in K_VALUES:
            raise ValueError("K must be 0, 1, 3 or 5")
        if k not in self.selection_by_k_:
            raise RuntimeError(f"K={k} has not been selected on source validation")
        return self.selection_by_k_[k]["gamma"]

    def selected_base(self, context, temporal, memory, *, k):
        return _base(context, temporal, memory, self.selected_gamma(k))

    def adapt(self, context_q, temporal_q, memory_q, query_cells,
              context_s, temporal_s, memory_s, support_cells, support_values,
              query_basis, support_basis, *, k):
        gamma = self.selected_gamma(k)
        base_q = _base(context_q, temporal_q, memory_q, gamma)
        base_s = _base(context_s, temporal_s, memory_s, gamma)
        return self.adapters_by_gamma_[gamma].adapt(
            base_q, query_cells, base_s, support_cells, support_values,
            query_basis=query_basis, support_basis=support_basis, k=k,
        )

    def to_dict(self):
        if not hasattr(self, "selection_by_k_"):
            raise RuntimeError("fit support-aware transfer before serialization")
        return deepcopy({
            "version": 1, "n_months": self.n_months, "gamma_values": list(self.gamma_values),
            "ridge_strengths": ["infinity" if np.isinf(value) else value for value in self.ridge_strengths],
            "gamma_k0": self.gamma_k0_, "selection_role": self.selection_role_,
            "selection_by_k": {str(k): choice for k, choice in self.selection_by_k_.items()},
            "gamma_scores": self.gamma_scores_,
            "adapters_by_gamma": {str(gamma): adapter.to_dict() for gamma, adapter in self.adapters_by_gamma_.items()},
            "selection_metric": "pooled source-validation native query MAE for each K",
            "tie_rule": "gamma zero, then locked K0 gamma, then smaller gamma",
            "profile_policy": "frozen ecological memory; no profile refitting",
            "k0_policy": "retain the previous ecological-memory gamma exactly",
        })

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1 or state.get("selection_role") != "source_validation":
            raise ValueError("unsupported support-aware transfer version or selection role")
        obj = cls(state["n_months"], gamma_values=state["gamma_values"],
                  ridge_strengths=[np.inf if value == "infinity" else float(value) for value in state["ridge_strengths"]])
        gamma_k0 = state["gamma_k0"]
        adapters = {float(gamma): SupportShapeAdapter.from_dict(saved)
                    for gamma, saved in state["adapters_by_gamma"].items()}
        choices = {int(k): deepcopy(choice) for k, choice in state["selection_by_k"].items()}
        if (gamma_k0 not in obj.gamma_values or set(adapters) != set(obj.gamma_values)
                or 0 not in choices or choices[0]["gamma"] != gamma_k0):
            raise ValueError("serialized transfer must retain all candidates and the locked K=0 gamma")
        for gamma, adapter in adapters.items():
            if (adapter.n_months != obj.n_months or adapter.ridge_strengths != obj.ridge_strengths
                    or getattr(adapter, "selection_role_", None) != "source_validation"):
                raise ValueError(f"invalid cached support adapter for gamma={gamma}")
        for k, choice in choices.items():
            if (k not in K_VALUES or choice["k"] != k or choice["gamma"] not in adapters
                    or choice["locked"] != (k == 0) or not choice["valid"]
                    or not np.isfinite(choice["mae"]) or choice["mae"] < 0):
                raise ValueError("invalid serialized per-K transfer selection")
            selected = adapters[choice["gamma"]].selection_by_k_.get(k)
            ridge = np.inf if choice["ridge_strength"] == "infinity" else choice["ridge_strength"]
            if selected != {"alpha": choice["alpha"], "ridge_strength": ridge}:
                raise ValueError("transfer selection must match its cached support adapter")
        obj.gamma_k0_, obj.adapters_by_gamma_, obj.selection_by_k_ = float(gamma_k0), adapters, choices
        obj.gamma_scores_, obj.selection_role_ = deepcopy(state["gamma_scores"]), state["selection_role"]
        return obj
