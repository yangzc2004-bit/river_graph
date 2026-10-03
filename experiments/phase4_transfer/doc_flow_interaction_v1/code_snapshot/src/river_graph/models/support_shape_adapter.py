"""Frozen temporal bases and support-conditioned residual heads.

The basis is fitted on source-station feature sequences only. Every station is
centered over its own complete feature timeline before fitting or projection;
callers must project that timeline once, then gather support/query cells. This
is retrospective reconstruction, not a claim of prospective forecasting.

The adapter learns a small ridge head from each station's support labels. Its
hyperparameters use pooled source-validation query MAE. Target query labels
are absent from the adaptation interface.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

K_VALUES = (0, 1, 3, 5)
ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_STRENGTHS = (0.1, 1.0, 10.0, float("inf"))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ridge_json(value):
    return "infinity" if np.isinf(value) else float(value)


def _ridge_float(value):
    return float("inf") if value == "infinity" else float(value)


class StationTemporalBasis:
    """PCA of within-station dynamics with source-fitted whitening.

    ``fit`` accepts [station, month, feature]. ``fit_station_sequences`` also
    accepts a stream of [month, feature] arrays, accumulating centered second
    moments without retaining all source features. Months have equal weight.
    ``transform`` accepts either a whole station timeline or a station batch.
    The default output has two components; near-zero variances use a finite
    eigenvalue floor. Target station centering uses covariates, never DOC labels.
    """

    def __init__(self, n_components=2, eigenvalue_floor=1e-8):
        self.n_components = _positive_integer(n_components, "n_components")
        if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be finite and positive")
        self.eigenvalue_floor = float(eigenvalue_floor)

    def fit(self, source_features, *, source_role="source_training"):
        features = np.asarray(source_features, dtype=np.float64)
        if features.ndim != 3:
            raise ValueError("source_features must be [station, month, feature]")
        return self.fit_station_sequences(features, source_role=source_role)

    def fit_station_sequences(self, sequences: Iterable[np.ndarray], *,
                              source_role="source_training"):
        if source_role != "source_training":
            raise ValueError("basis fitting requires source_training features")
        total, n_months, n_stations, n_features = None, 0, 0, None
        for sequence in sequences:
            features = np.asarray(sequence, dtype=np.float64)
            if (features.ndim != 2 or min(features.shape) < 1
                    or not np.isfinite(features).all()):
                raise ValueError("each station needs finite [month, feature] values")
            if n_features is None:
                n_features = features.shape[1]
                total = np.zeros((n_features, n_features), dtype=np.float64)
            elif features.shape[1] != n_features:
                raise ValueError("source feature dimensions differ between stations")
            centered = features - features.mean(axis=0, keepdims=True)
            total += centered.T @ centered
            n_months += len(features)
            n_stations += 1
        if not n_stations or n_features < self.n_components:
            raise ValueError("source sequences must support the requested component count")
        covariance = total / n_months
        if not np.isfinite(covariance).all():
            raise ValueError("source covariance is not finite")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:self.n_components]
        components = eigenvectors[:, order].T
        # Fix the otherwise arbitrary sign of each eigenvector.
        for vector in components:
            if vector[np.argmax(np.abs(vector))] < 0:
                vector *= -1
        self.components_ = components
        self.eigenvalues_ = np.maximum(eigenvalues[order], 0.0)
        self.n_features_in_ = n_features
        self.n_source_stations_ = n_stations
        self.n_source_months_ = n_months
        return self

    def transform(self, features):
        if not hasattr(self, "components_"):
            raise RuntimeError("fit the source basis before projection")
        array = np.asarray(features, dtype=np.float64)
        if (array.ndim not in (2, 3) or min(array.shape) < 1
                or array.shape[-1] != self.n_features_in_ or not np.isfinite(array).all()):
            raise ValueError("features must contain finite, dimension-matched station timelines")
        centered = array - array.mean(axis=-2, keepdims=True)
        scale = np.sqrt(np.maximum(self.eigenvalues_, self.eigenvalue_floor))
        projected = (centered @ self.components_.T) / scale
        if not np.isfinite(projected).all():
            raise ValueError("projected temporal basis is not finite")
        return projected

    def to_dict(self):
        if not hasattr(self, "components_"):
            raise RuntimeError("fit the source basis before serialization")
        return {"version": 1, "n_components": self.n_components,
                "eigenvalue_floor": self.eigenvalue_floor,
                "n_features": self.n_features_in_, "n_source_stations": self.n_source_stations_,
                "n_source_months": self.n_source_months_,
                "components": self.components_.tolist(), "eigenvalues": self.eigenvalues_.tolist()}

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported temporal-basis version")
        obj = cls(state["n_components"], state["eigenvalue_floor"])
        obj.n_features_in_ = _positive_integer(state["n_features"], "n_features")
        obj.n_source_stations_ = _positive_integer(state["n_source_stations"], "n_source_stations")
        obj.n_source_months_ = _positive_integer(state["n_source_months"], "n_source_months")
        obj.components_ = np.asarray(state["components"], dtype=np.float64)
        obj.eigenvalues_ = np.asarray(state["eigenvalues"], dtype=np.float64)
        if (obj.components_.shape != (obj.n_components, obj.n_features_in_)
                or obj.eigenvalues_.shape != (obj.n_components,)
                or not np.isfinite(obj.components_).all()
                or not np.isfinite(obj.eigenvalues_).all() or (obj.eigenvalues_ < 0).any()):
            raise ValueError("invalid serialized temporal basis")
        return obj


@dataclass(frozen=True)
class SupportShapeEpisode:
    """Source-validation support/query task for one K, possibly many stations.

    Cell IDs are flattened station-major indices. Bases have shape [rows, d]
    and must come from whole-station projection before selecting these rows.
    Query labels are used only by ``fit`` to select hyperparameters.
    """

    k: int
    query_cells: np.ndarray
    query_values: np.ndarray
    query_prediction: np.ndarray
    support_cells: np.ndarray
    support_values: np.ndarray
    support_prediction: np.ndarray
    query_basis: np.ndarray
    support_basis: np.ndarray


class SupportShapeAdapter:
    """Support-fitted station level and temporal-shape correction in log1p space.

    Ridge minimizes mean squared centered support residual plus lambda*||a||².
    Each K has its own source-validation-selected alpha and ridge strength.
    Use ``ridge_strengths=(float('inf'),)`` for the matched constant-only arm.
    Selection pools query losses across episodes, weighting each query cell
    equally. Ties prefer smaller alpha, then stronger ridge (constant first).
    """

    def __init__(self, *, n_months, alpha_values=ALPHA_VALUES,
                 ridge_strengths=RIDGE_STRENGTHS):
        self.n_months = _positive_integer(n_months, "n_months")
        alpha = np.asarray(tuple(alpha_values), dtype=float)
        ridge = np.asarray(tuple(ridge_strengths), dtype=float)
        if (alpha.ndim != 1 or not len(alpha) or not np.isfinite(alpha).all()
                or ((alpha < 0) | (alpha > 1)).any()):
            raise ValueError("alpha candidates must be finite values in [0, 1]")
        if ridge.ndim != 1 or not len(ridge) or np.isnan(ridge).any() or (ridge <= 0).any():
            raise ValueError("ridge candidates must be positive or infinity")
        self.alpha_values = tuple(float(value) for value in np.unique(alpha))
        self.ridge_strengths = tuple(float(value) for value in np.unique(ridge))
        self.selection_by_k_ = {0: {"alpha": 0.0, "ridge_strength": float("inf")}}
        self.selection_scores_ = []

    @staticmethod
    def _values(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"{name} must be finite, nonnegative and one-dimensional")
        return array

    def _prepare(self, query_prediction, query_cells, support_prediction, support_cells,
                 support_values, query_basis, support_basis, k):
        if isinstance(k, bool) or k not in K_VALUES:
            raise ValueError("K must be 0, 1, 3 or 5")
        indices = []
        for values in (query_cells, support_cells):
            raw = np.asarray(values)
            if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
                raise ValueError("cell identities must be one-dimensional integers")
            cells = raw.astype(np.int64)
            if (cells < 0).any() or len(np.unique(cells)) != len(cells):
                raise ValueError("cell identities must be nonnegative and unique")
            indices.append(cells)
        query, support = indices
        if np.intersect1d(query, support).size:
            raise ValueError("support and query cells must be disjoint")
        pred = self._values(query_prediction, "query_prediction")
        support_pred = self._values(support_prediction, "support_prediction")
        values = self._values(support_values, "support_values")
        if pred.shape != query.shape or support_pred.shape != support.shape or values.shape != support.shape:
            raise ValueError("predictions and values must align with their cell identities")
        qb, sb = np.asarray(query_basis, dtype=float), np.asarray(support_basis, dtype=float)
        if (qb.ndim != 2 or sb.ndim != 2 or qb.shape[0] != len(query)
                or sb.shape != (len(support), qb.shape[1]) or qb.shape[1] < 1
                or not np.isfinite(qb).all() or not np.isfinite(sb).all()):
            raise ValueError("basis arrays must be finite and align with predictions")
        stations, counts = np.unique(support // self.n_months, return_counts=True)
        if k == 0:
            if len(support):
                raise ValueError("K=0 requires empty support")
        elif (not np.array_equal(stations, np.unique(query // self.n_months))
              or not (counts == k).all()):
            raise ValueError("every query station must have exactly K support cells")
        return pred, query, support_pred, support, values, qb, sb

    def _apply(self, prepared, *, alpha, ridge_strength, k):
        pred, query, support_pred, support, values, qb, sb = prepared
        if k == 0 or (alpha == 0 and np.isinf(ridge_strength)):
            return pred.copy()
        output = np.log1p(pred)
        residual = np.log1p(values) - np.log1p(support_pred)
        for station in np.unique(query // self.n_months):
            q = query // self.n_months == station
            s = support // self.n_months == station
            mean_residual = float(residual[s].mean())
            output[q] += alpha * mean_residual
            if k > 1 and np.isfinite(ridge_strength):
                mean_basis = sb[s].mean(axis=0)
                centered = sb[s] - mean_basis
                gram = centered.T @ centered / k + ridge_strength * np.eye(qb.shape[1])
                rhs = centered.T @ (residual[s] - mean_residual) / k
                if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
                    raise FloatingPointError("nonfinite support ridge moments")
                coefficient = np.linalg.solve(gram, rhs)
                output[q] += (qb[q] - mean_basis) @ coefficient
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.maximum(np.expm1(output), 0.0)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("support correction produced nonfinite DOC predictions")
        return prediction

    def fit(self, episodes: Iterable[SupportShapeEpisode], *, selection_role):
        if selection_role != "source_validation":
            raise ValueError("adapter selection requires source_validation episodes")
        grouped = {}
        for episode in episodes:
            prepared = self._prepare(
                episode.query_prediction, episode.query_cells, episode.support_prediction,
                episode.support_cells, episode.support_values, episode.query_basis,
                episode.support_basis, episode.k)
            truth = self._values(episode.query_values, "source validation query values")
            if truth.shape != prepared[0].shape or not len(truth):
                raise ValueError("validation queries must be aligned and nonempty")
            grouped.setdefault(episode.k, []).append((prepared, truth))
        if not grouped:
            raise ValueError("at least one source-validation episode is required")
        selected = {0: {"alpha": 0.0, "ridge_strength": float("inf")}}
        scores = []
        for k, tasks in sorted(grouped.items()):
            candidates = []
            for alpha in ((0.0,) if k == 0 else self.alpha_values):
                for ridge in ((float("inf"),) if k == 0 else self.ridge_strengths):
                    total_error, total_cells, valid = 0.0, 0, True
                    for prepared, truth in tasks:
                        try:
                            prediction = self._apply(prepared, alpha=alpha, ridge_strength=ridge, k=k)
                        except (FloatingPointError, np.linalg.LinAlgError):
                            valid = False
                            break
                        total_error += float(np.abs(prediction - truth).sum())
                        total_cells += len(truth)
                    mae = total_error / total_cells if valid else None
                    if mae is not None and not np.isfinite(mae):
                        valid, mae = False, None
                    scores.append({"k": k, "alpha": alpha, "ridge_strength": _ridge_json(ridge),
                                   "mae": mae, "valid": valid,
                                   "n_query_cells": sum(len(truth) for _, truth in tasks)})
                    if valid:
                        candidates.append((mae, alpha, -ridge))
            if not candidates:
                raise FloatingPointError(f"no finite validation candidate for K={k}")
            _, alpha, negative_ridge = min(candidates)
            selected[k] = {"alpha": alpha, "ridge_strength": -negative_ridge}
        self.selection_by_k_, self.selection_scores_ = selected, scores
        self.selection_role_ = selection_role
        return self

    def adapt(self, query_prediction, query_cells, support_prediction, support_cells,
              support_values, *, query_basis, support_basis, k):
        prepared = self._prepare(query_prediction, query_cells, support_prediction, support_cells,
                                 support_values, query_basis, support_basis, k)
        if k not in self.selection_by_k_:
            raise RuntimeError(f"K={k} has not been selected on source validation")
        if k == 0:
            # Preserve both values and dtype rather than taking a log/inverse round trip.
            return np.asarray(query_prediction).copy()
        return self._apply(prepared, **self.selection_by_k_[k], k=k)

    def to_dict(self):
        return {"version": 1, "n_months": self.n_months,
                "alpha_values": list(self.alpha_values),
                "ridge_strengths": [_ridge_json(value) for value in self.ridge_strengths],
                "selection_role": getattr(self, "selection_role_", None),
                "selection_by_k": {str(k): {"alpha": choice["alpha"],
                    "ridge_strength": _ridge_json(choice["ridge_strength"])}
                    for k, choice in self.selection_by_k_.items()},
                "selection_scores": self.selection_scores_}

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported support-adapter version")
        obj = cls(n_months=state["n_months"], alpha_values=state["alpha_values"],
                  ridge_strengths=[_ridge_float(value) for value in state["ridge_strengths"]])
        choices = {}
        for key, value in state["selection_by_k"].items():
            k, alpha, ridge = int(key), float(value["alpha"]), _ridge_float(value["ridge_strength"])
            if (k not in K_VALUES or not np.isfinite(alpha) or not 0 <= alpha <= 1
                    or np.isnan(ridge) or ridge <= 0
                    or (k == 0 and (alpha != 0 or not np.isinf(ridge)))
                    or (k != 0 and (alpha not in obj.alpha_values or ridge not in obj.ridge_strengths))):
                raise ValueError("invalid serialized support-adapter choice")
            choices[k] = {"alpha": alpha, "ridge_strength": ridge}
        if 0 not in choices:
            raise ValueError("serialized adapter must retain the exact K=0 path")
        role = state.get("selection_role")
        if role not in (None, "source_validation") or (len(choices) > 1 and role is None):
            raise ValueError("invalid serialized selection role")
        obj.selection_by_k_ = choices
        obj.selection_scores_ = list(state["selection_scores"])
        if role is not None:
            obj.selection_role_ = role
        return obj
