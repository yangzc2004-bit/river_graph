"""Station-cross-validated expert fusion shrunk towards environmental prediction.

For DOC, fit z = z_C + b0 + bC*z_C + bR*(z_R-z_C), with z = log1p(y).
All three correction coefficients are penalized towards zero. Expert models
remain fixed. Station CV evaluates this fusion stage conditionally on those
experts; it is not nested end-to-end validation if expert selection used the
same source-validation labels.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import numpy as np

DEFAULT_STRENGTHS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_CONVEX_WEIGHTS = tuple(round(float(weight), 10) for weight in np.linspace(0, 1, 21))


def _values(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must be one-dimensional, finite and nonnegative")
    return result


def _stations(values, n_rows):
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != n_rows:
        raise ValueError("station IDs must be one-dimensional and aligned with predictions")
    if result.dtype.kind not in "iuUS":
        allowed = (str, np.str_, int, np.integer)
        if result.dtype.kind != "O" or any(
            not isinstance(value, allowed) or isinstance(value, (bool, np.bool_)) for value in result
        ):
            raise ValueError("station IDs must be strings or integers without missing values")
    canonical = result.astype(str)
    if any(not value.strip() for value in canonical):
        raise ValueError("station IDs must not be empty")
    return canonical


def _grid(values, name, *, positive):
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional grid")
    if positive and (result <= 0).any():
        raise ValueError(f"{name} must be positive")
    if not positive and ((result < 0).any() or (result > 1).any()):
        raise ValueError(f"{name} must be nonempty, finite and in [0, 1]")
    return tuple(float(value) for value in np.unique(result))


def _design(context, temporal):
    context_z, temporal_z = np.log1p(context), np.log1p(temporal)
    return np.column_stack([np.ones(len(context)), context_z, temporal_z - context_z])


def _predict(context, temporal, kind, coefficients):
    if kind == "context":
        return context.copy()
    if kind == "convex" and coefficients[2] == 1.0:
        return temporal.copy()
    z = np.log1p(context) + _design(context, temporal) @ np.asarray(coefficients)
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.maximum(np.expm1(z), 0.0)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite fusion prediction; no upper clipping is applied")
    return prediction


def _fit_coefficients(candidate, context, temporal, truth):
    if candidate["kind"] == "context":
        return np.zeros(3, dtype=np.float64)
    if candidate["kind"] == "convex":
        return np.array([0.0, 0.0, candidate["weight"]], dtype=np.float64)
    design = _design(context, temporal)
    residual = np.log1p(truth) - np.log1p(context)
    # Mean squared residual + lambda * ||b||²: duplicating all rows leaves
    # lambda's interpretation unchanged. The intercept is penalized too.
    gram = design.T @ design / len(context)
    target = design.T @ residual / len(context)
    return np.linalg.solve(gram + candidate["strength"] * np.eye(3), target)


class RegularizedStationFusion:
    """Select fixed-expert fusion by pooled held-station raw-scale MAE.

    Five deterministic folds contain disjoint source-validation station IDs.
    Candidate selection uses each cell once, not an unweighted mean of fold
    MAEs. No one-standard-error gate is applied. Exact score ties prefer the
    context expert, then convex candidates in ascending weight, then ridge
    candidates in descending strength. Chosen coefficients are refit using all
    source validation for target inference.

    ``predict_crossfit`` applies the selected candidate's held-station fold
    coefficients at any supplied dates for known source-validation stations.
    Its candidate was still selected using all CV scores; these predictions
    are cross-fitted coefficient estimates, not a nested performance estimate.
    """

    def __init__(self, *, strengths: Iterable[float] = DEFAULT_STRENGTHS,
                 convex_weights: Iterable[float] = DEFAULT_CONVEX_WEIGHTS,
                 n_splits: int = 5, seed: int = 42):
        if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
            raise ValueError("n_splits must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.strengths = _grid(strengths, "strengths", positive=True)
        self.convex_weights = _grid(convex_weights, "convex weights", positive=False)
        self.n_splits, self.seed = int(n_splits), int(seed)

    def _candidates(self):
        candidates = [{"name": "context", "kind": "context", "strength": None, "weight": 0.0}]
        candidates.extend({"name": f"convex_{weight:g}", "kind": "convex", "strength": None,
                           "weight": weight} for weight in self.convex_weights if weight > 0)
        candidates.extend({"name": f"ridge_{strength:g}", "kind": "ridge", "strength": strength,
                           "weight": None} for strength in reversed(self.strengths))
        return candidates

    def fit(self, context_prediction, temporal_prediction, truth, station_ids, *, selection_role):
        if selection_role != "source_validation":
            raise ValueError("fusion selection must use source_validation predictions")
        context = _values(context_prediction, "context_prediction")
        temporal = _values(temporal_prediction, "temporal_prediction")
        y = _values(truth, "truth")
        if not len(context) or context.shape != temporal.shape or context.shape != y.shape:
            raise ValueError("expert predictions and source-validation labels must align and be nonempty")
        stations = _stations(station_ids, len(context))
        ids = np.unique(stations)
        if len(ids) < self.n_splits:
            raise ValueError("source validation needs at least n_splits distinct stations")
        station_folds = np.array_split(np.random.default_rng(self.seed).permutation(ids), self.n_splits)
        mapping = {str(station): fold for fold, held in enumerate(station_folds) for station in held}
        assignment = np.array([mapping[station] for station in stations], dtype=np.int64)
        folds = [{"fold": fold, "held_stations": sorted(held.tolist()),
                  "n_validation_cells": int((assignment == fold).sum()),
                  "n_training_cells": int((assignment != fold).sum())}
                 for fold, held in enumerate(station_folds)]
        results = []
        for candidate in self._candidates():
            total_loss, valid, fold_results = 0.0, True, []
            for fold in range(self.n_splits):
                held = assignment == fold
                coefficients = _fit_coefficients(candidate, context[~held], temporal[~held], y[~held])
                row = {"fold": fold, "n_cells": int(held.sum()), "coefficients": coefficients.tolist()}
                try:
                    predicted = _predict(context[held], temporal[held], candidate["kind"], coefficients)
                    absolute_error = np.abs(predicted - y[held])
                    row.update(valid=True, mae=float(absolute_error.mean()))
                    total_loss += float(absolute_error.sum())
                except ValueError:
                    valid = False
                    row.update(valid=False, mae=None)
                fold_results.append(row)
            results.append({**candidate, "valid": valid, "cv_mae": total_loss / len(y) if valid else None,
                            "n_cells": len(y), "fold_results": fold_results})
        chosen = min((result for result in results if result["valid"]), key=lambda row: row["cv_mae"])
        coefficients = _fit_coefficients(chosen, context, temporal, y)
        refit_prediction = _predict(context, temporal, chosen["kind"], coefficients)
        self.selected_ = {key: chosen[key] for key in ("name", "kind", "strength", "weight", "cv_mae")}
        self.selected_["coefficients"] = coefficients.tolist()
        self.selected_["refit_training_mae"] = float(np.abs(refit_prediction - y).mean())
        self.coefficients_ = coefficients
        self.cv_results_ = results
        self.folds_, self.station_to_fold_ = folds, dict(sorted(mapping.items()))
        self.crossfit_coefficients_ = np.asarray([row["coefficients"] for row in chosen["fold_results"]])
        self.selection_role_ = selection_role
        self.n_fit_rows_, self.n_fit_stations_ = len(y), len(ids)
        return self

    def predict(self, context_prediction, temporal_prediction):
        if not hasattr(self, "selected_"):
            raise RuntimeError("fit must be called before prediction")
        context = _values(context_prediction, "context_prediction")
        temporal = _values(temporal_prediction, "temporal_prediction")
        if context.shape != temporal.shape:
            raise ValueError("expert predictions must be aligned")
        return _predict(context, temporal, self.selected_["kind"], self.coefficients_)

    def predict_crossfit(self, context_prediction, temporal_prediction, station_ids):
        """Use fold coefficients excluding each station, including new dates."""
        if not hasattr(self, "selected_"):
            raise RuntimeError("fit must be called before cross-fitted prediction")
        context = _values(context_prediction, "context_prediction")
        temporal = _values(temporal_prediction, "temporal_prediction")
        if context.shape != temporal.shape:
            raise ValueError("expert predictions must be aligned")
        stations = _stations(station_ids, len(context))
        unknown = sorted(set(stations) - self.station_to_fold_.keys())
        if unknown:
            raise ValueError(f"crossfit requires known source-validation stations: {unknown}")
        assignment = np.array([self.station_to_fold_[station] for station in stations])
        prediction = np.empty_like(context)
        for fold in range(self.n_splits):
            held = assignment == fold
            if held.any():
                prediction[held] = _predict(context[held], temporal[held], self.selected_["kind"],
                                            self.crossfit_coefficients_[fold])
        return prediction

    def to_dict(self):
        if not hasattr(self, "selected_"):
            raise RuntimeError("fit must be called before serialization")
        return deepcopy({
            "schema_version": 1, "strengths": list(self.strengths),
            "convex_weights": list(self.convex_weights), "n_splits": self.n_splits, "seed": self.seed,
            "selected": self.selected_, "cv_results": self.cv_results_, "folds": self.folds_,
            "station_to_fold": self.station_to_fold_,
            "crossfit_coefficients": self.crossfit_coefficients_.tolist(),
            "selection_role": self.selection_role_, "n_fit_rows": self.n_fit_rows_,
            "n_fit_stations": self.n_fit_stations_,
            "cv_scope": "conditional fusion-stage station CV with fixed experts",
            "objective": "mean log-residual squared loss + strength * squared correction coefficients",
            "selection_metric": "pooled held-station raw MAE, cell weighted",
        })

    @classmethod
    def from_dict(cls, state):
        if state.get("schema_version") != 1 or state.get("selection_role") != "source_validation":
            raise ValueError("unsupported fusion state or selection role")
        model = cls(strengths=state["strengths"], convex_weights=state["convex_weights"],
                    n_splits=state["n_splits"], seed=state["seed"])
        selected = deepcopy(state["selected"])
        if selected["kind"] not in {"context", "convex", "ridge"}:
            raise ValueError("invalid fitted candidate kind")
        coefficients = np.asarray(selected["coefficients"], dtype=np.float64)
        crossfit = np.asarray(state["crossfit_coefficients"], dtype=np.float64)
        if (coefficients.shape != (3,) or not np.isfinite(coefficients).all()
                or crossfit.shape != (model.n_splits, 3) or not np.isfinite(crossfit).all()):
            raise ValueError("invalid fitted correction coefficients")
        mapping = state["station_to_fold"]
        if (len(mapping) != state["n_fit_stations"] or any(
            not isinstance(fold, int) or not 0 <= fold < model.n_splits for fold in mapping.values()
        )):
            raise ValueError("invalid cross-fitted station mapping")
        model.selected_, model.coefficients_ = selected, coefficients
        model.crossfit_coefficients_ = crossfit
        model.cv_results_, model.folds_ = deepcopy(state["cv_results"]), deepcopy(state["folds"])
        model.station_to_fold_ = dict(sorted(mapping.items()))
        model.n_fit_rows_, model.n_fit_stations_ = state["n_fit_rows"], state["n_fit_stations"]
        model.selection_role_ = state["selection_role"]
        return model
