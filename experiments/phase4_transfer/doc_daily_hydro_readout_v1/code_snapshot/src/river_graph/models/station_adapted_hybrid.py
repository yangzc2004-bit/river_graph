"""Frozen expert fusion and support-only station adaptation for DOC.

The temporal expert is the *complete* local-base-plus-neural-residual prediction,
in native concentration units, not the neural delta alone. Fusion and shrinkage
are selected on source-internal held-out predictions. Target-station adaptation
then reads only the explicitly supplied support labels. This is retrospective
record reconstruction: support observations may occur after query months.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


def _values(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or not np.isfinite(out).all() or (out < 0).any():
        raise ValueError(f"{name} must be finite, nonnegative and one-dimensional")
    return out


def _cells(cells: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(cells)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must contain one-dimensional integer cell indices")
    out = raw.astype(np.int64)
    if (out < 0).any() or len(np.unique(out)) != len(out):
        raise ValueError(f"{name} must be unique nonnegative cell indices")
    return out


def _inverse(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.maximum(np.expm1(z), 0.0)
    if not np.isfinite(result).all():
        raise ValueError("transformed prediction overflowed")
    return result


def station_residual_correction(
    query_prediction: np.ndarray,
    query_cells: np.ndarray,
    support_prediction: np.ndarray,
    support_cells: np.ndarray,
    support_values: np.ndarray,
    *,
    n_months: int,
    alpha: float,
) -> np.ndarray:
    """Add a shrunken mean support residual in log1p space for each station.

    Cell identifiers explicitly align predictions and observations. In contrast
    with a station level-matching correction, residuals compare truth and model
    *at the same support months*. The query's seasonal pattern is retained.
    Query labels are intentionally absent from this interface.
    """
    if isinstance(n_months, bool) or not isinstance(n_months, (int, np.integer)) or n_months < 1:
        raise ValueError("n_months must be a positive integer")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    prediction = _values(query_prediction, "query_prediction")
    support_pred = _values(support_prediction, "support_prediction")
    labels = _values(support_values, "support_values")
    query = _cells(query_cells, "query_cells")
    support = _cells(support_cells, "support_cells")
    if prediction.shape != query.shape:
        raise ValueError("query predictions and cells must be aligned")
    if support_pred.shape != support.shape or labels.shape != support.shape:
        raise ValueError("support predictions, labels and cells must be aligned")
    if np.intersect1d(support, query).size:
        raise ValueError("support and query must be disjoint")
    if not len(support) or alpha == 0.0:
        # Avoid a log/inverse round trip: K=0 is exactly the original expert.
        return prediction.copy()
    out = np.log1p(prediction)
    support_residual = np.log1p(labels) - np.log1p(support_pred)
    support_station = support // n_months
    query_station = query // n_months
    for station in np.unique(query_station):
        selected = support_station == station
        if selected.any():
            out[query_station == station] += alpha * float(support_residual[selected].mean())
    return _inverse(out)


@dataclass(frozen=True)
class CalibrationEpisode:
    """One source-internal station-holdout episode at a specified support K.

    The query population must be fixed across K outside this low-level class.
    Query labels here are source validation labels, never outer-test labels.
    """

    k: int
    query_cells: np.ndarray
    query_values: np.ndarray
    context_query_prediction: np.ndarray
    temporal_query_prediction: np.ndarray
    support_cells: np.ndarray
    support_values: np.ndarray
    context_support_prediction: np.ndarray
    temporal_support_prediction: np.ndarray


class StationAdaptedHybrid:
    """Select expert fusion and support correction with matched selection budgets.

    ``fit_fusion`` accepts source-validation predictions only. A context-only
    candidate remains available and its selection is recorded transparently.
    ``fit_calibration`` selects alpha independently for context and hybrid at each
    K, using the same source-validation episodes and candidate grid for both.
    Episodes are weighted equally; cells are weighted equally within each episode. The frozen fitted
    object subsequently predicts without accepting any query labels.
    """

    def __init__(
        self,
        *,
        n_months: int,
        blend_weights: Iterable[float] = tuple(np.linspace(0.0, 1.0, 201)),
        shrinkage_weights: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        allow_affine: bool = True,
    ) -> None:
        if isinstance(n_months, bool) or not isinstance(n_months, (int, np.integer)) or n_months < 1:
            raise ValueError("n_months must be a positive integer")
        self.n_months = int(n_months)
        self.blend_weights = self._weights(blend_weights)
        self.shrinkage_weights = self._weights(shrinkage_weights)
        self.allow_affine = bool(allow_affine)
        self.alpha_by_arm_: dict[str, dict[int, float]] = {
            "context": {0: 0.0}, "hybrid": {0: 0.0},
        }

    @staticmethod
    def _weights(weights: Iterable[float]) -> tuple[float, ...]:
        values = np.asarray(tuple(weights), dtype=np.float64)
        if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()):
            raise ValueError("candidate weights must be nonempty and in [0, 1]")
        return tuple(float(value) for value in np.unique(values))

    @staticmethod
    def _require_source_validation(selection_role: str) -> None:
        if selection_role != "source_validation":
            raise ValueError("selection must use source_validation predictions")

    def fit_fusion(
        self,
        context_prediction: np.ndarray,
        temporal_prediction: np.ndarray,
        truth: np.ndarray,
        *,
        selection_role: str,
    ) -> StationAdaptedHybrid:
        """Choose a source-validation fusion using raw-scale MAE.

        Log-affine coefficients are ordinary least squares on source-validation
        log1p labels. Its fitted validation score is a selection score, not an
        independent performance estimate. Ties favor simpler earlier candidates.
        """
        self._require_source_validation(selection_role)
        context = _values(context_prediction, "context_prediction")
        temporal = _values(temporal_prediction, "temporal_prediction")
        y = _values(truth, "truth")
        if not len(y) or context.shape != y.shape or temporal.shape != y.shape:
            raise ValueError("source-validation expert predictions and labels must be aligned and nonempty")
        candidates = [
            {"name": "context", "coefficients": [0.0, 1.0, 0.0]},
            {"name": "temporal", "coefficients": [0.0, 0.0, 1.0]},
        ]
        for weight in self.blend_weights:
            if 0.0 < weight < 1.0:
                candidates.append({"name": "geometric", "weight": weight,
                                   "coefficients": [0.0, 1.0 - weight, weight]})
        design = np.column_stack([np.ones(len(y)), np.log1p(context), np.log1p(temporal)])
        if self.allow_affine:
            coefficients = np.linalg.lstsq(design, np.log1p(y), rcond=None)[0]
            candidates.append({"name": "log_affine", "coefficients": coefficients.tolist()})
        self.fusion_candidates_: list[dict] = []
        for candidate in candidates:
            if candidate["name"] == "context":
                prediction = context
            elif candidate["name"] == "temporal":
                prediction = temporal
            else:
                try:
                    prediction = _inverse(design @ candidate["coefficients"])
                except ValueError:
                    self.fusion_candidates_.append({**candidate, "mae": None, "valid": False})
                    continue
            self.fusion_candidates_.append({
                **candidate, "mae": float(np.abs(prediction - y).mean()), "valid": True,
            })
        self.fusion_ = min(
            (candidate for candidate in self.fusion_candidates_ if candidate["valid"]),
            key=lambda candidate: candidate["mae"],
        ).copy()
        self.fusion_selection_role_ = selection_role
        # A new fusion changes support residuals, invalidating prior shrinkage.
        self.alpha_by_arm_ = {"context": {0: 0.0}, "hybrid": {0: 0.0}}
        self.calibration_scores_: list[dict] = []
        if hasattr(self, "calibration_selection_role_"):
            del self.calibration_selection_role_
        return self

    def predict_components(
        self, context_prediction: np.ndarray, temporal_prediction: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not hasattr(self, "fusion_"):
            raise RuntimeError("fit_fusion must be called before prediction")
        context = _values(context_prediction, "context_prediction")
        temporal = _values(temporal_prediction, "temporal_prediction")
        if context.shape != temporal.shape:
            raise ValueError("expert predictions must be aligned")
        if self.fusion_["name"] == "context":
            hybrid = context.copy()
        elif self.fusion_["name"] == "temporal":
            hybrid = temporal.copy()
        else:
            intercept, context_weight, temporal_weight = self.fusion_["coefficients"]
            hybrid = _inverse(intercept + context_weight * np.log1p(context)
                              + temporal_weight * np.log1p(temporal))
        return {"context": context.copy(), "temporal": temporal.copy(), "hybrid": hybrid}

    def _check_support_k(self, query_cells: np.ndarray, support_cells: np.ndarray, k: int) -> None:
        if k not in (0, 1, 3, 5):
            raise ValueError("K must be 0, 1, 3 or 5")
        query = _cells(query_cells, "query_cells")
        support = _cells(support_cells, "support_cells")
        query_stations = np.unique(query // self.n_months)
        support_stations, counts = np.unique(support // self.n_months, return_counts=True)
        if k == 0:
            if len(support):
                raise ValueError("K=0 requires empty support")
        elif (not np.array_equal(query_stations, support_stations) or not (counts == k).all()):
            raise ValueError("each query station must have exactly K support cells")

    def fit_calibration(
        self, episodes: Iterable[CalibrationEpisode], *, selection_role: str,
    ) -> StationAdaptedHybrid:
        """Select each arm's per-K shrinkage from source-validation episodes only."""
        self._require_source_validation(selection_role)
        if not hasattr(self, "fusion_"):
            raise RuntimeError("fit_fusion must be called before calibration")
        scores: list[dict] = []
        for episode_index, episode in enumerate(episodes):
            self._check_support_k(episode.query_cells, episode.support_cells, episode.k)
            truth = _values(episode.query_values, "query_values")
            if not len(truth):
                raise ValueError("calibration queries must be nonempty")
            query = self.predict_components(episode.context_query_prediction, episode.temporal_query_prediction)
            support = self.predict_components(episode.context_support_prediction, episode.temporal_support_prediction)
            if truth.shape != query["context"].shape:
                raise ValueError("calibration query labels and predictions must be aligned")
            alphas = (0.0,) if episode.k == 0 else self.shrinkage_weights
            for alpha in alphas:
                row = {"episode": episode_index, "k": episode.k, "alpha": alpha}
                for arm in ("context", "hybrid"):
                    prediction = station_residual_correction(
                        query[arm], episode.query_cells, support[arm], episode.support_cells,
                        episode.support_values, n_months=self.n_months, alpha=alpha,
                    )
                    row[f"{arm}_mae"] = float(np.abs(prediction - truth).mean())
                scores.append(row)
        if not scores:
            raise ValueError("at least one source-validation calibration episode is required")
        selected = {"context": {0: 0.0}, "hybrid": {0: 0.0}}
        for arm, per_k in selected.items():
            for k in sorted({row["k"] for row in scores}):
                alphas = sorted({row["alpha"] for row in scores if row["k"] == k})
                per_k[k] = min(alphas, key=lambda alpha: np.mean([
                    row[f"{arm}_mae"] for row in scores if row["k"] == k and row["alpha"] == alpha
                ]))
        self.alpha_by_arm_ = selected
        self.calibration_scores_ = scores
        self.calibration_selection_role_ = selection_role
        return self

    def adapt(
        self,
        query_prediction: np.ndarray,
        query_cells: np.ndarray,
        support_prediction: np.ndarray,
        support_cells: np.ndarray,
        support_values: np.ndarray,
        *,
        k: int,
        arm: str,
    ) -> np.ndarray:
        """Apply the specified arm's frozen shrinkage without query labels."""
        self._check_support_k(query_cells, support_cells, k)
        if arm not in self.alpha_by_arm_:
            raise ValueError("arm must be context or hybrid")
        if k not in self.alpha_by_arm_[arm]:
            raise RuntimeError(f"shrinkage for {arm} K={k} has not been fitted on source validation")
        return station_residual_correction(
            query_prediction, query_cells, support_prediction, support_cells, support_values,
            n_months=self.n_months, alpha=self.alpha_by_arm_[arm][k],
        )

    def to_dict(self) -> dict:
        """Return JSON-compatible fitted parameters and validation diagnostics."""
        if not hasattr(self, "fusion_"):
            raise RuntimeError("fit_fusion must be called before serialization")
        return deepcopy({
            "schema_version": 1,
            "n_months": self.n_months,
            "blend_weights": list(self.blend_weights),
            "shrinkage_weights": list(self.shrinkage_weights),
            "allow_affine": self.allow_affine,
            "fusion": self.fusion_,
            "fusion_candidates": self.fusion_candidates_,
            "fusion_selection_role": self.fusion_selection_role_,
            "alpha_by_arm": self.alpha_by_arm_,
            "calibration_scores": self.calibration_scores_,
            "calibration_selection_role": getattr(self, "calibration_selection_role_", None),
        })

    @classmethod
    def from_dict(cls, state: dict) -> StationAdaptedHybrid:
        """Restore frozen parameters; this method never refits from test data."""
        if state.get("schema_version") != 1:
            raise ValueError("unsupported StationAdaptedHybrid schema")
        model = cls(n_months=state["n_months"], blend_weights=state["blend_weights"],
                    shrinkage_weights=state["shrinkage_weights"], allow_affine=state["allow_affine"])
        model._require_source_validation(state["fusion_selection_role"])
        fusion = deepcopy(state["fusion"])
        coefficients = np.asarray(fusion["coefficients"], dtype=np.float64)
        if (fusion["name"] not in {"context", "temporal", "geometric", "log_affine"}
                or coefficients.shape != (3,) or not np.isfinite(coefficients).all()
                or not np.isfinite(fusion["mae"]) or fusion["mae"] < 0):
            raise ValueError("invalid serialized fusion parameters")
        model.fusion_ = fusion
        model.fusion_candidates_ = deepcopy(state["fusion_candidates"])
        model.fusion_selection_role_ = state["fusion_selection_role"]
        if set(state["alpha_by_arm"]) != {"context", "hybrid"}:
            raise ValueError("serialized shrinkage must include context and hybrid")
        for arm, parameters in state["alpha_by_arm"].items():
            parsed = {int(k): float(alpha) for k, alpha in parameters.items()}
            if (parsed.get(0) != 0.0 or not set(parsed).issubset({0, 1, 3, 5})
                    or not all(np.isfinite(alpha) and 0 <= alpha <= 1 for alpha in parsed.values())):
                raise ValueError("invalid serialized shrinkage parameters")
            model.alpha_by_arm_[arm] = parsed
        calibration_role = state.get("calibration_selection_role")
        if calibration_role is not None:
            model._require_source_validation(calibration_role)
            model.calibration_selection_role_ = calibration_role
        elif any(set(parameters) != {0} for parameters in model.alpha_by_arm_.values()):
            raise ValueError("fitted shrinkage needs source-validation selection metadata")
        model.calibration_scores_ = deepcopy(state["calibration_scores"])
        return model
