"""Station-balanced ecological memory of held-station DOC prediction errors.

Source context predictions must be genuinely out of fold. Each donor station
has equal total weight regardless of its record length. The memory estimates
the same context residual as the fixed temporal model; their corrections are
therefore blended, rather than added as independent errors.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

MODES = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
ECOLOGY_NAMES = ("forest", "agriculture", "urban", "wetland", "precipitation",
                 "log_temperature_normal", "soil_organic_matter", "elevation", "baseflow_index")


def _native(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be a finite nonnegative one-dimensional array")
    return array


def _cells(values, name, size):
    array = np.asarray(values)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or not len(array)
            or (array < 0).any() or (array >= size).any()
            or len(np.unique(array)) != len(array)):
        raise ValueError(f"{name} must contain unique, nonempty integer cell identities within the grid")
    return array.astype(np.int64, copy=False)


def _positive_grid(values, name, *, integers=False):
    array = np.asarray(tuple(values))
    if (array.ndim != 1 or not len(array) or not np.isfinite(array).all()
            or (array <= 0).any() or (integers and array.dtype.kind not in "iu")):
        raise ValueError(f"{name} must be a nonempty positive {'integer ' if integers else ''}grid")
    return tuple(int(v) if integers else float(v) for v in np.unique(array))


def _blend(context, temporal, delta, gamma):
    if gamma == 0:
        return temporal.copy()
    prediction = np.maximum(0.0, context + (1.0 - gamma) * (temporal - context) + gamma * delta)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite ecological residual prediction")
    return prediction


class EcologicalResidualTransfer:
    """Fit a native-DOC residual profile using static ecological neighbors.

    ``fit`` accepts only selected source and source-validation label vectors.
    Source cells determine every scale and profile. Validation selects neighbor
    count, ridge strength and blending weight by pooled raw-scale MAE. Prediction
    uses cached station profiles and requires no label bank.

    The stored two coefficients are dimensionless: native residual prediction is
    ``residual_scale * (theta[0] + theta[1] * u)`` where ``u`` is standardized
    ``log1p(context)``. Bias modes have an exactly zero slope. The smooth-MAE
    objective is ``sum(weight * sqrt(scaled_error**2 + .05**2)) + ridge*theta²``.
    """

    def __init__(self, mode="ecological_affine", *, k_grid=(20, 40, 80),
                 ridge_grid=(.1, 1.0), gamma_grid=(0.0, .25, .5, 1.0)):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.k_grid = _positive_grid(k_grid, "k_grid", integers=True)
        self.ridge_grid = _positive_grid(ridge_grid, "ridge_grid")
        gamma = np.asarray(tuple(gamma_grid), dtype=np.float64)
        if (gamma.ndim != 1 or not len(gamma) or not np.isfinite(gamma).all()
                or (gamma < 0).any() or (gamma > 1).any() or not (gamma == 0).any()):
            raise ValueError("gamma_grid must be finite, within [0,1], and include exact zero")
        self.gamma_grid = tuple(float(v) for v in np.unique(gamma))

    def fit(self, regime, source_cells, source_oof_native, source_y, *, n_months,
            validation_cells, validation_y, validation_context, validation_temporal,
            selection_role="source_validation"):
        if selection_role != "source_validation":
            raise ValueError("selection_role must be source_validation")
        if isinstance(n_months, bool) or not isinstance(n_months, (int, np.integer)) or n_months < 1:
            raise ValueError("n_months must be a positive integer")
        raw = np.asarray(regime, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 13 or not len(raw):
            raise ValueError("regime must have station rows and at least 13 columns")
        self.n_nodes_, self.n_months_ = len(raw), int(n_months)
        source = _cells(source_cells, "source_cells", len(raw) * n_months)
        validation = _cells(validation_cells, "validation_cells", len(raw) * n_months)
        p = _native(source_oof_native, "source_oof_native")
        y = _native(source_y, "source_y")
        vy = _native(validation_y, "validation_y")
        vc = _native(validation_context, "validation_context")
        vt = _native(validation_temporal, "validation_temporal")
        if p.shape != source.shape or y.shape != source.shape:
            raise ValueError("selected source prediction/label vectors must align with source_cells")
        if any(v.shape != validation.shape for v in (vy, vc, vt)):
            raise ValueError("selected validation vectors must align with validation_cells")
        source_order, validation_order = np.argsort(source), np.argsort(validation)
        source, p, y = source[source_order], p[source_order], y[source_order]
        validation = validation[validation_order]
        vy, vc, vt = (v[validation_order] for v in (vy, vc, vt))
        source_ids, inverse, counts = np.unique(source // n_months, return_inverse=True, return_counts=True)
        validation_ids = np.unique(validation // n_months)
        if len(source_ids) < 2:
            raise ValueError("at least 2 source stations are needed for self-excluded memory")
        if np.intersect1d(source_ids, validation_ids).size:
            raise ValueError("source and source-validation stations must be disjoint")
        weights = 1.0 / (len(source_ids) * counts[inverse])
        z = np.log1p(p)
        mean = float(weights @ z)
        sd = max(float(np.sqrt(weights @ ((z - mean)**2))), 1e-6)
        residual_scale = max(float(weights @ np.abs(y - p)), 1e-6)
        self.normalization_ = {"log_context_mean": mean, "log_context_sd": sd,
                               "residual_scale": residual_scale,
                               "weighting": "equal source station; equal cells within station",
                               "context_sd_floor": 1e-6, "residual_scale_floor": 1e-6}
        ecology = raw[:, 4:13].copy()
        valid = np.isfinite(ecology) & (ecology != -1.0)
        medians, iqrs, active = np.zeros(9), np.ones(9), np.zeros(9, dtype=bool)
        for j in range(9):
            observed = ecology[source_ids, j][valid[source_ids, j]]
            if len(observed):
                active[j] = True
                medians[j] = np.median(observed)
                q25, q75 = np.quantile(observed, [.25, .75])
                iqrs[j] = q75 - q25 if q75 > q25 else 1.0
        standardized = (np.where(valid, ecology, medians) - medians) / iqrs
        standardized[:, ~active] = 0.0
        if not np.isfinite(standardized).all():
            raise FloatingPointError("nonfinite scaled ecology")
        self.ecology_scaler_ = {"columns": list(range(4, 13)), "feature_names": list(ECOLOGY_NAMES),
                                "median": medians.tolist(), "iqr": iqrs.tolist(),
                                "active": active.tolist(), "zero_iqr_scale": 1.0,
                                "missing": "-1 or nonfinite", "eligible_min_valid_features": 5,
                                "distance": "mean squared source-scaled ecological difference"}
        eligible = valid.sum(1) >= 5
        source_eligible = source_ids[eligible[source_ids]]
        # These temporary arrays never enter serialized state. A new station's
        # profile fits only donor observations, excluding its own source row.
        by_station = {int(station): np.flatnonzero(inverse == j) for j, station in enumerate(source_ids)}
        station_counts = {int(station): int(count) for station, count in zip(source_ids, counts)}
        u = (z - mean) / sd
        design = np.column_stack((np.ones(len(p)), u)) if self.mode.endswith("affine") else np.ones((len(p), 1))
        target = (y - p) / residual_scale
        coefficient_cache = {}

        def neighbors(station, k):
            remaining = source_ids[source_ids != station]
            fallback = None
            nearest = radius = None
            if self.mode.startswith("global"):
                donors = remaining
            else:
                pool = source_eligible[source_eligible != station]
                if not eligible[station]:
                    donors, fallback = remaining, "target_insufficient_ecology"
                elif not len(pool):
                    donors, fallback = remaining, "no_eligible_source_donors"
                else:
                    distance = np.mean((standardized[pool] - standardized[station])**2, axis=1)
                    order = np.lexsort((pool, distance))[:min(k, len(pool))]
                    donors = pool[order]
                    nearest, radius = float(distance[order[0]]), float(distance[order[-1]])
            size = len(donors)
            diagnostic = {"station": int(station), "donor_ids": donors.tolist(),
                          "donor_weights": [1.0 / size] * size,
                          "donor_source_counts": [station_counts[int(d)] for d in donors],
                          "donor_count": size, "effective_count": float(size), "max_weight": 1.0 / size,
                          "nearest_distance": nearest, "radius_distance": radius,
                          "fallback": fallback, "ecology_valid_features": int(valid[station].sum()),
                          "source_self_excluded": bool(station in by_station),
                          "eligible_source_donors": int(np.sum(source_eligible != station))}
            return donors, diagnostic

        def coefficients(donors, ridge):
            key = (ridge, tuple(sorted(int(station) for station in donors)))
            if key not in coefficient_cache:
                indices = np.concatenate([by_station[s] for s in key[1]])
                cell_weights = np.concatenate([np.full(len(by_station[s]), 1.0 / (len(donors) * len(by_station[s])))
                                               for s in key[1]])
                x, r = design[indices], target[indices]

                def objective(theta):
                    error = r - x @ theta
                    smooth = np.sqrt(error**2 + .05**2)
                    loss = float(cell_weights @ smooth + ridge * (theta @ theta))
                    gradient = -(x.T @ (cell_weights * error / smooth)) + 2.0 * ridge * theta
                    if not np.isfinite(loss) or not np.isfinite(gradient).all():
                        raise FloatingPointError("nonfinite ecological profile objective")
                    return loss, gradient

                result = minimize(objective, np.zeros(x.shape[1], dtype=np.float64), jac=True,
                                  method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
                if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
                    raise RuntimeError(f"ecological profile optimization failed: {result.message}")
                theta = np.zeros(2, dtype=np.float64)
                theta[:x.shape[1]] = result.x
                coefficient_cache[key] = (theta, {"objective": float(result.fun), "iterations": int(result.nit),
                                                   "gradient_max_abs": float(np.max(np.abs(result.jac))),
                                                   "success": True, "message": str(result.message)})
            return coefficient_cache[key]

        candidates = []
        ks = (None,) if self.mode.startswith("global") else tuple(reversed(self.k_grid))
        val_station = validation // n_months
        val_u = (np.log1p(vc) - mean) / sd
        for ridge in reversed(self.ridge_grid):
            for k in ks:
                delta = np.empty_like(vc)
                for station in validation_ids:
                    donors, _ = neighbors(station, k)
                    theta, _ = coefficients(donors, ridge)
                    at = val_station == station
                    delta[at] = residual_scale * (theta[0] + theta[1] * val_u[at])
                for gamma in self.gamma_grid:
                    score = float(np.mean(np.abs(_blend(vc, vt, delta, gamma) - vy)))
                    candidates.append({"k": k, "ridge": ridge, "gamma": gamma, "validation_mae": score})
        chosen = min(candidates, key=lambda row: (row["validation_mae"], row["gamma"] != 0,
                                                  -row["ridge"], -(row["k"] or 0), row["gamma"]))
        self.selected_ = deepcopy(chosen)
        self.selected_status_ = "prior_temporal_fallback" if chosen["gamma"] == 0 else "ecological_residual_blend"
        self.coefficients_ = np.empty((len(raw), 2), dtype=np.float64)
        self.station_diagnostics_ = []
        for station in range(len(raw)):
            donors, diagnostic = neighbors(station, chosen["k"])
            theta, optimizer = coefficients(donors, chosen["ridge"])
            self.coefficients_[station] = theta
            self.station_diagnostics_.append({**diagnostic, "optimizer": deepcopy(optimizer)})
        self.validation_trace_ = candidates
        self.selection_role_ = selection_role
        self.source_station_ids_ = source_ids.tolist()
        self.source_station_counts_ = counts.tolist()
        self.source_cells_ = source.tolist()
        self.validation_cells_ = validation.tolist()
        self.validation_station_ids_ = validation_ids.tolist()
        self.baseline_validation_mae_ = float(np.mean(np.abs(vt - vy)))
        return self

    def _grid_prediction(self, values, name):
        if not hasattr(self, "selected_"):
            raise RuntimeError("fit must precede prediction")
        result = np.asarray(values, dtype=np.float64)
        if (result.shape != (self.n_nodes_, self.n_months_) or not np.isfinite(result).all()
                or (result < 0).any()):
            raise ValueError(f"{name} must be a finite nonnegative full station-month grid")
        return result

    def predict_delta(self, context_native):
        context = self._grid_prediction(context_native, "context_native")
        u = (np.log1p(context) - self.normalization_["log_context_mean"]) / self.normalization_["log_context_sd"]
        delta = self.normalization_["residual_scale"] * (self.coefficients_[:, :1] + self.coefficients_[:, 1:] * u)
        if not np.isfinite(delta).all():
            raise FloatingPointError("nonfinite ecological memory delta")
        return delta

    def predict(self, context_native, temporal_native):
        context = self._grid_prediction(context_native, "context_native")
        temporal = self._grid_prediction(temporal_native, "temporal_native")
        if self.selected_["gamma"] == 0:
            return temporal.copy()
        return _blend(context, temporal, self.predict_delta(context), self.selected_["gamma"])

    def to_dict(self):
        if not hasattr(self, "selected_"):
            raise RuntimeError("fit must precede serialization")
        return deepcopy({
            "schema_version": 1, "mode": self.mode, "k_grid": list(self.k_grid),
            "ridge_grid": list(self.ridge_grid), "gamma_grid": list(self.gamma_grid),
            "n_nodes": self.n_nodes_, "n_months": self.n_months_, "selection_role": self.selection_role_,
            "selected": self.selected_, "selected_status": self.selected_status_,
            "validation_trace": self.validation_trace_, "baseline_validation_mae": self.baseline_validation_mae_,
            "normalization": self.normalization_, "ecology_scaler": self.ecology_scaler_,
            "coefficients": self.coefficients_.tolist(), "station_diagnostics": self.station_diagnostics_,
            "source_station_ids": self.source_station_ids_, "source_station_counts": self.source_station_counts_,
            "source_cells": self.source_cells_, "validation_cells": self.validation_cells_,
            "validation_station_ids": self.validation_station_ids_,
            "objective": "station-balanced mean sqrt(scaled_error^2 + 0.05^2) + ridge * squared coefficients",
            "coefficient_formula": "native_delta = residual_scale * (theta0 + theta1 * standardized_log_context)",
            "optimizer": {"name": "L-BFGS-B", "maxiter": 200, "ftol": 1e-12, "gtol": 1e-9,
                          "initialization": "exact zero", "dtype": "float64"},
            "selection_metric": "pooled source-validation K0 native MAE",
            "selection_ties": "gamma zero, then larger ridge, then larger k, then smaller gamma",
            "donor_rule": "equal donor-station weights; source receiving station always excluded",
        })

    @classmethod
    def from_dict(cls, state):
        if state.get("schema_version") != 1 or state.get("selection_role") != "source_validation":
            raise ValueError("unsupported ecological memory schema or selection role")
        model = cls(state["mode"], k_grid=state["k_grid"], ridge_grid=state["ridge_grid"],
                    gamma_grid=state["gamma_grid"])
        model.n_nodes_, model.n_months_ = int(state["n_nodes"]), int(state["n_months"])
        coeff = np.asarray(state["coefficients"], dtype=np.float64)
        if coeff.shape != (model.n_nodes_, 2) or not np.isfinite(coeff).all():
            raise ValueError("invalid ecological coefficient grid")
        if model.mode.endswith("bias") and np.any(coeff[:, 1] != 0):
            raise ValueError("bias memory must have zero slopes")
        norms = state["normalization"]
        if (not all(np.isfinite(norms[k]) for k in ("log_context_mean", "log_context_sd", "residual_scale"))
                or norms["log_context_sd"] <= 0 or norms["residual_scale"] <= 0):
            raise ValueError("invalid ecological normalization")
        if (state["selected"]["gamma"] not in model.gamma_grid
                or state["selected"]["ridge"] not in model.ridge_grid
                or len(state["station_diagnostics"]) != model.n_nodes_):
            raise ValueError("invalid ecological fitted selection/diagnostics")
        model.coefficients_ = coeff
        for key in ("selected", "selected_status", "validation_trace", "baseline_validation_mae", "normalization",
                    "ecology_scaler", "station_diagnostics", "source_station_ids", "source_station_counts",
                    "source_cells", "validation_cells", "validation_station_ids", "selection_role"):
            setattr(model, key + "_", deepcopy(state[key]))
        return model
