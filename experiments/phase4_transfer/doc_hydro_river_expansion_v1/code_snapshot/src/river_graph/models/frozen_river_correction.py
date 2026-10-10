"""Support-matched river messages on a fixed complete DOC predictor.

The small regularized readout changes only the river correction. Receiver
labels are absent from bank construction and message features. It is fitted on
held-gradient validation stations and assessed on a separate whole HUC4.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from river_graph.models.river_structure_residual import (
    LAGS,
    path_candidates,
    upstream_paths,
)


def history_bank(residual, max_age=12):
    residual = np.asarray(residual, float)
    if residual.ndim != 2 or np.isinf(residual).any():
        raise ValueError("source values must be finite or missing station-months")
    values = np.zeros((*residual.shape, len(LAGS)))
    ages = np.full_like(values, max_age+1)
    last, age = np.zeros_like(residual), np.full_like(residual, max_age+1)
    for station, row in enumerate(residual):
        value, elapsed = 0., max_age+1
        for month, observed in enumerate(row):
            if np.isfinite(observed):
                value, elapsed = observed, 0
            else:
                elapsed += 1
            last[station, month], age[station, month] = value, elapsed
    for index, lag in enumerate(LAGS):
        if lag < residual.shape[1]:
            values[:, lag:, index] = last[:, :residual.shape[1]-lag]
            ages[:, lag:, index] = age[:, :residual.shape[1]-lag]
    valid = ages <= max_age
    return np.where(valid, values, 0), ages, valid


def matched_messages(edges, structures, receivers, bank_names, residual, *, candidates=20):
    """Replace source DOC identity, retaining slots, common support and ages.

    Donors are non-ancestors in the complete available station graph, without
    the primary 3000-km message cap. Drainage area and observation availability
    determine matching, never DOC concentrations or receiving outcomes.
    """
    names, receivers = np.asarray(bank_names, str), np.asarray(receivers, str)
    real = path_candidates(edges, structures, receivers, names, residual, candidates=candidates)
    values, ages, valid = history_bank(residual)
    ancestors = upstream_paths(edges, structures, receivers, names, candidates=len(names), max_km=np.inf)
    physical = structures.set_index("station")
    physical.index = physical.index.astype(str)
    area = np.log1p(physical.loc[names, "drainage_area_km2"].to_numpy(float))
    comids = physical.loc[names, "comid"].to_numpy()
    owners = real["river_owner"].copy()
    swapped = np.zeros_like(real["river_values"])
    swap_age = np.full_like(swapped, 13.)
    swap_valid = np.zeros_like(real["river_valid"])
    records = []
    for row, receiver in enumerate(receivers):
        excluded = {s for s, *_ in ancestors[row]} | {receiver}
        excluded_comids = set(physical.loc[list(excluded), "comid"])
        pool = [i for i, name in enumerate(names) if name not in excluded and comids[i] not in excluded_comids]
        for slot, original in enumerate(real["river_owner"][row]):
            if original < 0:
                continue
            if not pool:
                raise ValueError("insufficient non-ancestor donors for matching")
            scores = []
            for donor in pool:
                both = valid[original] & valid[donor]
                either = valid[original] | valid[donor]
                jaccard = both.sum()/max(1, either.sum())
                scores.append(abs(area[donor]-area[original])+2*(1-jaccard))
            chosen = pool.pop(int(np.argmin(scores)))
            owners[row, slot] = chosen
            swapped[row, :, slot] = values[chosen]
            swap_age[row, :, slot] = ages[chosen]
            swap_valid[row, :, slot] = valid[chosen]
            records.append({"receiver": receiver, "slot": slot, "real_source": names[original],
                "control_source": names[chosen], "source_is_nonancestor": True,
                "source_log_area_difference": abs(area[chosen]-area[original])})
    common = real["river_valid"] & swap_valid
    age = np.where(common, np.maximum(real["river_age"], swap_age), 0.)
    raw_valid = real["river_valid"].copy()
    true = {**real, "river_values": np.where(common, real["river_values"], 0.),
            "river_valid": common, "river_age": age}
    fake = {**true, "river_owner": owners,
            "river_values": np.where(common, swapped, 0.)}
    return true, fake, records, raw_valid


def message_features(messages, cells, baseline, daily, morphology, *, structured):
    """Lagged graph aggregation with optional path and whole-form interactions."""
    cells = np.asarray(cells, np.int64)
    n, t = np.asarray(baseline).shape
    if cells.ndim != 1 or (cells < 0).any() or (cells >= n*t).any():
        raise ValueError("cells must identify receiving station-months")
    row, month = cells//t, cells % t
    value = messages["river_values"][row, month]
    valid = messages["river_valid"][row, month]
    age = messages["river_age"][row, month]
    value = np.where(valid, value, 0.)
    denominator = valid.sum(1).clip(min=1)
    mean = value.sum(1)/denominator
    aged = (value*np.exp(-age/12)).sum(1)/denominator
    context = np.column_stack([np.ones(len(cells)), np.asarray(daily)[row, month],
                              np.log1p(np.asarray(baseline)[row, month])/5])
    core = np.concatenate([(mean[:, :, None]*context[:, None]).reshape(len(cells), -1), aged], axis=1)
    path = messages["river_path"][row]
    path_state = np.einsum("bcl,bcp->blp", value, path)/denominator[:, :, None]
    shape = mean[:, :, None]*np.asarray(morphology)[row, None]
    distance = np.expm1(path[..., 0]*np.log1p(3000))
    kernels = np.stack([np.exp(-distance/50), np.exp(-distance/200)], -1)
    decay = np.einsum("bcl,bcp->blp", value, kernels)/denominator[:, :, None]
    structural = np.concatenate([path_state.reshape(len(cells), -1), shape.reshape(len(cells), -1),
                                 decay.reshape(len(cells), -1)], axis=1)
    if not structured:
        structural[:] = 0.
    features = np.concatenate([core, structural], axis=1)*(1+np.asarray(baseline)[row, month, None])
    support = valid.any((1, 2))
    if not np.isfinite(features).all() or np.any(features[~support] != 0):
        raise ValueError("finite source-only features and exact zero unsupported rows required")
    return features, support


class FrozenRiverCorrection:
    """Station-held regularization of an additive native-concentration message.

    No intercept is fitted: zero source departures and absent observations give
    exactly zero correction. Alpha is selected on receiver-station CV, with the
    unchanged predictor available as the zero-correction option.
    """

    ALPHAS = (.1, 1., 10., 100., 1000.)

    @staticmethod
    def _fit(x, target, stations, alpha):
        scale = np.sqrt(np.mean(x*x, axis=0)).clip(min=1e-6)
        _, inverse, counts = np.unique(stations, return_inverse=True, return_counts=True)
        weight = len(stations)/(len(counts)*counts[inverse])
        fit = Ridge(alpha=alpha, fit_intercept=False).fit(x/scale, target, sample_weight=weight)
        return fit.coef_/scale

    def fit(self, x, truth, baseline, stations):
        x, truth, baseline = np.asarray(x, float), np.asarray(truth, float), np.asarray(baseline, float)
        stations = np.asarray(stations)
        if x.ndim != 2 or truth.shape != baseline.shape or len(x) != len(truth) or len(stations) != len(x):
            raise ValueError("calibration arrays must have identical rows")
        if not np.isfinite(x).all() or not np.isfinite(truth).all() or not np.isfinite(baseline).all():
            raise ValueError("only finite calibration inputs and labels may be fitted")
        self.coef_ = np.zeros(x.shape[1])
        self.scores_ = [{"alpha": None, "validation_mae": float(np.abs(truth-baseline).mean())}]
        self.alpha_ = None
        groups = len(np.unique(stations))
        self.n_calibration_stations_ = groups
        self.n_calibration_cells_ = len(x)
        if groups < 3 or not np.any(x):
            self.selection_reason_ = "insufficient supported calibration stations or messages"
            return self
        folds = list(GroupKFold(n_splits=3).split(x, groups=stations))
        for alpha in self.ALPHAS:
            prediction = np.empty(len(truth))
            for train, val in folds:
                coefficient = self._fit(x[train], (truth-baseline)[train], stations[train], alpha)
                prediction[val] = np.maximum(0., baseline[val]+x[val]@coefficient)
            self.scores_.append({"alpha": alpha, "validation_mae": float(np.abs(prediction-truth).mean())})
        best = min(self.scores_, key=lambda r: r["validation_mae"])
        self.alpha_ = best["alpha"]
        self.selection_reason_ = "three-fold station CV; zero correction competes with learned readouts"
        if self.alpha_ is not None:
            self.coef_ = self._fit(x, truth-baseline, stations, self.alpha_)
        return self

    def predict(self, x, baseline):
        x, base = np.asarray(x, float), np.asarray(baseline, float)
        if x.ndim != 2 or x.shape[1] != len(self.coef_) or len(base) != len(x):
            raise ValueError("prediction rows/features differ from fitted correction")
        result = np.maximum(0., base+x@self.coef_)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite river correction")
        return result

    def to_dict(self):
        return {"selected_alpha": self.alpha_, "selection_reason": self.selection_reason_,
                "station_cv_scores": self.scores_, "n_calibration_stations": self.n_calibration_stations_,
                "n_calibration_cells": self.n_calibration_cells_, "coefficient_norm": float(np.linalg.norm(self.coef_)),
                "frozen_complete_base": True, "no_intercept": True}
