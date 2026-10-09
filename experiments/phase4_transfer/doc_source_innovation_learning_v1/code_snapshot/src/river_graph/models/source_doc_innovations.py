"""Sparse, calendar-aligned source DOC departures for unmonitored receivers.

This library contains source training observations only. Its ecology and
seasonal statistics are fitted once; inference accepts no receiving labels.
The historical control draws exclusively from earlier same-season readings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def source_residual_grid(truth, oof_log, source_cells):
    """Select permitted source cells before inspecting labels or OOF values."""
    truth, oof_log, cells = np.asarray(truth), np.asarray(oof_log), np.asarray(source_cells)
    if (truth.ndim != 2 or oof_log.shape != truth.shape or cells.ndim != 1
            or cells.dtype.kind not in "iu" or not len(cells)
            or (cells < 0).any() or (cells >= truth.size).any()
            or len(np.unique(cells)) != len(cells)):
        raise ValueError("aligned grids and unique source cells are required")
    y, base = truth.ravel()[cells], oof_log.ravel()[cells]
    if not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(base).all():
        raise ValueError("source labels and OOF predictions must be finite")
    values = y-np.maximum(0., np.expm1(base))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite source residual")
    stations = np.unique(cells//truth.shape[1])
    residual = np.full((len(stations), truth.shape[1]), np.nan)
    residual[np.searchsorted(stations, cells//truth.shape[1]), cells % truth.shape[1]] = values
    return stations, residual


def corrected_prediction(base, innovation, alpha):
    base, innovation = np.asarray(base, float), np.asarray(innovation, float)
    if (base.shape != innovation.shape or not np.isfinite(base).all() or (base < 0).any()
            or not np.isfinite(innovation).all() or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("finite aligned nonnegative base and alpha within [0,1] required")
    # Preserve the baseline exactly when validation selects no correction.
    return base.copy() if alpha == 0 else np.maximum(0., base+alpha*innovation)


def select_correction(base, innovation, validation_y, *, selection_role="source_validation"):
    if selection_role != "source_validation":
        raise ValueError("correction selection requires source_validation")
    y = np.asarray(validation_y, float)
    if y.shape != np.shape(base) or not y.size or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("selected validation labels must align with predictions")
    candidates = [{"alpha": alpha, "mae": float(np.abs(corrected_prediction(base, innovation, alpha)-y).mean())}
                  for alpha in (0., .25, .5, 1.)]
    chosen = min(candidates, key=lambda row: (row["mae"], row["alpha"]))
    return {"alpha": chosen["alpha"], "validation_mae": chosen["mae"],
            "selection_role": selection_role, "candidates": candidates}


def _calendar(months):
    values = np.asarray(months, str)
    if values.ndim != 1 or not len(values):
        raise ValueError("nonempty monthly calendar required")
    periods = pd.PeriodIndex(values, freq="M")
    if periods.hasnans or periods.has_duplicates:
        raise ValueError("unique finite calendar months required")
    return periods.asi8, periods.month.to_numpy()


class SourceDOCInnovationLibrary:
    """Retrieve native DOC innovations from at most20 ecological neighbours.

    ``ecology`` uses the retained encoder's already-normalized nine features.
    ``residuals`` is source-only station-blocked OOF DOC minus its reference.
    Fit statistics may use the full permitted source-training record, as in the
    spatial task. With those statistics fixed, prediction never reads later
    source observations. This is not a temporal holdout training protocol.
    """

    def __init__(self, *, candidate_count=20, control_seed=42):
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
            raise ValueError("positive integer candidate_count required")
        self.candidate_count, self.control_seed = candidate_count, int(control_seed)

    def fit(self, source_names, months, ecology, residuals):
        names = np.asarray(source_names, str)
        eco, residual = np.asarray(ecology, float), np.asarray(residuals, float)
        ordinal, season = _calendar(months)
        if (names.ndim != 1 or len(names) < 2 or len(np.unique(names)) != len(names)
                or eco.shape != (len(names), 9) or not np.isfinite(eco).all()
                or residual.shape != (len(names), len(ordinal)) or np.isinf(residual).any()
                or not np.isfinite(residual).any(axis=1).all() or not (np.diff(ordinal) > 0).all()):
            raise ValueError("unique source IDs, sorted months and finite source ecology/residuals required")
        self.source_names_ = names.copy()
        self.months_, self.month_ordinals_ = np.asarray(months, str).copy(), ordinal.copy()
        self.ecology_ = eco.copy()
        means = np.nanmean(residual, axis=1)
        self.seasonal_mean_ = np.repeat(means[:, None], 12, axis=1)
        for month in range(1, 13):
            selected = residual[:, season == month]
            count = np.isfinite(selected).sum(axis=1)
            self.seasonal_mean_[:, month-1] = np.divide(
                np.nansum(selected, axis=1), count, out=means.copy(), where=count > 0)
        self.innovations_ = residual-self.seasonal_mean_[:, season-1]
        distances = np.sqrt(((eco[:, None]-eco[None])**2).sum(-1))
        np.fill_diagonal(distances, np.inf)
        nearest = np.sort(distances, axis=1)[:, :min(5, len(names)-1)]
        positive = nearest[(nearest > 0) & np.isfinite(nearest)]
        self.sigma_ = float(np.median(positive)) if len(positive) else 1.
        return self

    def historical_control(self):
        """Freeze earlier-year same-month donors without reading future values."""
        out = np.full_like(self.innovations_, np.nan)
        chosen_month = np.full(self.innovations_.shape, -1, dtype=np.int64)
        season = self.month_ordinals_ % 12
        for row, values in enumerate(self.innovations_):
            # Later availability at another source cannot alter this source's
            # earlier control draws through a shared random-stream position.
            rng = np.random.default_rng(np.random.SeedSequence([self.control_seed, row]))
            prior = [[] for _ in range(12)]
            for t, value in enumerate(values):
                if np.isfinite(value):
                    previous = prior[int(season[t])]
                    if previous:
                        index = previous[int(rng.integers(len(previous)))]
                        out[row, t], chosen_month[row, t] = values[index], self.month_ordinals_[index]
                    previous.append(t)
        return out, chosen_month

    def predict_components(self, receiver_names, receiver_ecology, months):
        names, eco = np.asarray(receiver_names, str), np.asarray(receiver_ecology, float)
        ordinals, _ = _calendar(months)
        if (names.ndim != 1 or len(np.unique(names)) != len(names)
                or eco.shape != (len(names), 9) or not np.isfinite(eco).all()):
            raise ValueError("unique receiver IDs and normalized nine-column ecology required")
        control, control_dates = self.historical_control()
        source_index = {int(month): t for t, month in enumerate(self.month_ordinals_)}
        result = {key: np.zeros((len(names), len(ordinals)), dtype=np.int64 if "count" in key else float)
                  for key in ("real_innovation", "historical_innovation", "support_count",
                              "all_current_support_count", "weight_mass")}
        donors = []
        for row, name in enumerate(names):
            distance = np.sqrt(((self.ecology_-eco[row])**2).sum(-1))
            pool = np.flatnonzero(self.source_names_ != name)
            order = np.argsort(distance[pool], kind="stable")[:self.candidate_count]
            candidates = pool[order]
            weights = np.exp(-distance[candidates]/self.sigma_)
            donors.append({"station": str(name), "source_stations": self.source_names_[candidates].tolist(),
                           "distances": distance[candidates].tolist(), "weights": weights.tolist()})
            for column, month in enumerate(ordinals):
                t = source_index.get(int(month))
                if t is None:
                    continue
                current, past = self.innovations_[candidates, t], control[candidates, t]
                available = np.isfinite(current)
                matched = available & np.isfinite(past)
                if (control_dates[candidates[matched], t] >= month).any():
                    raise ValueError("historical control reads a current/future source month")
                selected_weights = weights[matched]
                mass = float(selected_weights.sum())
                result["all_current_support_count"][row, column] = available.sum()
                result["support_count"][row, column] = matched.sum()
                result["weight_mass"][row, column] = mass
                result["real_innovation"][row, column] = selected_weights @ current[matched]/(1.+mass)
                result["historical_innovation"][row, column] = selected_weights @ past[matched]/(1.+mass)
        return {**result, "donors": donors}

    def save(self, path):
        np.savez_compressed(path, source_names=self.source_names_, months=self.months_,
            ecology=self.ecology_, seasonal_mean=self.seasonal_mean_, innovations=self.innovations_,
            sigma=np.array(self.sigma_), candidate_count=np.array(self.candidate_count),
            control_seed=np.array(self.control_seed))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            model = cls(candidate_count=int(saved["candidate_count"]), control_seed=int(saved["control_seed"]))
            model.source_names_, model.months_ = saved["source_names"].copy(), saved["months"].copy()
            model.month_ordinals_, _ = _calendar(model.months_)
            model.ecology_, model.seasonal_mean_ = saved["ecology"].copy(), saved["seasonal_mean"].copy()
            model.innovations_, model.sigma_ = saved["innovations"].copy(), float(saved["sigma"])
        return model

    def to_dict(self):
        return {"model": "source_DOC_innovation_library", "source_stations": len(self.source_names_),
            "months": len(self.months_), "source_observations": int(np.isfinite(self.innovations_).sum()),
            "candidate_count": self.candidate_count, "sigma": self.sigma_, "control_seed": self.control_seed,
            "zero_innovation_prior_weight": 1., "residual_units": "native mg/L",
            "calendar_mean": "source-only season means; station-mean fallback",
            "receiving_water_quality_required": False, "source_future_values_read_at_inference": False,
            "source_preprocessing_fixed_before_inference": True}
