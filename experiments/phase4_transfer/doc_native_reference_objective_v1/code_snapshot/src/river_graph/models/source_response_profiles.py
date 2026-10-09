"""Robust source-station seasonal/hydro profiles of OOF DOC errors."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from river_graph.models.doc_source_retrieval import SourceResidualBank


class SourceResponseBank(SourceResidualBank):
    """Keep source preprocessing and retrieval candidates; strengthen profiles.

    Smooth native-MAE profiles match the reconstruction objective. Donors shrink
    toward the station-balanced population response with fixed strength 0.1.
    """

    def __init__(self, *, ridge=.1, max_donors=20):
        super().__init__(ridge=ridge, max_donors=max_donors)

    def fit(self, ecology, hydro, hydro_valid, months, cells, oof_prediction, truth,
            *, station_names, excluded_station_names=()):
        super().fit(ecology, hydro, hydro_valid, months, cells, oof_prediction, truth,
                    station_names=station_names, excluded_station_names=excluded_station_names)
        cells = np.asarray(cells, dtype=np.int64)
        periods = len(months)
        _, inverse, counts = np.unique(cells//periods, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[inverse])
        design = self.design(np.asarray(oof_prediction), np.asarray(hydro)[cells//periods, cells % periods],
            np.asarray(hydro_valid)[cells//periods, cells % periods], np.asarray(months)[cells % periods])
        residual = (np.asarray(truth)-oof_prediction)/self.residual_scale_

        def solve(x, y, w, anchor):
            def objective(coef):
                error = x@coef-y
                smooth = np.sqrt(error**2+.05**2)
                penalty = coef-anchor
                return float(w@smooth+self.ridge*(penalty@penalty)), x.T@(w*error/smooth)+2*self.ridge*penalty
            result = minimize(objective, anchor.copy(), jac=True, method="L-BFGS-B",
                              options={"maxiter": 200, "maxls": 100, "ftol": 1e-12})
            if not result.success or not np.isfinite(result.x).all():
                raise RuntimeError(f"source response optimization failed: {result.message}")
            return result.x

        population = solve(design, residual, weights, np.zeros(6))
        profiles = []
        for station, count in enumerate(counts):
            selected = inverse == station
            profiles.append(solve(design[selected], residual[selected], np.full(count, 1/count), population))
        self.profiles_ = np.asarray(profiles)
        self.keys_[..., -6:] = self.profiles_
        self.profile_method_ = "smooth_native_MAE_station_balanced_population_shrinkage"
        return self
