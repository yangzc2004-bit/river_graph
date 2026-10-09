"""Compact source-OOF DOC bias correction with observable hydro conditions."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

from river_graph.models.causal_flow_features import build_causal_flow_features

HYDRO_NAMES = ("temperature", "signed_log_discharge", "temperature_visible",
               "discharge_visible", "flow_anomaly", "flow_history_fraction")


def hydro_calibration_inputs(dataset):
    """Use hydro only; hidden values and future months cannot enter a row."""
    x, mask = np.asarray(dataset["x"], dtype=float), np.asarray(dataset["x_mask"])
    if (x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape
            or not np.isin(mask, [0, 1]).all() or not np.isfinite(x[mask.astype(bool)]).all()):
        raise ValueError("aligned finite observed temperature/discharge are required")
    visible = mask.astype(bool)
    values = np.where(visible, x, 0.)
    flow = build_causal_flow_features({"x": x, "x_mask": mask})["full"]
    discharge = np.sign(values[..., 1])*np.log1p(np.abs(values[..., 1]))
    return np.stack([values[..., 0], discharge, visible[..., 0], visible[..., 1],
                     flow[..., 0], flow[..., 6]], axis=-1).astype(np.float64)


def _vectors(base, hydro):
    p, h = np.asarray(base, dtype=float), np.asarray(hydro, dtype=float)
    if (p.ndim != 1 or not len(p) or h.shape != (len(p), len(HYDRO_NAMES))
            or not np.isfinite(p).all() or not np.isfinite(h).all() or (p < 0).any()
            or not np.isin(h[:, 2:4], [0, 1]).all()):
        raise ValueError("finite nonnegative predictions and aligned hydro rows are required")
    return p, h


class ConcentrationHydroCalibration:
    """Fit a small log-space correction by smoothed native-concentration MAE.

    Source station-hidden OOF predictions supply the reference, not in-sample
    fitted tree predictions. All statistics and coefficients use source training
    rows only. The conditional model has ten coefficients; the log-affine control
    has two. Ridge=.1 and feature clipping at three source SD are fixed, rather
    than selected from geographical or independent-basin outcomes.
    """

    def __init__(self, mode="concentration_hydro", *, ridge=.1, smooth_eps=.05):
        if mode not in {"log_affine", "concentration_hydro"}:
            raise ValueError("unsupported concentration calibration mode")
        if not np.isfinite([ridge, smooth_eps]).all() or ridge <= 0 or smooth_eps <= 0:
            raise ValueError("positive finite regularization and smoothing are required")
        self.mode, self.ridge, self.smooth_eps = mode, float(ridge), float(smooth_eps)

    def _design(self, base, hydro):
        p, h = _vectors(base, hydro)
        n = self.normalization_
        u = np.clip((np.log1p(p)-n["log_base_mean"])/n["log_base_sd"], -3., 3.)
        x = [np.ones(len(p)), u]
        if self.mode == "concentration_hydro":
            temperature = np.clip((h[:, 0]-n["temperature_mean"])/n["temperature_sd"], -3., 3.)*h[:, 2]
            discharge = np.clip((h[:, 1]-n["flow_mean"])/n["flow_sd"], -3., 3.)*h[:, 3]
            x += [np.maximum(0., u-n["concentration_knot"]), temperature, discharge,
                  h[:, 2], h[:, 3], h[:, 4], h[:, 5], u*h[:, 4]]
        return np.column_stack(x)

    def fit(self, source_oof_prediction, source_y, source_hydro, *, fit_role="source_training_oof"):
        if fit_role != "source_training_oof":
            raise ValueError("fit only source-training OOF residuals")
        p, h = _vectors(source_oof_prediction, source_hydro)
        y = np.asarray(source_y, dtype=float)
        if y.shape != p.shape or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("finite selected source labels must match OOF rows")
        z = np.log1p(p)

        def location(values):
            return (float(np.mean(values)), max(float(np.std(values)), 1e-6)) if len(values) else (0., 1.)

        zm, zs = location(z)
        tm, ts = location(h[h[:, 2].astype(bool), 0])
        qm, qs = location(h[h[:, 3].astype(bool), 1])
        self.normalization_ = {"log_base_mean": zm, "log_base_sd": zs,
            "temperature_mean": tm, "temperature_sd": ts, "flow_mean": qm, "flow_sd": qs,
            "concentration_knot": float(np.quantile(np.clip((z-zm)/zs, -3., 3.), .75)),
            "log_residual_scale": max(float(np.std(np.log1p(y)-z)), .1),
            "native_loss_scale": max(float(np.mean(np.abs(y-p))), .1),
            "statistic_role": fit_role}
        design = self._design(p, h)
        scale, loss_scale = self.normalization_["log_residual_scale"], self.normalization_["native_loss_scale"]

        def objective(theta):
            adjusted = z+scale*(design@theta)
            prediction = np.expm1(np.maximum(0., adjusted))
            error = (prediction-y)/loss_scale
            smooth = np.sqrt(error**2+self.smooth_eps**2)
            value = float(smooth.mean()+self.ridge*(theta@theta))
            derivative = error/smooth*(prediction+1.)*(adjusted > 0)*scale/loss_scale
            gradient = design.T@derivative/len(y)+2*self.ridge*theta
            return value, gradient

        self.optimization_ = []
        for maxls in (50, 200):
            result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                bounds=[(-3., 3.)]*design.shape[1],
                options={"maxiter": 300, "maxls": maxls, "ftol": 1e-11, "gtol": 1e-7})
            finite = bool(np.isfinite(result.x).all() and np.isfinite(result.fun) and np.isfinite(result.jac).all())
            self.optimization_.append({"success": bool(result.success), "finite": finite,
                "message": str(result.message), "iterations": int(result.nit), "maxls": maxls,
                "objective": float(result.fun) if finite else None})
            if result.success and finite:
                self.coefficients_ = result.x.copy()
                break
        else:
            raise RuntimeError(f"source calibration optimization failed: {self.optimization_}")
        self.fit_role_, self.n_source_rows_ = fit_role, len(y)
        self.source_mae_before_ = float(np.mean(np.abs(y-p)))
        self.source_mae_after_ = float(np.mean(np.abs(y-self.predict(p, h))))
        return self

    def predict(self, base_prediction, hydro):
        p, h = _vectors(base_prediction, hydro)
        design = self._design(p, h)
        if not np.any(self.coefficients_):
            return p.copy()
        adjusted = np.log1p(p)+self.normalization_["log_residual_scale"]*(design@self.coefficients_)
        prediction = np.expm1(np.maximum(0., adjusted))
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite calibrated DOC prediction")
        return prediction

    def to_dict(self):
        return {"schema": 1, "mode": self.mode, "ridge": self.ridge, "smooth_eps": self.smooth_eps,
            "fit_role": self.fit_role_, "n_source_rows": self.n_source_rows_,
            "hydro_names": list(HYDRO_NAMES), "normalization": deepcopy(self.normalization_),
            "coefficients": self.coefficients_.tolist(), "optimization": deepcopy(self.optimization_),
            "source_mae_before": self.source_mae_before_, "source_mae_after": self.source_mae_after_}

    @classmethod
    def from_dict(cls, state):
        if state["schema"] != 1 or state["fit_role"] != "source_training_oof" or state["hydro_names"] != list(HYDRO_NAMES):
            raise ValueError("unsupported source calibration state")
        model = cls(state["mode"], ridge=state["ridge"], smooth_eps=state["smooth_eps"])
        coefficients = np.asarray(state["coefficients"], dtype=float)
        if coefficients.shape != ((2,) if model.mode == "log_affine" else (10,)) or not np.isfinite(coefficients).all():
            raise ValueError("invalid calibration coefficients")
        model.normalization_, model.coefficients_ = deepcopy(state["normalization"]), coefficients.copy()
        model.optimization_, model.fit_role_ = deepcopy(state["optimization"]), state["fit_role"]
        model.n_source_rows_ = state["n_source_rows"]
        model.source_mae_before_, model.source_mae_after_ = state["source_mae_before"], state["source_mae_after"]
        return model
