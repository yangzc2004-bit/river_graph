"""Source-normalized watershed ecology and concentration for the DOC readout.

Only station-blocked OOF context predictions fit concentration normalization.
Source rows receive those predictions at observed training cells; no full-fit
source predictions fill the remaining cells. Full-cohort inference uses the
provided fixed-context predictions. No DOC label values enter this builder.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from river_graph.models.causal_flow_features import FEATURE_NAMES as FLOW_FEATURE_NAMES

ECOLOGY_NAMES = ("forest", "agriculture", "urban", "wetland", "precipitation",
                 "log_temperature_normal", "soil_organic_matter", "elevation", "baseflow_index")
FEATURE_NAMES = (*FLOW_FEATURE_NAMES, *(f"ecology_{name}" for name in ECOLOGY_NAMES),
                 *(f"ecology_{name}_valid" for name in ECOLOGY_NAMES),
                 "context_log_concentration", "context_valid")
INTERACTION_INDICES = {
    "additive": (0, 2, 4),
    "concentration": (0, 2, 4, 28),
    "ecological": (0, 2, 4, *range(10, 19), 28),
}


def _compressed(value, center, scale):
    # Equivalent to z/(1+abs(z)), without forming a possibly huge z first.
    difference = value - center
    result = difference / (scale + np.abs(difference))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite compressed readout feature")
    return result


def build_regime_head_features(regime, source_cells, source_oof_native,
                               full_context_native, flow_full, *, n_months):
    """Return source/full ``[station, month, 30]`` float32 readout features.

    Regime columns 4:13 contain nine watershed ecological attributes. Source
    stations each contribute once to their median/IQR, irrespective of DOC
    record length. Missing (-1 or nonfinite) values are median-imputed; a
    zero IQR uses scale one. A column unobserved at every source station is
    inactive and has value zero everywhere. Its validity flags still report
    raw attribute availability. Validity is one for observed, zero for missing.

    Context normalization uses log1p of selected source OOF predictions with
    equal total weight per source station, population SD with floor 1e-6.
    Source context values/flags are zero outside the supplied source cells.
    Source station order is ascending; cells and OOF values are sorted jointly.

    Hydro causality belongs to ``build_causal_flow_features``. This function
    copies that supplied block without any temporal averaging. Context and
    ecological scalers are fixed source-training statistics, not online fits.
    """
    if isinstance(n_months, bool) or not isinstance(n_months, Integral) or n_months < 1:
        raise ValueError("n_months must be a positive integer")
    raw_regime = np.asarray(regime, dtype=np.float64)
    context = np.asarray(full_context_native, dtype=np.float64)
    flow = np.asarray(flow_full)
    if raw_regime.ndim != 2 or raw_regime.shape[1] < 13 or not len(raw_regime):
        raise ValueError("regime must contain station rows and at least 13 columns")
    shape = (len(raw_regime), int(n_months))
    if context.shape != shape or not np.isfinite(context).all() or (context < 0).any():
        raise ValueError("full_context_native must be a finite nonnegative [station, month] grid")
    if flow.shape != (*shape, 10) or not np.isfinite(flow).all():
        raise ValueError("flow_full must be finite [station, month, 10]")
    cells = np.asarray(source_cells)
    if (cells.ndim != 1 or cells.dtype.kind not in "iu" or not len(cells)
            or (cells < 0).any() or (cells >= context.size).any()
            or len(np.unique(cells)) != len(cells)):
        raise ValueError("source_cells must contain nonempty unique integer identities in the grid")
    oof = np.asarray(source_oof_native, dtype=np.float64)
    if oof.shape != cells.shape or not np.isfinite(oof).all() or (oof < 0).any():
        raise ValueError("source_oof_native must be finite nonnegative and align with selected source_cells")
    order = np.argsort(cells)
    cells, oof = cells[order].astype(np.int64, copy=False), oof[order]
    stations, inverse, counts = np.unique(cells // n_months, return_inverse=True, return_counts=True)

    ecology = raw_regime[:, 4:13]
    valid = np.isfinite(ecology) & (ecology != -1.0)
    median, iqr, active = np.zeros(9), np.ones(9), np.zeros(9, dtype=bool)
    for j in range(9):
        observed = ecology[stations, j][valid[stations, j]]
        if len(observed):
            active[j] = True
            median[j] = np.median(observed)
            q25, q75 = np.quantile(observed, [.25, .75])
            iqr[j] = q75 - q25 if q75 > q25 else 1.0
    values = _compressed(np.where(valid, ecology, median), median, iqr)
    values[:, ~active] = 0.0

    weights = 1.0 / (len(stations) * counts[inverse])
    oof_log = np.log1p(oof)
    log_mean = float(weights @ oof_log)
    log_sd = max(float(np.sqrt(weights @ ((oof_log - log_mean)**2))), 1e-6)
    if not np.isfinite(log_mean) or not np.isfinite(log_sd):
        raise FloatingPointError("nonfinite source context normalization")
    full = np.empty((*shape, 30), dtype=np.float32)
    full[..., :10] = flow
    full[..., 10:19] = values[:, None, :]
    full[..., 19:28] = valid[:, None, :]
    full[..., 28] = _compressed(np.log1p(context), log_mean, log_sd)
    full[..., 29] = 1.0
    source = full[stations].copy()
    source[..., 28:30] = 0.0
    source[inverse, cells % n_months, 28] = _compressed(oof_log, log_mean, log_sd)
    source[inverse, cells % n_months, 29] = 1.0
    if not np.isfinite(full).all() or not np.isfinite(source).all():
        raise FloatingPointError("nonfinite regime head feature")
    return {
        "version": 1, "source_extra": source, "full_extra": full,
        "source_station_ids": stations, "feature_names": list(FEATURE_NAMES),
        "interaction_indices": {arm: list(indices) for arm, indices in INTERACTION_INDICES.items()},
        "normalization": {
            "ecology": {"columns": list(range(4, 13)), "feature_names": list(ECOLOGY_NAMES),
                        "median": median.tolist(), "iqr": iqr.tolist(), "active": active.tolist(),
                        "missing": "-1 or nonfinite", "zero_iqr_scale": 1.0,
                        "imputation": "source unique-station median; zero if no observed source values",
                        "inactive_values": "zero at every station; raw validity flag retained",
                        "weighting": "each source station once"},
            "context": {"log_mean": log_mean, "log_sd": log_sd, "sd_floor": 1e-6,
                        "transform": "log1p(native context prediction)",
                        "weighting": "equal source stations, equal observed cells within station",
                        "source_cells": len(cells), "source_station_ids": stations.tolist(),
                        "source_station_counts": counts.tolist(),
                        "source_visibility": "selected OOF cells only; zero value/validity elsewhere",
                        "full_visibility": "provided context predictions at every station-month"},
            "compression": "z/(1+abs(z)); z uses the saved source-only center and scale",
            "validity": "one when observed; zero when missing",
        },
    }
