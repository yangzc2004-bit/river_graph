"""Bounded, unit-invariant discharge features using current/past inputs only."""

from __future__ import annotations

from numbers import Integral

import numpy as np

FEATURE_NAMES = (
    "flow_relative_anomaly",          # 0: signed compressed prior-12 anomaly
    "flow_relative_anomaly_valid",    # 1
    "flow_change_1",                  # 2: symmetric relative one-month change
    "flow_change_1_valid",            # 3
    "flow_change_3",                  # 4: symmetric relative three-month change
    "flow_change_3_valid",            # 5
    "flow_past12_observed_fraction",  # 6: previous observed count / 12
    "flow_age_capped",                # 7: min(age, cap) / cap, including current
    "flow_age_valid",                 # 8: an observation has occurred by now
    "flow_current_valid",             # 9
)
VALUE_FEATURE_INDICES = (0, 2, 4)
FRESHNESS_FEATURE_INDICES = (1, 3, 5, 6, 7, 8, 9)
FLAG_FEATURE_INDICES = (1, 3, 5, 8, 9)
HISTORY_MONTHS = 12
MIN_HISTORY_OBSERVATIONS = 3


def build_causal_flow_features(dataset, *, age_cap_months=12) -> dict:
    """Return a named ``full[N,T,10]`` float32 discharge feature block.

    Only ``dataset['x'][:,:,1]`` and its corresponding ``x_mask`` are used.
    No DOC values, temperature, fitted statistics or future inputs are read.
    Columns are consecutive calendar months, as in the existing dataset grid.

    With observed current discharge q, let mu and a be the mean and mean
    absolute observed discharge in the previous 12 columns, excluding current.
    If >=3 past values exist and a>0, anomaly r=(q-mu)/a is compressed as
    r/(1+abs(r)); the algebraically equivalent (q-mu)/(a+abs(q-mu)) avoids
    an intermediate large ratio. Otherwise value=0 and its validity flag=0.

    Lag-l changes are (q-q_l)/(abs(q)+abs(q_l)), for l in {1,3}, requiring
    both observations and a positive denominator. Two observed zeros have an
    undefined relative scale: change=0, validity=0. Observed zero flows still
    count toward history and reset observation age. Negative flow signs are
    preserved; multiplying discharge by any positive unit conversion factor
    leaves the features unchanged, apart from floating-point rounding.

    History fraction is count/12, even near the beginning of the grid. Age is
    measured from the latest observation at or before current, capped then
    divided by ``age_cap_months``. Before any observation age=0 and age_valid=0.
    Value channels lie in [-1,1]; validity/freshness channels lie in [0,1].
    For the matched freshness-only ablation, zero ``value_feature_indices``
    and retain all other columns without changing feature dimensions.
    """
    if isinstance(age_cap_months, bool) or not isinstance(age_cap_months, Integral):
        raise TypeError("age_cap_months must be a positive integer")
    if age_cap_months < 1:
        raise ValueError("age_cap_months must be positive")
    inputs, masks = np.asarray(dataset["x"]), np.asarray(dataset["x_mask"])
    if (inputs.ndim != 3 or inputs.shape[-1] < 2 or masks.shape != inputs.shape
            or min(inputs.shape[:2]) < 1):
        raise ValueError("hydro x/x_mask must be aligned nonempty [station, month, channel] arrays")
    raw_flow = np.asarray(inputs[..., 1], dtype=np.float64)
    raw_mask = masks[..., 1]
    if not np.isin(raw_mask, (0, 1)).all():
        raise ValueError("flow visibility must contain only zero/one or boolean values")
    observed = raw_mask.astype(bool)
    if not np.isfinite(raw_flow[observed]).all():
        raise ValueError("observed discharge values must be finite")
    flow = np.where(observed, raw_flow, 0.0)
    n, months = flow.shape
    full = np.zeros((n, months, len(FEATURE_NAMES)), dtype=np.float32)
    last_observed = np.full(n, -1, dtype=np.int64)

    for month in range(months):
        current, current_valid = flow[:, month], observed[:, month]
        start = max(0, month - HISTORY_MONTHS)
        history = flow[:, start:month]
        history_valid = observed[:, start:month]
        count = history_valid.sum(axis=1)
        divisor = np.maximum(count, 1)
        mean = history.sum(axis=1) / divisor
        scale = np.abs(history).sum(axis=1) / divisor
        anomaly_valid = current_valid & (count >= MIN_HISTORY_OBSERVATIONS) & (scale > 0)
        difference = current - mean
        anomaly = np.divide(difference, scale + np.abs(difference), out=np.zeros(n),
                            where=anomaly_valid)
        full[:, month, 0], full[:, month, 1] = anomaly, anomaly_valid

        for lag, column in ((1, 2), (3, 4)):
            if month < lag:
                continue
            previous = flow[:, month - lag]
            denominator = np.abs(current) + np.abs(previous)
            valid = current_valid & observed[:, month - lag] & (denominator > 0)
            change = np.divide(current - previous, denominator, out=np.zeros(n), where=valid)
            full[:, month, column], full[:, month, column + 1] = change, valid

        full[:, month, 6] = count / HISTORY_MONTHS
        last_observed[current_valid] = month
        known = last_observed >= 0
        age = np.where(known, month - last_observed, 0)
        full[:, month, 7] = np.minimum(age, age_cap_months) / age_cap_months
        full[:, month, 8], full[:, month, 9] = known, current_valid

    if not np.isfinite(full).all():
        raise ValueError("nonfinite causal flow feature")
    return {"full": full, "version": 1, "age_cap_months": int(age_cap_months),
            "history_months": HISTORY_MONTHS, "min_history_observations": MIN_HISTORY_OBSERVATIONS,
            "feature_names": list(FEATURE_NAMES),
            "value_feature_indices": list(VALUE_FEATURE_INDICES),
            "freshness_feature_indices": list(FRESHNESS_FEATURE_INDICES),
            "flag_feature_indices": list(FLAG_FEATURE_INDICES)}
