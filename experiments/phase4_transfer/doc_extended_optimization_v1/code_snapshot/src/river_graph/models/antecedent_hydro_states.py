"""Causal hydro-climate descriptors for retrospective new-site reconstruction."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("seasonal_flow_anomaly", "seasonal_flow_anomaly_valid",
    "antecedent_flow_deficit", "antecedent_flow_deficit_valid",
    "flow_recovery_after_deficit", "flow_recovery_after_deficit_valid",
    "warming_after_cool_history", "warming_after_cool_history_valid")
VALUE_INDICES = (0, 2, 4, 6)


def antecedent_hydro_states(dataset):
    """Use only current/past x and x_mask; fit no statistics or target model.

    Flow anomaly compares current flow with the same month12/24 months earlier.
    Deficit compares the preceding3-month mean with the preceding12-month mean,
    requiring2/3 and3/12 observed months. Recovery is positive one-month relative
    flow change multiplied by positive deficit. Thermal recovery analogously
    combines positive one-month warming with preceding3-month cooling relative
    to preceding12 months. Temperature differences use the dataset's Celsius
    unit and a10-degree compression scale. Descriptors are proxies, not measured
    storage, flushing rates or causal coefficients. Negative flow is unavailable
    for these descriptors; old input channels remain unchanged.
    """
    x, mask = np.asarray(dataset["x"], float), np.asarray(dataset["x_mask"])
    if x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape or min(x.shape[:2]) < 1:
        raise ValueError("hydro states require aligned nonempty [station,month,2] x/x_mask")
    if not np.isin(mask, (0, 1)).all() or not np.isfinite(x[mask.astype(bool)]).all():
        raise ValueError("observed hydro values and binary visibility must be valid")
    available = mask.astype(bool)
    available[..., 1] &= x[..., 1] >= 0
    values = np.where(available, x, 0.)
    n, months = x.shape[:2]
    full = np.zeros((n, months, 8), dtype=np.float32)

    def history(t, width, channel):
        use = available[:, max(0, t-width):t, channel]
        count = use.sum(axis=1)
        mean = values[:, max(0, t-width):t, channel].sum(axis=1)/np.maximum(count, 1)
        return mean, count

    for t in range(months):
        q, q_valid = values[:, t, 1], available[:, t, 1]
        if t >= 24:
            reference = .5*(values[:, t-12, 1]+values[:, t-24, 1])
            valid = q_valid & available[:, t-12, 1] & available[:, t-24, 1] & (reference > 0)
            gap = q-reference
            full[:, t, 0] = np.divide(gap, reference+np.abs(gap), out=np.zeros(n), where=valid)
            full[:, t, 1] = valid
        long_flow, long_count = history(t, 12, 1)
        recent_flow, recent_count = history(t, 3, 1)
        deficit_valid = (long_count >= 3) & (recent_count >= 2) & (long_flow > 0)
        gap = np.maximum(0., long_flow-recent_flow)
        deficit = np.divide(gap, long_flow+gap, out=np.zeros(n), where=deficit_valid)
        full[:, t, 2], full[:, t, 3] = deficit, deficit_valid
        if t == 0:
            continue
        prior = values[:, t-1, 1]
        recovery_valid = deficit_valid & q_valid & available[:, t-1, 1] & (q+prior > 0)
        rising = np.divide(np.maximum(0., q-prior), q+prior, out=np.zeros(n), where=recovery_valid)
        full[:, t, 4], full[:, t, 5] = rising*deficit, recovery_valid
        long_temp, long_count = history(t, 12, 0)
        recent_temp, recent_count = history(t, 3, 0)
        valid = (long_count >= 3) & (recent_count >= 2) & available[:, t, 0] & available[:, t-1, 0]
        cold, warming = np.maximum(0., long_temp-recent_temp), np.maximum(0., values[:, t, 0]-values[:, t-1, 0])
        full[:, t, 6] = np.where(valid, cold/(10.+cold)*warming/(10.+warming), 0.)
        full[:, t, 7] = valid
    if not np.isfinite(full).all():
        raise ValueError("nonfinite causal hydro-climate descriptor")
    availability_only = full.copy()
    availability_only[..., VALUE_INDICES] = 0.
    return {"full": full, "availability_only": availability_only, "feature_names": list(FEATURE_NAMES)}
