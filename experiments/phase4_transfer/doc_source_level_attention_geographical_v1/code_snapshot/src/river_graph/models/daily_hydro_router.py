"""Preserve the monthly expert when numeric daily discharge inputs are absent.

Routing is applied to native base predictions before any ecological mixture or
K-shot adaptation. Those downstream operations may change either routed value;
their outputs are not asserted to equal the original monthly/daily experts.
"""

from __future__ import annotations

import numpy as np

from river_graph.models.daily_flow_features import (
    FEATURE_NAMES,
    VALIDITY_FEATURE_INDICES,
)


class DailyHydroRouter:
    """A fixed, label-free choice between existing monthly and daily experts.

    For any prediction shape ``S``, the daily block must have shape ``S + (8,)``.
    If at least one of flags 5, 6 and 7 is one, select the daily prediction;
    otherwise retain the monthly prediction. Adequately observed all-zero flow
    has valid flags and therefore uses the daily expert. Descriptor magnitudes
    and DOC labels are not used to determine the route. There is no fitting.
    """

    VERSION = 1
    RULE = "any_numeric_descriptor_valid_selects_daily_else_monthly"

    @staticmethod
    def _validated(monthly_prediction, daily_prediction, daily_features):
        monthly = np.asarray(monthly_prediction)
        daily = np.asarray(daily_prediction)
        features = np.asarray(daily_features)
        for name, values in (("monthly prediction", monthly), ("daily prediction", daily),
                             ("daily features", features)):
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{name} must be real numeric values")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite")
        if monthly.ndim < 1 or monthly.size == 0 or monthly.shape != daily.shape:
            raise ValueError("monthly and daily predictions must have the same nonempty shape")
        if features.shape != (*monthly.shape, len(FEATURE_NAMES)):
            raise ValueError("daily features must have prediction shape plus eight channels")
        flags = features[..., list(VALIDITY_FEATURE_INDICES)]
        if not np.isin(flags, (0, 1)).all():
            raise ValueError("numeric daily descriptor validity flags must be zero or one")
        count = flags.sum(axis=-1).astype(np.int8)
        return monthly, daily, count

    def predict(self, monthly_prediction, daily_prediction, daily_features) -> np.ndarray:
        """Return the selected native base value at each aligned cell."""
        monthly, daily, count = self._validated(monthly_prediction, daily_prediction, daily_features)
        return np.where(count > 0, daily, monthly)

    def predict_components(self, monthly_prediction, daily_prediction, daily_features) -> dict:
        """Expose the route and its two inputs without invoking downstream fits."""
        monthly, daily, count = self._validated(monthly_prediction, daily_prediction, daily_features)
        use_daily = count > 0
        return {"routed_pred": np.where(use_daily, daily, monthly),
                "uses_daily": use_daily, "daily_numeric_valid_count": count,
                "monthly_pred": monthly.copy(), "daily_pred": daily.copy()}

    def to_dict(self) -> dict:
        """Serialize the fixed semantics; no learned state is stored."""
        return {"model_class": type(self).__name__, "version": self.VERSION,
                "rule": self.RULE, "daily_feature_names": list(FEATURE_NAMES),
                "validity_feature_indices": list(VALIDITY_FEATURE_INDICES),
                "input_space": "native prediction units",
                "stage": "base routing before ecological mixture and support adaptation",
                "fitted_parameters": False, "target_label_dependency": "none"}

    @classmethod
    def from_dict(cls, payload: dict) -> DailyHydroRouter:
        """Reload only the specified version and observation-availability rule."""
        model = cls()
        if payload != model.to_dict():
            raise ValueError("daily hydro router metadata differs from the fixed supported rule")
        return model
