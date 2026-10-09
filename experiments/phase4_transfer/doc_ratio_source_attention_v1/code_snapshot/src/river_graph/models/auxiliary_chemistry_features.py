"""Observed pH/EC covariates for DOC reconstruction on an aligned monthly grid.

Auxiliary chemistry is explicitly available at DOC-held-out stations. These
inputs describe a DOC-missing, chemistry-observed setting, not an entirely
unmonitored station. No DOC values, split roles or fitted statistics are read.
Current-month means are retrospective monthly inputs, not advance forecasts.
"""
from __future__ import annotations

import copy

import numpy as np

FEATURE_NAMES = ("aux_ph_div14", "aux_log1p_conductance", "aux_ph_visible", "aux_ec_visible")
VALUE_FEATURE_INDICES = (0, 1)
AVAILABILITY_FEATURE_INDICES = (2, 3)
MODES = ("no_aux", "masks", "chemistry")
POLICY = {
    "visibility": "observed pH/EC are exogenous known covariates in all DOC split roles",
    "station_setting": "DOC held out; conventional water chemistry may be observed",
    "time_scope": "same calendar-month means only; no future-month input or forward fill",
    "units": {"ph": "standard units", "spec_conductance": "uS/cm"},
    "transforms": {"ph": "ph/14", "spec_conductance": "log1p(spec_conductance)"},
    "quality": {"ph_range": [0, 14], "spec_conductance_range": [0, 100000],
                "detection": "existing tensor builder excludes nonempty detection-condition records",
                "aggregation": "arithmetic mean of accepted measurements in each station/calendar month",
                "fractions_and_parameter_codes": "retain the existing tensor definitions without reclassification",
                "zero_conductance": "retained as an observed value under the existing QC policy"},
    "fitted_statistics": "none",
    "doc_value_dependency": "none",
    "active": "actual pH_visible OR EC_visible; identical across all three feature modes",
    "modes": {"no_aux": "four zeros, actual active retained separately",
              "masks": "numeric values zero, observed masks retained",
              "chemistry": "numeric values and observed masks retained"},
}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _grid(dataset):
    sites = np.asarray(dataset["site_no"]).astype(str)
    months = np.asarray(dataset["months"]).astype(str)
    if (sites.ndim != 1 or months.ndim != 1 or not len(sites) or not len(months)
            or len(set(sites)) != len(sites) or len(set(months)) != len(months)):
        raise ValueError("station/month identifiers must be nonempty, unique vectors")
    x, mask = _array(dataset["x"]), _array(dataset["x_mask"])
    if x.shape != (len(sites), len(months), 2) or mask.shape != x.shape:
        raise ValueError("hydro inputs must have aligned [station,month,2] shape")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("hydro visibility must be binary")
    edges = _array(dataset["edge_index"])
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu"
            or (edges < 0).any() or (edges >= len(sites)).any()):
        raise ValueError("edge_index must contain valid integer node indices")
    if "feature_channels" in dataset and list(dataset["feature_channels"]) != ["temperature", "discharge"]:
        raise ValueError("hydro channel order must be temperature, discharge")
    return sites, months, x, mask, edges


def _aligned(doc, auxiliary, name):
    reference, candidate = _grid(doc), _grid(auxiliary)
    for key, a, b in zip(("site_no", "months", "x", "x_mask", "edge_index"),
                         reference, candidate, strict=True):
        same = (np.array_equal(a, b, equal_nan=True) if key == "x"
                else np.array_equal(a, b))
        if not same:
            raise ValueError(f"{name} {key} must exactly match the DOC grid")
    if ("edge_attr" in doc) != ("edge_attr" in auxiliary):
        raise ValueError(f"{name} edge_attr presence must match DOC")
    if "edge_attr" in doc and not np.array_equal(_array(doc["edge_attr"]), _array(auxiliary["edge_attr"]),
                                                equal_nan=True):
        raise ValueError(f"{name} edge_attr must exactly match DOC")
    shape = (len(reference[0]), len(reference[1]))
    values, mask = _array(auxiliary["y"]), _array(auxiliary["y_mask"])
    if values.shape != shape or mask.shape != shape or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} labels and binary mask must align with the DOC grid")
    visible = mask.astype(bool)
    observed = values[visible]
    upper = 14 if name == "ph" else 100000
    if not np.isfinite(observed).all() or (observed < 0).any() or (observed > upper).any():
        raise ValueError(f"{name} observed values violate the existing finite/unit-range QC")
    # Hidden placeholders are irrelevant, including NaNs. Never log them.
    return np.where(visible, values, 0).astype(np.float64), visible


def apply_auxiliary_mode(full, mode):
    """Return a detached mode-specific block; never derive the active gate here.

    ``active`` must come from the original chemistry pack, because ``no_aux``
    deliberately zeroes its flag channels while retaining the same experiment
    routing footprint as both other arms.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    full = _array(full)
    if (full.ndim != 3 or full.shape[-1] != 4 or not np.isfinite(full).all()
            or not np.isin(full[..., 2:], (0, 1)).all()):
        raise ValueError("auxiliary features must be finite [station,month,4] with binary flags")
    output = full.astype(np.float32, copy=True)
    if mode == "no_aux":
        output.fill(0)
    elif mode == "masks":
        output[..., :2] = 0
    return output


def build_auxiliary_chemistry_features(doc, ph, ec, *, mode="chemistry"):
    """Return float32 ``full[N,T,4]`` and the shared boolean ``active[N,T]``.

    The DOC keys accessed are site_no/months/x/x_mask/edge_index, plus optional
    edge_attr/feature_channels. DOC y and y_mask are not used. The supplied
    auxiliary tensors must already use standard pH units and uS/cm under the
    preserved monthly aggregation/QC definition. Actual auxiliary observations
    are not hidden merely because their DOC target is held out.
    """
    ph_value, ph_mask = _aligned(doc, ph, "ph")
    ec_value, ec_mask = _aligned(doc, ec, "spec_conductance")
    full = np.stack((ph_value / 14, np.log1p(ec_value), ph_mask, ec_mask), axis=-1).astype(np.float32)
    return {"full": apply_auxiliary_mode(full, mode), "active": ph_mask | ec_mask,
            "feature_names": list(FEATURE_NAMES), "value_feature_indices": list(VALUE_FEATURE_INDICES),
            "availability_feature_indices": list(AVAILABILITY_FEATURE_INDICES),
            "mode": mode, "policy": copy.deepcopy(POLICY)}
