"""Label roles for zero-observation DOC reconstruction and geographical replication."""
from __future__ import annotations

import numpy as np

HUC4_BLOCKS = ("1013", "1019", "0708", "1030", "1101")
WATER_INPUT_KEYS = ("ph", "ph_mask", "spec_conductance", "spec_conductance_mask",
                    "chemistry", "chemistry_mask")


def development_labels(dataset, split):
    """Expose only source/validation labels; never materialise target truth in a fit."""
    shape = tuple(dataset["y"].shape)
    result = np.full(shape, np.nan, dtype=np.float64)
    for role in ("train", "val"):
        cells = np.asarray(split[role], dtype=np.int64)
        result.ravel()[cells] = np.asarray(dataset["y"]).ravel()[cells]
    return result


def strip_auxiliary_water(dataset):
    """Return the original DOC/hydro/ecology interface without auxiliary chemistry."""
    return {key: value for key, value in dataset.items() if key not in WATER_INPUT_KEYS}


def geographical_split(y_mask, huc_cd, target_huc4):
    """Rotate whole HUC4 roles, retaining sparse stations for the main K0 endpoint."""
    mask = np.asarray(y_mask, dtype=bool)
    if mask.ndim != 2 or len(huc_cd) != mask.shape[0] or target_huc4 not in HUC4_BLOCKS:
        raise ValueError("aligned station/month mask and one scheduled HUC4 are required")
    codes = np.asarray([str(value).split(".")[0].zfill(8)[:4] for value in huc_cd])
    validation = HUC4_BLOCKS[(HUC4_BLOCKS.index(target_huc4) + 1) % len(HUC4_BLOCKS)]
    cells = np.flatnonzero(mask)
    station = cells // mask.shape[1]
    split = {"test": cells[codes[station] == target_huc4],
             "val": cells[codes[station] == validation],
             "train": cells[~np.isin(codes[station], [target_huc4, validation])],
             "context": np.empty(0, dtype=np.int64)}
    if any(not len(split[role]) for role in ("train", "val", "test")):
        raise ValueError("each geographical role needs observed DOC cells")
    return split, {"target_huc4": target_huc4, "validation_huc4": validation,
                   "k0_query": "all valid DOC cells, including stations with fewer than six observations",
                   "curve_eligible": np.flatnonzero(mask.sum(1) >= 6).tolist()}


def validate_zero_observation_view(inputs, station_ids):
    """Check the existing M1 view has no local DOC values, visibility or history."""
    raw = np.asarray(inputs["raw"])[station_ids]
    support = np.asarray(inputs["support"])[station_ids]
    # The nine M1 fields follow the complete H2X base, including hydraulic
    # regime columns. Their offset is not necessarily10.
    width = raw.shape[-1]
    local_columns = [8, 9, width-9, width-7, *range(width-6, width-2)]
    if width < 19 or np.any(raw[..., local_columns] != 0) or np.any(support[..., 0] != 0):
        raise ValueError("unmonitored station contains local DOC information")
