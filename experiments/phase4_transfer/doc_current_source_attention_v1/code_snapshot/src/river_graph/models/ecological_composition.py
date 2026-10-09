"""Label-free detailed StreamCat land-cover inputs for new-station DOC."""
from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ("pctconif2019ws", "pctdecid2019ws", "pctmxfst2019ws", "pctcrop2019ws", "pcthay2019ws",
          "pcturbhi2019ws", "pcturbmd2019ws", "pcturblo2019ws", "pcturbop2019ws", "pctwdwet2019ws", "pcthbwet2019ws")
FAMILIES = ((0, 1, 2), (3, 4), (5, 6, 7, 8), (9, 10))
FEATURE_NAMES = (*FIELDS, *(f"{name}_valid" for name in FIELDS))


def _comid(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = text.removesuffix(".0")
    if not text.isdigit() or int(text) < 1:
        raise ValueError("COMID must be a positive integer identity")
    return str(int(text))


def composition_inputs(site_no, nodes, attributes):
    """Align eleven percentages and validity flags, without using target labels.

    StreamCat has one record per normalized COMID. Multiple stations may share
    a catchment. Real zero is valid; missing/out-of-range values become zero
    with validity zero. Percentages are divided by100, with no fitted scaler.
    The matched aggregate control repeats broad group totals at every class
    position and keeps the exact detailed-input validity flags.
    """
    if not {"site_no", "comid"} <= set(nodes) or not {"comid", *FIELDS} <= set(attributes):
        raise ValueError("station COMID mapping and all predefined composition fields are required")
    names = np.asarray(site_no, str)
    mapping = nodes.copy()
    mapping["site_no"] = mapping.site_no.astype(str)
    if not mapping.site_no.is_unique or len(np.unique(names)) != len(names):
        raise ValueError("station identity must be unique")
    mapping = mapping.set_index("site_no")
    if not set(names) <= set(mapping.index):
        raise ValueError("a model station is absent from the COMID mapping")
    table = attributes.copy()
    table["comid"] = table.comid.map(_comid)
    if table.comid.isna().any() or not table.comid.is_unique:
        raise ValueError("StreamCat COMID identity must be present and unique")
    table = table.set_index("comid")
    ids = mapping.loc[names, "comid"].map(_comid)
    raw = table.reindex(ids.tolist()).loc[:, FIELDS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(raw) & (raw >= 0) & (raw <= 100)
    fractions = np.where(valid, raw, 0.)/100.
    if (fractions.sum(axis=1) > 1.+1e-5).any():
        raise ValueError("mutually exclusive land-cover classes exceed100%")
    aggregate = np.zeros_like(fractions)
    totals = []
    for family in FAMILIES:
        total = fractions[:, family].sum(axis=1)
        aggregate[:, family] = total[:, None]
        totals.append(total)
    aggregate[~valid] = 0.
    return {"detailed": np.column_stack([fractions, valid]).astype(np.float32),
            "aggregate_control": np.column_stack([aggregate, valid]).astype(np.float32),
            "family_totals": np.column_stack(totals), "valid": valid,
            "matched_comid": ids.isin(table.index).to_numpy(), "site_no": names,
            "comid": ids.to_numpy(), "feature_names": list(FEATURE_NAMES)}
