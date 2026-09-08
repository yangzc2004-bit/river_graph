"""Temporal aggregation: observations -> monthly (station x month) matrices.

Labels keep monthly resolution (the MVP prediction target is monthly-mean
DOC); features are monthly means of covariates over the same grid. This
avoids collapsing the dataset to one climatological value per station.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_monthly(obs: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate point observations to monthly means per station."""
    if obs.empty:
        return pd.DataFrame(columns=["site_no", "month", value_col])
    out = obs.copy()
    out["month"] = out["date"].values.astype("datetime64[M]")
    return (
        out.groupby(["site_no", "month"], as_index=False)[value_col]
        .mean()
        .sort_values(["site_no", "month"])
        .reset_index(drop=True)
    )


def to_matrix(
    monthly: pd.DataFrame, value_col: str, sites: list[str], months: pd.DatetimeIndex
) -> np.ndarray:
    """Pivot monthly means into an (n_sites, n_months) matrix, NaN = unobserved."""
    mat = np.full((len(sites), len(months)), np.nan)
    if monthly.empty:
        return mat
    site_idx = {s: i for i, s in enumerate(sites)}
    month_idx = {m: j for j, m in enumerate(months)}
    for r in monthly.itertuples(index=False):
        i = site_idx.get(r.site_no)
        j = month_idx.get(r.month)
        if i is not None and j is not None:
            mat[i, j] = getattr(r, value_col)
    return mat
