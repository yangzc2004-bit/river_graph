"""Within-month discharge descriptors on the frozen monthly hydro footprint.

Only daily discharge, station/month identifiers and the monthly discharge mask
are consumed. Current-month descriptors are retrospective reconstruction inputs,
not forecasts made before the current month has finished.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "daily_flow_width", "daily_flow_flashiness", "daily_flow_rising_fraction",
    "daily_flow_day_fraction", "daily_flow_pair_fraction",
    "daily_flow_width_valid", "daily_flow_flashiness_valid", "daily_flow_rising_valid",
)
VALUE_FEATURE_INDICES = (0, 1, 2)
AVAILABILITY_FEATURE_INDICES = (3, 4, 5, 6, 7)
VALIDITY_FEATURE_INDICES = (5, 6, 7)
MAX_ABS_DISCHARGE_CFS = 3_000_000.0
COVERAGE_FRACTION = 0.8
POLICY = {
    "parameter_code": "00060", "statistic_code": "00003", "raw_unit": "cfs",
    "series_selection": "first matching _00060_00003 column in each RDB header; alternatives ignored",
    "daily_qc": "finite discharge with abs(discharge_cfs) <= 3000000; signed and zero retained",
    "qualifiers": "retained in inventory, not filtered; same finite-value policy as frozen monthly data",
    "duplicate_policy": "equal accepted station-day values collapse; conflicting accepted values exclude day",
    "gap_policy": "no filling; pairs require exactly consecutive days in the same calendar month",
    "quantile_method": "linear", "coverage_fraction": COVERAGE_FRACTION,
    "width_requirement": "n_unique_valid_days >= ceil(0.8 * calendar_days)",
    "pair_requirement": "n_true_within_month_pairs >= ceil(0.8 * (calendar_days - 1))",
    "all_zero_flow": "valid ratios are zero with validity flag one when coverage is sufficient",
    "monthly_footprint": "all eight features zero wherever frozen x_mask[:,:,1] is zero",
    "fitted_statistics": "none", "target_label_dependency": "none",
    "time_scope": "same calendar month only; retrospective current-month covariate",
    "formulas": {
        "width": "(Q90-Q10) / (mean(abs(q)) + Q90-Q10)",
        "flashiness": "sum(abs(q_current-q_previous)) / sum(abs(q_current)+abs(q_previous)) over true pairs",
        "rising_fraction": "count(q_current > q_previous) / n_true_pairs",
        "day_fraction": "n_unique_valid_days / calendar_days",
        "pair_fraction": "n_true_within_month_pairs / (calendar_days-1)",
    },
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_first_daily_discharge(cache_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read the first mean-discharge series per header, with an explicit audit.

    Alternative series are recorded but never pooled or used to fill gaps. Rows
    remain duplicated here; :func:`build_daily_flow_features` applies daily QC
    and station-day reconciliation. Qualifiers do not change the frozen policy.
    """
    paths = sorted(Path(cache_dir).glob("dv_*.rdb"))
    if not paths:
        raise FileNotFoundError(f"no dv_*.rdb files under {cache_dir}")
    frames, inventory, multiple_series = [], [], []
    for path in paths:
        sites, dates, values, qualifiers = [], [], [], []
        headers = 0
        selected = None
        with path.open(encoding="utf-8", errors="replace", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if not row or row[0].startswith("#"):
                    continue
                if row[0] == "agency_cd":
                    headers += 1
                    matches = [i for i, name in enumerate(row) if name.endswith("_00060_00003")]
                    selected = None
                    if matches:
                        column = matches[0]
                        qualifier = row.index(row[column] + "_cd") if row[column] + "_cd" in row else None
                        selected = (row.index("site_no"), row.index("datetime"), column, qualifier)
                        if len(matches) > 1:
                            multiple_series.append({"path": str(path), "header": headers,
                                                    "selected": row[column],
                                                    "ignored": [row[i] for i in matches[1:]]})
                    continue
                if selected is None or row[0] != "USGS":
                    continue
                si, di, vi, qi = selected
                if len(row) <= max(si, di, vi):
                    raise ValueError(f"short daily-discharge row in {path}")
                sites.append(row[si])
                dates.append(row[di])
                values.append(row[vi])
                qualifiers.append(row[qi] if qi is not None and len(row) > qi else "")
        quality_counts = dict(Counter(qualifiers))
        frame = pd.DataFrame({"site_no": pd.Categorical(sites),
                              "date": pd.to_datetime(dates, errors="coerce", format="mixed"),
                              "discharge_cfs": pd.to_numeric(pd.Series(values), errors="coerce")})
        frames.append(frame)
        inventory.append({"path": str(path), "sha256": file_sha256(path),
                          "bytes": path.stat().st_size, "headers": headers,
                          "first_series_rows": len(frame), "qualifier_counts": quality_counts})
    daily = pd.concat(frames, ignore_index=True)
    daily["site_no"] = daily["site_no"].astype("category")
    return daily, {"raw_cache_files": inventory,
                   "multiple_series_headers": multiple_series,
                   "raw_first_series_rows": len(daily)}


def reconcile_daily_discharge(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply frozen magnitude QC and reduce to conflict-free station-days."""
    required = {"site_no", "date", "discharge_cfs"}
    if not required.issubset(daily.columns):
        raise ValueError(f"daily discharge requires {sorted(required)}")
    dates = pd.to_datetime(daily["date"], errors="coerce", format="mixed")
    if dates.dt.tz is not None:
        raise ValueError("daily dates must use an unambiguous timezone-free calendar")
    values = pd.to_numeric(daily["discharge_cfs"], errors="coerce").to_numpy(dtype=np.float64)
    date_valid = dates.notna().to_numpy()
    value_valid = np.isfinite(values) & (np.abs(values) <= MAX_ABS_DISCHARGE_CFS)
    site_valid = daily["site_no"].notna().to_numpy()
    accepted = date_valid & value_valid & site_valid
    cleaned = pd.DataFrame({"site_no": daily.loc[accepted, "site_no"].astype(str).to_numpy(),
                            "date": dates[accepted].dt.normalize().to_numpy(),
                            "discharge_cfs": values[accepted]})
    summary = {"input_rows": len(daily), "invalid_date_rows": int((~date_valid).sum()),
               "invalid_site_rows": int((~site_valid).sum()),
               "nonfinite_value_rows": int((~np.isfinite(values)).sum()),
               "magnitude_excluded_rows": int((np.isfinite(values) & ~value_valid).sum()),
               "accepted_rows_before_reconciliation": int(accepted.sum())}
    if cleaned.empty:
        summary.update({"duplicated_station_days": 0, "duplicate_excess_rows": 0,
                        "conflicting_station_days": 0, "conflicting_rows": 0,
                        "unique_valid_station_days": 0, "negative_days": 0, "zero_days": 0,
                        "calendar_start": None, "calendar_end": None})
        return cleaned, summary
    grouped = cleaned.groupby(["site_no", "date"], observed=True)["discharge_cfs"].agg(
        ["min", "max", "size"])
    conflict = grouped["min"] != grouped["max"]
    summary.update({"duplicated_station_days": int((grouped["size"] > 1).sum()),
                    "duplicate_excess_rows": int((grouped["size"] - 1).sum()),
                    "conflicting_station_days": int(conflict.sum()),
                    "conflicting_rows": int(grouped.loc[conflict, "size"].sum())})
    cleaned = grouped.loc[~conflict, ["min"]].reset_index().rename(columns={"min": "discharge_cfs"})
    summary.update({"unique_valid_station_days": len(cleaned),
                    "negative_days": int((cleaned["discharge_cfs"] < 0).sum()),
                    "zero_days": int((cleaned["discharge_cfs"] == 0).sum()),
                    "calendar_start": str(cleaned["date"].min().date()) if len(cleaned) else None,
                    "calendar_end": str(cleaned["date"].max().date()) if len(cleaned) else None})
    return cleaned, summary


def build_daily_flow_features(dataset, daily_discharge: pd.DataFrame) -> dict:
    """Build ``full[N,T,8]`` without reading any target labels or predictions.

    Dataset keys accessed are exactly ``site_no``, ``months`` and ``x_mask``.
    No statistics are fitted. Positive unit rescaling leaves the descriptors
    unchanged provided both inputs satisfy the fixed raw-unit QC bound.
    """
    sites = np.asarray(dataset["site_no"]).astype(str)
    months = pd.DatetimeIndex(pd.to_datetime(np.asarray(dataset["months"]))).to_period("M")
    masks = np.asarray(dataset["x_mask"])
    if (sites.ndim != 1 or len(sites) == 0 or len(months) == 0
            or len(set(sites)) != len(sites) or months.has_duplicates or months.isna().any()):
        raise ValueError("station/month identifiers must be nonempty and unique")
    if masks.ndim != 3 or masks.shape[:2] != (len(sites), len(months)) or masks.shape[2] < 2:
        raise ValueError("x_mask must align with [station,month,>=2 hydro channels]")
    if not np.isin(masks[..., 1], (0, 1)).all():
        raise ValueError("monthly discharge mask must contain only zero and one")
    observed = masks[..., 1].astype(bool)
    daily, summary = reconcile_daily_discharge(daily_discharge)
    full = np.zeros((len(sites), len(months), len(FEATURE_NAMES)), dtype=np.float32)
    station_map = {site: i for i, site in enumerate(sites)}
    month_map = {month: i for i, month in enumerate(months)}
    daily = daily[daily["site_no"].isin(station_map)].copy()
    daily["month"] = daily["date"].dt.to_period("M")
    daily = daily[daily["month"].isin(month_map)].copy()
    summary["grid_unique_valid_days"] = len(daily)
    summary["grid_stations_with_daily_data"] = int(daily["site_no"].nunique())
    summary["grid_negative_days"] = int((daily["discharge_cfs"] < 0).sum())
    if not daily.empty:
        # int64 before multiplication is essential for this 233,478-cell grid.
        daily["cell"] = (daily["site_no"].map(station_map).astype(np.int64) * len(months)
                         + daily["month"].map(month_map).astype(np.int64))
        daily = daily.sort_values(["cell", "date"])
        daily["absolute"] = daily["discharge_cfs"].abs()
        grouped = daily.groupby("cell", sort=True)
        count = grouped.size()
        absolute_mean = grouped["absolute"].mean()
        quantiles = grouped["discharge_cfs"].quantile([0.1, 0.9], interpolation="linear").unstack()
        cells = count.index.to_numpy(dtype=np.int64)
        station_index, month_index = np.divmod(cells, len(months))
        calendar_days = months.days_in_month.to_numpy()[month_index]
        day_fraction = count.to_numpy() / calendar_days
        width_valid = count.to_numpy() >= np.ceil(COVERAGE_FRACTION * calendar_days)
        width_range = (quantiles[0.9] - quantiles[0.1]).to_numpy()
        denominator = absolute_mean.to_numpy() + width_range
        width = np.divide(width_range, denominator, out=np.zeros(len(cells)), where=denominator > 0)
        full[station_index, month_index, 0] = np.where(width_valid, width, 0)
        full[station_index, month_index, 3] = day_fraction
        full[station_index, month_index, 5] = width_valid

        pair = ((daily["cell"] == daily["cell"].shift())
                & (daily["date"].diff() == pd.Timedelta(days=1)))
        previous = daily["discharge_cfs"].shift()
        pairs = daily.loc[pair, ["cell"]].copy()
        pairs["difference"] = (daily["discharge_cfs"] - previous).loc[pair].to_numpy()
        pairs["absolute_change"] = pairs["difference"].abs()
        pairs["absolute_endpoints"] = (daily["absolute"] + previous.abs()).loc[pair].to_numpy()
        pairs["rising"] = (pairs["difference"] > 0).astype(float)
        pair_groups = pairs.groupby("cell")
        pair_count = pair_groups.size().reindex(cells, fill_value=0).to_numpy()
        pair_sums = pair_groups[["absolute_change", "absolute_endpoints", "rising"]].sum().reindex(
            cells, fill_value=0)
        pair_valid = pair_count >= np.ceil(COVERAGE_FRACTION * (calendar_days - 1))
        flash = np.divide(pair_sums["absolute_change"].to_numpy(),
                          pair_sums["absolute_endpoints"].to_numpy(), out=np.zeros(len(cells)),
                          where=pair_sums["absolute_endpoints"].to_numpy() > 0)
        rising = np.divide(pair_sums["rising"].to_numpy(), pair_count,
                           out=np.zeros(len(cells)), where=pair_count > 0)
        full[station_index, month_index, 1] = np.where(pair_valid, flash, 0)
        full[station_index, month_index, 2] = np.where(pair_valid, rising, 0)
        full[station_index, month_index, 4] = pair_count / (calendar_days - 1)
        full[station_index, month_index, 6] = pair_valid
        full[station_index, month_index, 7] = pair_valid
    summary["daily_months_before_monthly_mask"] = int((full[..., 3] > 0).sum())
    summary["daily_months_excluded_by_monthly_mask"] = int(((full[..., 3] > 0) & ~observed).sum())
    full[~observed] = 0
    summary.update({"grid_shape": list(full.shape), "monthly_discharge_visible_cells": int(observed.sum()),
                    "retained_daily_months": int((full[..., 3] > 0).sum()),
                    "valid_width_cells": int(full[..., 5].sum()),
                    "valid_flashiness_cells": int(full[..., 6].sum()),
                    "valid_rising_cells": int(full[..., 7].sum()),
                    "valid_all_three_cells": int((full[..., 5:].min(axis=-1) == 1).sum())})
    if not np.isfinite(full).all() or np.any(full < 0) or np.any(full > 1):
        raise ValueError("daily feature block must be finite and bounded in [0,1]")
    return {"full": full, "feature_names": list(FEATURE_NAMES),
            "value_feature_indices": list(VALUE_FEATURE_INDICES),
            "availability_feature_indices": list(AVAILABILITY_FEATURE_INDICES),
            "validity_feature_indices": list(VALIDITY_FEATURE_INDICES),
            "policy": POLICY, "quality_summary": summary}


def availability_only(features: np.ndarray) -> np.ndarray:
    """Matched ablation: retain availability columns, zero numeric descriptors."""
    full = np.asarray(features)
    if full.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("daily feature block must have eight channels")
    result = full.copy()
    result[..., list(VALUE_FEATURE_INDICES)] = 0
    return result


def minimum_valid_days(calendar_days: int) -> tuple[int, int]:
    """The frozen integer day/pair thresholds, exposed for inspection."""
    return (math.ceil(COVERAGE_FRACTION * calendar_days),
            math.ceil(COVERAGE_FRACTION * (calendar_days - 1)))
