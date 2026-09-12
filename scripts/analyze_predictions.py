"""A3: bounded diagnostics on the STORED historical predictions. No training.

Only the prediction files allowed by the A0 manifest are read. Everything here
is descriptive: this is one fixed historical model per (model, mask), so the
numbers are properties of that stored run, not uncertainty estimates over
training seeds.

Caveats are emitted with the outputs, not only in the report:

* ``upstream_degree == 0`` is labelled "no upstream monitoring neighbour in
  the graph". It is NOT the same as a true hydrological source: that needs a
  river-network attribute check, which this phase does not perform.
* Kriging does not predict every test cell (0% on e2a_strict, ~95-99%
  elsewhere), so error magnitudes are only comparable at equal coverage; n and
  coverage are reported next to every error number.
* The model was trained on mixed station types, so ST-vs-other splits describe
  sensitivity within a mixed-training model; they are not an ST-only
  validation.
* The five-seed per-cell predictions do not exist, so comparisons that would
  need them are reported as not-run rather than approximated.

Usage:
    python scripts/analyze_predictions.py
    python scripts/analyze_predictions.py --date 20260912
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics
from river_graph.experiments.prediction_sources import (
    PredictionSource,
    describe_report,
    load_sources,
)

CURRENT = Path("experiments/predictions")
HISTORICAL = Path("experiments/predictions_historical")
ANALYSIS = Path("experiments/analysis")
MASKS_DIR = Path("experiments/masks")
MANIFEST = Path("experiments/frozen_results/prediction_manifest_20260912.json")
STATION_TYPES = Path("experiments/analysis/training_station_types_20260911.csv")
GRAPH_NODES = Path("data/processed/graph_nodes.csv")

MODEL_DATASETS = {
    "B0": "data/processed/mississippi_graph_v02.pt",
    "B1": "data/processed/mississippi_graph_v02.pt",
    "B2": "data/processed/mississippi_graph_v02.pt",
    "B3": "data/processed/mississippi_graph_v02.pt",
    "G0": "data/processed/mississippi_graph_v02.pt",
    "H1": "data/processed/mississippi_graph_v02.pt",
    "H2": "data/processed/mississippi_graph_v03.pt",
    "H2E": "data/processed/mississippi_graph_v04.pt",
    "H2X": "data/processed/mississippi_graph_v04.pt",
}

SCENARIO_OF = {
    "e1_r20_seed42": "E1", "e1_r20_seed43": "E1", "e1_r20_seed44": "E1",
    "e1_r40_seed42": "E1_r40", "e1_r40_seed43": "E1_r40", "e1_r40_seed44": "E1_r40",
    "e1_r60_seed42": "E1_r60", "e1_r60_seed43": "E1_r60", "e1_r60_seed44": "E1_r60",
    "e2a_strict": "E2a", "e2b_partial": "E2b",
    "e3_spatial_seed42": "E3", "e3_spatial_seed43": "E3",
    "e3_spatial_seed44": "E3",
}

# Comparison against the five-seed refresh; the predictions for these do not
# exist, so the columns are recorded as unavailable rather than approximated.
FIVE_SEED_MODELS = {"H1": "H1_directed_river", "H2": "H2_transport_river",
                    "H2X": "H2X_transport_enc_river"}
FIVE_SEED_MASKS = {
    "e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44",
    "e2a_strict", "e2b_partial",
    "e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44",
}

MIN_CELLS_FOR_R2 = 10


def dataset_for(model: str) -> str:
    for prefix, path in MODEL_DATASETS.items():
        if model.startswith(prefix):
            return path
    return "data/processed/mississippi_graph_v02.pt"


def normalize_site(value: object) -> str:
    """USGS station ids: metadata CSVs may store them as numbers (leading zeros
    stripped), the dataset stores zero-padded strings. Always return a string so
    ids survive a CSV round trip."""
    text = str(value).strip()
    text = text.removesuffix(".0")
    if len(text) < 8 and text.isdigit():
        return text.zfill(8)
    return text


def graph_roles(dataset: dict) -> pd.DataFrame:
    """Upstream / downstream degree per station from the edge list.

    Only says something about the *monitoring graph*: a station with no
    upstream neighbour in this graph is a headwater of the observed network,
    which is weaker than a hydrological source claim.
    """
    edge_index = dataset["edge_index"]
    sites = list(dataset["site_no"])
    n = len(sites)
    up = np.zeros(n, dtype=int)
    down = np.zeros(n, dtype=int)
    src, dst = edge_index[0], edge_index[1]
    for a, b in zip(src, dst):
        up[int(b)] += 1
        down[int(a)] += 1
    return pd.DataFrame({
        "station": sites,
        "upstream_degree": up,
        "downstream_degree": down,
        "graph_role": np.where(up == 0, "no_upstream_neighbour",
                               "has_upstream_neighbour"),
    })


def huc2_from_code(code: object) -> int | None:
    """HUC2 region from one ``huc_cd`` value, or None if it cannot be read.

    ``huc_cd`` is an INTEGER in ``graph_nodes.csv``, so a leading zero is gone
    and the field length is inconsistent: ``05010001`` -> ``5010001`` (7),
    ``08070100`` -> ``8070100`` (7), an HUC12 keeps all 12 digits, and a
    zero-stripped HUC12 arrives with 11. The level therefore has to be inferred
    from the length and the value re-padded before the first two digits (the
    HUC2 region) can be read:

    * 7-8 digits   -> HUC8,  pad to 8  (``5010001`` -> ``05010001`` -> 05)
    * 9-10 digits  -> HUC10, pad to 10 (``60300020403`` -> ``0603000204`` -> 06)
    * 11-12 digits -> HUC12, pad to 12 (``101202021305`` -> 10)

    Station-number prefixes must NOT be used instead: a USGS station id is not a
    HUC code. Station 06438000 (Belle Fourche River, SD) belongs to HUC2 10 even
    though its id begins with "06".
    """
    text = str(code).strip()
    if text.lower() in ("", "nan", "none"):
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 7:
        return None
    if len(digits) <= 8:
        padded = digits.zfill(8)
    elif len(digits) <= 10:
        padded = digits.zfill(10)
    else:
        padded = digits.zfill(12)
    return int(padded[:2])


def huc2_lookup(dataset: dict) -> pd.DataFrame:
    """HUC2 region per station, from the authoritative ``huc_cd`` column.

    Stations without a readable code are marked unknown rather than guessed.
    """
    sites = list(dataset["site_no"])
    known: dict[str, int] = {}
    if GRAPH_NODES.exists():
        nodes = pd.read_csv(GRAPH_NODES, dtype={"site_no": str})
        for sid, code in zip(nodes["site_no"], nodes["huc_cd"]):
            value = huc2_from_code(code)
            if value is not None:
                known[str(sid).strip()] = value
    rows = []
    for site in sites:
        key = str(site).strip()
        huc = known.get(key)
        rows.append({
            "station": key,
            "huc2": huc if huc is not None else -1,
            "huc2_source": "graph_nodes.huc_cd" if huc is not None else "unknown",
        })
    return pd.DataFrame(rows)


def station_observations(dataset: dict) -> pd.DataFrame:
    """True per-station label statistics, straight from the dataset.

    Aggregating the per-(model, mask) ``mean_true`` column would describe the
    test splits rather than the observations, so means and maxima for a station
    are computed here from its actual observed monthly values.
    """
    y = dataset["y"].numpy()
    y_mask = dataset["y_mask"].numpy()
    months = list(dataset["months"])
    rows = []
    for i, site in enumerate(dataset["site_no"]):
        observed = y_mask[i]
        values = y[i][observed]
        if len(values) == 0:
            rows.append({"station": normalize_site(site), "n_observed_months": 0,
                         "obs_mean": np.nan, "obs_max": np.nan,
                         "obs_min": np.nan, "obs_median": np.nan,
                         "n_obs_ge_100": 0, "n_obs_ge_400": 0,
                         "first_month": None, "last_month": None})
            continue
        idx = np.flatnonzero(observed)
        rows.append({
            "station": normalize_site(site),
            "n_observed_months": len(values),
            "obs_mean": float(np.mean(values)),
            "obs_max": float(np.max(values)),
            "obs_min": float(np.min(values)),
            "obs_median": float(np.median(values)),
            "n_obs_ge_100": int((values >= 100).sum()),
            "n_obs_ge_400": int((values >= 400).sum()),
            "first_month": months[int(idx[0])],
            "last_month": months[int(idx[-1])],
        })
    return pd.DataFrame(rows)


def station_rows(
    files: list[PredictionSource],
    station_meta: pd.DataFrame,
    datasets: dict[str, dict],
    masks: dict[str, dict],
) -> pd.DataFrame:
    """Per (model, mask, station) errors plus the station's attributes.

    Station ids are emitted as zero-padded strings: USGS ids carry leading
    zeros, and writing them as integers would silently turn 06438000 into
    6438000 in every output file.
    """
    rows = []
    for source in files:
        path, _batch = source.path, source.batch
        df = pd.read_parquet(path)
        model = str(df["model"].iloc[0])
        mask_name = str(df["mask"].iloc[0])
        dpath = dataset_for(model)
        if dpath not in datasets:
            datasets[dpath] = load_dataset(dpath)
        if mask_name not in masks:
            masks[mask_name] = load_mask(mask_name, MASKS_DIR)
        ds, split = datasets[dpath], masks[mask_name]
        y = ds["y"].numpy()
        _n, t = y.shape
        sites = list(ds["site_no"])
        months = list(ds["months"])
        test_idx = np.asarray(split["test"], dtype=np.int64)
        want = [(sites[i // t], months[i % t]) for i in test_idx]
        lut = dict(zip(zip(df["station"].tolist(), df["month"].tolist()),
                       zip(df["y_true"].tolist(), df["y_pred"].tolist())))
        yt = np.array([lut.get(k, (np.nan, np.nan))[0] for k in want], float)
        yp = np.array([lut.get(k, (np.nan, np.nan))[1] for k in want], float)
        station_of_cell = np.array([k[0] for k in want])

        for station in pd.unique(station_of_cell):
            sel = station_of_cell == station
            a, b = yt[sel], yp[sel]
            ok = np.isfinite(a) & np.isfinite(b)
            m = metrics(a, b)
            rows.append({
                "model": model, "mask": mask_name,
                "scenario": SCENARIO_OF.get(mask_name, "other"),
                "station": station,
                "n_test_cells": int(sel.sum()),
                "n_predicted": int(ok.sum()),
                "coverage": float(ok.mean()) if sel.sum() else np.nan,
                "mae": m["mae"], "rmse": m["rmse"],
                "r2": m["r2"] if ok.sum() >= MIN_CELLS_FOR_R2 else np.nan,
                "pbias": m["pbias"],
                "mean_true": float(np.nanmean(a)) if ok.sum() else np.nan,
                "mean_pred": float(np.nanmean(b)) if ok.sum() else np.nan,
            })
    out = pd.DataFrame(rows)
    out["station"] = out["station"].map(normalize_site)
    return out.merge(station_meta, on="station", how="left")


def aggregate_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Station-equal-weighted and cell-weighted summaries side by side."""
    rows = []
    for key, sub in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        cells = sub["n_predicted"].sum()
        weighted_mae = (np.nansum(sub["mae"] * sub["n_predicted"]) / cells
                        if cells else np.nan)
        weighted_rmse = np.sqrt(
            np.nansum(sub["rmse"] ** 2 * sub["n_predicted"]) / cells
        ) if cells else np.nan
        row = dict(zip(keys, key))
        row.update({
            "n_stations": len(sub),
            "n_cells": int(cells),
            "mae_station_equal": float(np.nanmean(sub["mae"])),
            "mae_cell_weighted": float(weighted_mae),
            "rmse_station_equal": float(np.nanmean(sub["rmse"])),
            "rmse_cell_weighted": float(weighted_rmse),
            "r2_station_equal": float(np.nanmean(sub["r2"]))
            if sub["r2"].notna().any() else float("nan"),
            "median_station_mae": float(np.nanmedian(sub["mae"])),
            "stations_with_r2": int(sub["r2"].notna().sum()),
            "min_coverage": float(np.nanmin(sub["coverage"])),
            "mean_coverage": float(np.nanmean(sub["coverage"])),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_cells(
    files: list[PredictionSource],
    datasets: dict[str, dict],
    masks: dict[str, dict],
    quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much of the test error comes from the highest-DOC cells?"""
    conc, top_rows = [], []
    for source in files:
        path = source.path
        df = pd.read_parquet(path)
        model = str(df["model"].iloc[0])
        mask_name = str(df["mask"].iloc[0])
        dpath = dataset_for(model)
        if dpath not in datasets:
            datasets[dpath] = load_dataset(dpath)
        ds = datasets[dpath]
        if mask_name not in masks:
            masks[mask_name] = load_mask(mask_name, MASKS_DIR)
        split = masks[mask_name]
        y = ds["y"].numpy()
        _n, t = y.shape
        sites, months = list(ds["site_no"]), list(ds["months"])
        test_idx = np.asarray(split["test"], dtype=np.int64)
        want = [(sites[i // t], months[i % t]) for i in test_idx]
        lut = dict(zip(zip(df["station"].tolist(), df["month"].tolist()),
                       zip(df["y_true"].tolist(), df["y_pred"].tolist())))
        yt = np.array([lut.get(k, (np.nan, np.nan))[0] for k in want], float)
        yp = np.array([lut.get(k, (np.nan, np.nan))[1] for k in want], float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        if ok.sum() == 0:
            continue
        yt_ok, yp_ok = yt[ok], yp[ok]
        want_ok = [w for w, keep in zip(want, ok) if keep]
        thr = float(np.quantile(yt_ok, quantile))
        err2 = (yt_ok - yp_ok) ** 2
        top = yt_ok >= thr
        conc.append({
            "model": model, "mask": mask_name,
            "scenario": SCENARIO_OF.get(mask_name, "other"),
            "coverage": float(ok.mean()),
            "n_cells": int(ok.sum()),
            "doc_q99_threshold": thr,
            "n_top_cells": int(top.sum()),
            "top_share_of_cells": float(top.mean()),
            "top_share_of_sse": float(err2[top].sum() / err2.sum()),
            "mae_all": float(np.mean(np.abs(yt_ok - yp_ok))),
            "mae_excl_top": float(np.mean(np.abs(yt_ok[~top] - yp_ok[~top]))),
        })
        if top.any():
            order = np.argsort(-err2 * top)[:5]
            for j in order:
                if not top[j]:
                    continue
                top_rows.append({
                    "model": model, "mask": mask_name,
                    "station": want_ok[j][0], "month": want_ok[j][1],
                    "y_true": float(yt_ok[j]), "y_pred": float(yp_ok[j]),
                    "abs_error": float(abs(yt_ok[j] - yp_ok[j])),
                })
    return pd.DataFrame(conc), pd.DataFrame(top_rows)


def _pooled_cell_weighted(n_cells: np.ndarray, sae: np.ndarray,
                          sse: np.ndarray,
                          idx: np.ndarray) -> tuple[float, float, float]:
    """Cell-weighted MAE / RMSE over the selected station clusters."""
    total = n_cells[idx].sum()
    if total == 0:
        return float("nan"), float("nan"), 0.0
    return (float(sae[idx].sum() / total),
            float(np.sqrt(sse[idx].sum() / total)), float(total))


def station_bootstrap(
    df: pd.DataFrame, metrics_to_use: tuple[str, ...] = ("mae", "rmse"),
    n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """STATION-clustered bootstrap of the cell-weighted aggregate.

    A scenario spans several masks, so one station contributes several rows
    (H2X/E3 has 243 rows but only 201 distinct stations). Resampling rows
    independently would resample station-and-split combinations and understate
    the interval, so a station is drawn once per replicate and ALL of its rows
    come with it. The statistic is recomputed from the pooled cells, which also
    keeps the point estimate internally consistent.

    It remains a sample-layer interval for one fixed historical model per mask
    and cannot stand in for the missing training-seed analysis.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (model, scenario), sub in df.groupby(["model", "scenario"]):
        sub = sub[sub["n_predicted"] > 0]
        if sub.empty:
            continue
        # one entry per station, so the resampling unit is the station
        clusters = []
        for station, group in sub.groupby("station"):
            w = group["n_predicted"].to_numpy(dtype=float)
            mae = group["mae"].to_numpy(dtype=float)
            rmse = group["rmse"].to_numpy(dtype=float)
            clusters.append({
                "station": station,
                "rows": len(group),
                "n_cells": float(w.sum()),
                "sse": float(np.nansum(rmse ** 2 * w)),
                "sae": float(np.nansum(mae * w)),
            })
        if len(clusters) < 3:
            continue
        cluster_df = pd.DataFrame(clusters)
        n_cells = cluster_df["n_cells"].to_numpy()
        sae = cluster_df["sae"].to_numpy()
        sse = cluster_df["sse"].to_numpy()
        n_stations = len(cluster_df)

        point_mae, point_rmse, _ = _pooled_cell_weighted(
            n_cells, sae, sse, np.arange(n_stations))
        stats = {m: [] for m in metrics_to_use}
        for _ in range(n_boot):
            idx = rng.integers(0, n_stations, n_stations)
            mae_v, rmse_v, c = _pooled_cell_weighted(n_cells, sae, sse, idx)
            if np.isnan(mae_v) or c == 0:
                continue
            if "mae" in stats:
                stats["mae"].append(mae_v)
            if "rmse" in stats:
                stats["rmse"].append(rmse_v)

        row = {"model": model, "scenario": scenario,
               "n_stations_resampled": n_stations,
               "n_rows_used": len(sub),
               "rows_per_station_max": int(cluster_df["rows"].max()),
               "bootstrap_unit": "station (all rows of a drawn station)",
               "n_resamples": n_boot}
        for metric, values in stats.items():
            if not values:
                continue
            arr = np.array(values)
            row[f"{metric}_point"] = point_mae if metric == "mae" else point_rmse
            row[f"{metric}_lo95"] = float(np.quantile(arr, 0.025))
            row[f"{metric}_hi95"] = float(np.quantile(arr, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def station_type_split(station_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive ST-vs-other comparison.

    The models were trained on mixed station types, so this describes
    sensitivity inside a mixed-training model. It is not an ST-only validation.
    """
    if "site_type_group" not in station_df.columns:
        raise RuntimeError("station attributes are missing from station_df")
    return aggregate_by(station_df, ["model", "scenario", "site_type_group"])


def model_summary(station_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by(station_df, ["model", "scenario", "mask"])


def five_seed_gap() -> pd.DataFrame:
    """Record explicitly which comparisons cannot be made."""
    rows = []
    for model, pred_stem in FIVE_SEED_MODELS.items():
        histories = sorted(p.stem.split("__", 1)[1]
                           for p in CURRENT.glob(f"{model}_*__*.parquet"))
        recovered = sorted(p.stem.split("__", 1)[1]
                           for p in HISTORICAL.glob(f"{model}_*__*.parquet"))
        rows.append({
            "model": model,
            "five_seed_prediction_files": 0,
            "historical_prediction_files": len(histories) + len(recovered),
            "multiseed_metrics_available": 8,
            "comparison_possible": False,
            "reason": ("per-cell predictions for training seeds 0-4 do not "
                       "exist; only metric-level multi-seed values are "
                       "available"),
            "prediction_stem_for_metrics": pred_stem,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc)
                    .date().strftime("%Y%m%d"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    date = args.date

    files, source_report = load_sources()
    print(describe_report(source_report))
    if source_report["missing"] or source_report["hash_changed"]:
        raise SystemExit(
            "the on-disk predictions no longer match the frozen manifest; "
            "rerun scripts/build_prediction_manifest.py after checking why")
    print()

    v02 = load_dataset("data/processed/mississippi_graph_v02.pt")
    roles = graph_roles(v02)
    huc = huc2_lookup(v02)
    station_meta = roles.merge(huc, on="station", how="left")

    types = pd.read_csv(STATION_TYPES)
    types["station"] = types["site_no"].map(normalize_site)
    types["site_type_group"] = np.where(types["site_tp_cd"] == "ST",
                                        "ST", "other")
    station_meta = station_meta.merge(
        types[["station", "site_tp_cd", "site_type_group", "DOC_monthly_labels"]],
        on="station", how="left")
    print(f"stations with a site type             : "
          f"{int(station_meta['site_tp_cd'].notna().sum())} / {len(station_meta)}")
    print("site type groups:", station_meta["site_type_group"]
          .value_counts(dropna=False).to_dict())
    unknown_huc = int((station_meta["huc2"] == -1).sum())
    print(f"stations with unknown HUC2            : {unknown_huc}")
    print("HUC2 values (authoritative):",
          sorted(h for h in station_meta["huc2"].unique() if h != -1))

    observations = station_observations(v02)
    station_meta = station_meta.merge(observations, on="station", how="left")

    datasets: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    station_df = station_rows(files, station_meta, datasets, masks)
    print(f"per-station rows                      : {len(station_df)}")

    per_station = station_df.sort_values(["model", "scenario", "mask", "station"])
    by_huc = aggregate_by(station_df, ["model", "scenario", "huc2"])
    by_role = aggregate_by(station_df,
                           ["model", "scenario", "graph_role"])
    by_degree = aggregate_by(station_df,
                             ["model", "scenario", "upstream_degree"])
    summary = model_summary(station_df)
    extremes, top_cells = extreme_cells(files, datasets, masks)
    boot = station_bootstrap(station_df, n_boot=args.bootstrap)
    type_split = station_type_split(station_df)
    gap = five_seed_gap()

    target = station_df[station_df["station"] == "06438000"]
    if target.empty:
        target = station_df[station_df["station"].str.zfill(8) == "06438000"]

    station_obs = station_meta[["station", "huc2", "huc2_source", "site_tp_cd",
                                "site_type_group", "n_observed_months",
                                "obs_mean", "obs_median", "obs_max", "obs_min",
                                "n_obs_ge_100", "n_obs_ge_400",
                                "first_month", "last_month"]]
    station_obs = station_obs.copy()
    station_obs["station"] = station_obs["station"].map(normalize_site)

    outputs = {
        "predictions_per_station": per_station,
        "predictions_by_huc2": by_huc,
        "predictions_by_graph_role": by_role,
        "predictions_by_upstream_degree": by_degree,
        "predictions_model_summary": summary,
        "predictions_extreme_concentration": extremes,
        "predictions_extreme_cells": top_cells,
        "predictions_station_bootstrap": boot,
        "predictions_station_type_split": type_split,
        "predictions_five_seed_gap": gap,
        "predictions_station_06438000": target,
        "predictions_station_observations": station_obs,
    }
    for name, frame in outputs.items():
        path = ANALYSIS / f"{name}_{date}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path}  ({len(frame)} rows)")

    # quick view for the console
    print()
    print("cell-weighted MAE by scenario (mean over the stored masks):")
    view = (summary.groupby(["model", "scenario"])["mae_cell_weighted"]
            .mean().unstack().round(3))
    print(view.to_string())


if __name__ == "__main__":
    main()
