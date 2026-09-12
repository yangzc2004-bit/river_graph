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
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics

CURRENT = Path("experiments/predictions")
HISTORICAL = Path("experiments/predictions_historical")
ANALYSIS = Path("experiments/analysis")
MASKS_DIR = Path("experiments/masks")
MANIFEST = Path("experiments/frozen_results/prediction_manifest_20260912.json")
STATION_TYPES = Path("experiments/analysis/training_station_types_20260911.csv")
HEADWATER = Path("experiments/analysis/headwater_recovery.csv")

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
    """USGS station ids: the metadata CSVs store them as int64 (leading zeros
    stripped), the dataset stores zero-padded strings."""
    text = str(value).strip()
    if len(text) < 8 and text.isdigit():
        return text.zfill(8)
    return text


def load_manifest() -> tuple[set[str], dict[str, str]]:
    """Return allowed filenames and their batch label, from the A0 manifest."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    allowed: set[str] = set()
    batch: dict[str, str] = {}
    for entry in payload["predictions"]:
        if entry.get("used_for_phase_a"):
            allowed.add(entry["file"])
            batch[entry["file"]] = "frozen_phase0"
    for entry in payload["historical_recovered"]:
        allowed.add(entry["file"])
        batch[entry["file"]] = "recovered_256d08e"
    return allowed, batch


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


def huc2_lookup(dataset: dict) -> pd.DataFrame:
    """HUC2 region per station: from the audit file where available, else from
    the 2-digit HUC prefix of the station id (USGS ids start with the HUC)."""
    sites = list(dataset["site_no"])
    known: dict[str, int] = {}
    if HEADWATER.exists():
        hw = pd.read_csv(HEADWATER)
        for sid, huc in zip(hw["station_id"], hw["huc2"]):
            known[normalize_site(sid)] = int(huc)
    rows = []
    for site in sites:
        huc = known.get(site)
        source = "audit_file"
        if huc is None:
            source = "station_id_prefix"
            prefix = site[:2]
            huc = int(prefix) if prefix.isdigit() else -1
        rows.append({"station": site, "huc2": huc, "huc2_source": source})
    return pd.DataFrame(rows)


def collect_files(allowed: set[str]) -> list[tuple[Path, str]]:
    out = []
    for directory in (CURRENT, HISTORICAL):
        for path in sorted(directory.glob("*.parquet")):
            if path.name in allowed:
                out.append((path, directory.name))
    return out


def station_rows(
    files: list[tuple[Path, str]],
    station_meta: pd.DataFrame,
    datasets: dict[str, dict],
    masks: dict[str, dict],
) -> pd.DataFrame:
    """Per (model, mask, station) errors plus the station's attributes."""
    rows = []
    for path, _batch in files:
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
    files: list[tuple[Path, str]],
    datasets: dict[str, dict],
    masks: dict[str, dict],
    quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much of the test error comes from the highest-DOC cells?"""
    conc, top_rows = [], []
    for path, _batch in files:
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


def station_bootstrap(
    df: pd.DataFrame, metrics_to_use: tuple[str, ...] = ("mae", "rmse"),
    n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Station-level bootstrap of the cell-weighted aggregate.

    Resamples STATIONS, so the interval reflects which stations happened to be
    held out. It is a sample-layer interval for one fixed historical model and
    cannot stand in for the missing training-seed analysis.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (model, scenario), sub in df.groupby(["model", "scenario"]):
        sub = sub[sub["n_predicted"] > 0]
        if len(sub) < 3:
            continue
        stats = {m: [] for m in metrics_to_use}
        mae = sub["mae"].to_numpy()
        rmse = sub["rmse"].to_numpy()
        w = sub["n_predicted"].to_numpy().astype(float)
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub), len(sub))
            ww = w[idx]
            if ww.sum() == 0:
                continue
            stats["mae"].append(float((mae[idx] * ww).sum() / ww.sum()))
            stats["rmse"].append(float(np.sqrt((rmse[idx] ** 2 * ww).sum()
                                               / ww.sum())))
        row = {"model": model, "scenario": scenario,
               "n_stations_resampled": len(sub),
               "bootstrap_unit": "station",
               "n_resamples": n_boot}
        for metric, values in stats.items():
            if not values:
                continue
            arr = np.array(values)
            row[f"{metric}_point"] = float(np.nansum(
                (mae if metric == "mae" else rmse ** 2) * w) / w.sum()
                if metric == "mae" else np.sqrt(
                    np.nansum(rmse ** 2 * w) / w.sum()))
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

    allowed, _batch = load_manifest()
    files = collect_files(allowed)
    print(f"prediction files allowed by the manifest: {len(allowed)}")
    print(f"files found on disk                     : {len(files)}")

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
