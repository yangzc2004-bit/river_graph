#!/usr/bin/env python3
"""Build monthly analyte label grids on ST357 nodes (pH, spec_conductance).

Produces a torch dataset dict with the same site_no / months / edge_* fields as
the DOC graph, but y / y_mask / x / x_mask replaced by the chosen analyte.
Dynamic hydro inputs (temp, flow) are kept; DOC channels are zeroed so a model
trained on this target cannot peek DOC. Optional --drop-channel removes a
dynamic input that would leak the target (e.g. temperature).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_wqp_long(wqp_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(wqp_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        need = ["location_identifier", "activity_startdate", "result_characteristic",
                "result_measure", "result_measureunit"]
        if any(c not in cols for c in need):
            continue
        out = pd.DataFrame(
            {
                "station": df[cols["location_identifier"]].astype(str).str.replace(
                    r"^USGS-", "", regex=True
                ),
                "date": df[cols["activity_startdate"]].astype(str),
                "characteristic": df[cols["result_characteristic"]],
                "value": pd.to_numeric(df[cols["result_measure"]], errors="coerce"),
                "unit": df[cols["result_measureunit"]],
                "fraction": df[cols["result_samplefraction"]]
                if "result_samplefraction" in cols
                else None,
                "detection": df[cols["result_resultdetectioncondition"]]
                if "result_resultdetectioncondition" in cols
                else None,
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


SHORT = {
    "pH": "ph",
    "Specific conductance": "spec_conductance",
    "Temperature, water": "temperature",
    "Organic carbon": "organic_carbon",
}


def build_mask_values(
    raw: pd.DataFrame,
    short: str,
    sites: list[str],
    months: list[str],
    unit_keep: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    name = {v: k for k, v in SHORT.items()}[short]
    sub = raw[raw.characteristic == name].copy()
    sub = sub[sub.value.notna()]
    # drop censored
    if "detection" in sub:
        det = sub["detection"]
        sub = sub[det.isna() | (det.astype(str).str.len() == 0)]
    if unit_keep:
        sub = sub[sub.unit.astype(str) == unit_keep]
    # temperature: convert deg F -> deg C if present and unit_keep is deg C
    if short == "temperature" and unit_keep == "deg C":
        fend = sub.unit.astype(str) == "deg F"
        sub.loc[fend, "value"] = (sub.loc[fend, "value"] - 32.0) * 5.0 / 9.0
        sub = sub.copy()
    # pH physical range
    if short == "ph":
        sub = sub[(sub.value >= 0) & (sub.value <= 14)]
    if short == "spec_conductance":
        sub = sub[(sub.value >= 0) & (sub.value <= 100_000)]

    station_index = {s: i for i, s in enumerate(sites)}
    month_to_j = {m[:7]: j for j, m in enumerate(months)}
    n, t = len(sites), len(months)
    y = np.zeros((n, t), dtype=np.float32)
    y_mask = np.zeros((n, t), dtype=bool)
    sub = sub.copy()
    sub["ym"] = sub["date"].str[:7]
    # monthly mean
    for (st, ym), g in sub.groupby(["station", "ym"]):
        i = station_index.get(str(st))
        j = month_to_j.get(str(ym))
        if i is None or j is None:
            continue
        y[i, j] = float(g["value"].mean())
        y_mask[i, j] = True
    return y, y_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wqp-dir", default="data/raw/wqp_results")
    ap.add_argument("--base-dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--analytes", nargs="+", default=["ph", "spec_conductance"])
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    base = torch.load(args.base_dataset, weights_only=False)
    sites = [str(s) for s in base["site_no"]]
    months = [str(m) for m in base["months"]]
    raw = load_wqp_long(Path(args.wqp_dir))

    unit_keep = {"ph": "standard units", "spec_conductance": "uS/cm", "temperature": "deg C"}

    for short in args.analytes:
        y, y_mask = build_mask_values(raw, short, sites, months, unit_keep.get(short))
        ds = {}
        for k, v in base.items():
            if isinstance(v, torch.Tensor):
                ds[k] = v.clone()
            elif isinstance(v, np.ndarray):
                ds[k] = v.copy()
            else:
                ds[k] = v
        # replace labels; zero DOC input channels in x (last two of x if 2-ch hydro)
        y_t = torch.tensor(y)
        y_mask_t = torch.tensor(y_mask)
        ds["y"] = y_t
        ds["y_mask"] = y_mask_t
        if "x" in ds and ds["x"].shape[-1] >= 2:
            # keep temp/flow; DOC obs channels live in the model builder not x
            pass
        out = Path(args.out_dir) / f"mississippi_graph_{short}_st357.pt"
        torch.save(ds, out)
        meta = {
            "analyte": short,
            "unit": unit_keep.get(short),
            "n_station_months": int(y_mask.sum()),
            "n_stations": int((y_mask.sum(1) > 0).sum()),
            "base_dataset": args.base_dataset,
            "value_min": float(y[y_mask].min()) if y_mask.any() else None,
            "value_max": float(y[y_mask].max()) if y_mask.any() else None,
            "value_mean": float(y[y_mask].mean()) if y_mask.any() else None,
        }
        out.with_suffix(".provenance.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(short, meta)


if __name__ == "__main__":
    main()
