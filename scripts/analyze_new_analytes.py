#!/usr/bin/env python3
"""Task 5: inventory candidate held-out analytes for few-shot adaptation.

Reads the per-station WQP cache and reports, for each characteristic:
coverage on the ST357 graph, unit mix, monthly co-observation for K-shot,
and whether the feature is currently a model input (must be dropped if so).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Currently in the DOC model's live inputs (gcn.IN_CHANNELS / H2 regime).
# Temperature is dynamic input; pH / EC are not.
MODEL_INPUTS = {
    "temperature": True,
    "ph": False,
    "spec_conductance": False,
    "organic_carbon": True,  # target DOC channel
}


def load_wqp(wqp_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(wqp_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        ch = cols.get("result_characteristic")
        if ch is None:
            continue
        val = cols.get("result_measure")
        un = cols.get("result_measureunit") or cols.get("result_measure_unit")
        fr = cols.get("result_samplefraction")
        dt = cols.get("activity_startdate") or cols.get("date")
        det = cols.get("result_resultdetectioncondition")
        out = pd.DataFrame(
            {
                "station": df[cols["location_identifier"]].astype(str)
                if "location_identifier" in cols
                else (df[cols["site_no"]].astype(str) if "site_no" in cols else f.stem),
                "date": df[dt] if dt else None,
                "characteristic": df[ch],
                "value": df[val] if val else None,
                "unit": df[un] if un else None,
                "fraction": df[fr] if fr else None,
                "detection": df[det] if det else None,
            }
        )
        if "location_identifier" not in cols and "site_no" not in cols:
            out["station"] = f.stem
        # WQP ids look like USGS-03010958; dataset site_no is 03010958
        out["station"] = (
            out["station"].astype(str).str.replace(r"^(USGS-|USGS)", "", regex=True)
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def monthly_mask(rows: pd.DataFrame, station_index: dict[str, int], months: list[str]) -> pd.ndarray:
    """(N, T) bool: station-month has at least one uncensored numeric value."""
    import numpy as np

    n, t = len(station_index), len(months)
    month_to_j = {m[:7]: j for j, m in enumerate(months)}
    mask = np.zeros((n, t), dtype=bool)
    df = rows.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df.value.notna()]
    if "detection" in df:
        df = df[df["detection"].isna() | (df["detection"].astype(str).str.len() == 0)]
    df["ym"] = df["date"].astype(str).str[:7]
    for st, ym in zip(df.station.astype(str), df.ym):
        i = station_index.get(st)
        j = month_to_j.get(ym)
        if i is not None and j is not None:
            mask[i, j] = True
    return mask


def kshot_feasibility(mask) -> dict:
    import numpy as np

    per_month = mask.sum(axis=0)
    active = per_month > 0
    rows = mask.sum(axis=1)
    # leave-one-out same-month support within station set
    loo = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                loo.append(int(mask[:, j].sum()) - 1)
    loo = np.array(loo) if loo else np.array([0])
    return {
        "n_station_months": int(mask.sum()),
        "n_stations_with_data": int((rows > 0).sum()),
        "mean_obs_per_active_month": float(per_month[active].mean()) if active.any() else 0.0,
        "frac_months_ge3": float((per_month[active] >= 3).mean()) if active.any() else 0.0,
        "frac_months_ge5": float((per_month[active] >= 5).mean()) if active.any() else 0.0,
        "loo_p_ge3": float((loo >= 3).mean()),
        "loo_p_ge5": float((loo >= 5).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wqp-dir", default="data/raw/wqp_results")
    ap.add_argument("--nodes", default="data/processed/graph_nodes.csv")
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--out-dir", default="experiments/analyte_inventory")
    args = ap.parse_args()

    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = torch.load(args.dataset, weights_only=False)
    sites = [str(s) for s in ds["site_no"]]
    months = [str(m) for m in ds["months"]]
    station_index = {s: i for i, s in enumerate(sites)}

    raw = load_wqp(Path(args.wqp_dir))
    raw.to_parquet(out / "wqp_long.parquet", index=False)

    # map characteristic -> short name (same as wqp.CHARACTERISTICS)
    name_map = {
        "Organic carbon": "organic_carbon",
        "Temperature, water": "temperature",
        "pH": "ph",
        "Specific conductance": "spec_conductance",
    }
    raw["short"] = raw.characteristic.map(name_map).fillna(raw.characteristic.astype(str).str.lower())

    rows = []
    for short, sub in raw.groupby("short"):
        unit_mix = Counter(sub.unit.astype(str)).most_common(6)
        frac_mix = Counter(sub.fraction.astype(str)).most_common(6)
        on_graph = sub[sub.station.astype(str).isin(station_index)]
        mask = monthly_mask(on_graph, station_index, months)
        feas = kshot_feasibility(mask)
        is_input = MODEL_INPUTS.get(short, None)
        rows.append(
            {
                "analyte": short,
                "n_rows_cache": len(sub),
                "n_stations_cache": sub.station.astype(str).nunique(),
                "n_rows_on_st357": len(on_graph),
                "units": json.dumps(unit_mix),
                "fractions": json.dumps(frac_mix),
                "is_model_input": is_input,
                "drop_input_if_target": bool(is_input),
                **feas,
            }
        )
        print(f"\n{short}: rows_on_st357={len(on_graph)} stations={feas['n_stations_with_data']}")
        print(f"  units={unit_mix}")
        print(f"  kshot: {feas}")
        print(f"  model_input={is_input}")

    inv = pd.DataFrame(rows).sort_values("n_station_months", ascending=False)
    inv.to_csv(out / "analyte_inventory.csv", index=False)

    recommendation = {
        "primary_unseen_candidates": ["ph", "spec_conductance"],
        "secondary_needs_input_drop": ["temperature"],
        "doc_is_current_target": ["organic_carbon"],
        "not_in_local_cache": [
            "dissolved_oxygen",
            "nitrate",
            "phosphorus",
            "chlorophyll_a",
        ],
        "notes": [
            "pH and specific conductance are abundant and not current model inputs.",
            "temperature is richer than DOC but is a live input channel; use only after dropping it from X.",
            "DO/N/P/chlorophyll are not in the local WQP cache; fetch first before designing support/query tasks.",
            "unit mixes for temperature include deg F and organic carbon includes % / g/kg — keep QC filters from quality.py.",
        ],
    }
    (out / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nwrote", out)


if __name__ == "__main__":
    main()
