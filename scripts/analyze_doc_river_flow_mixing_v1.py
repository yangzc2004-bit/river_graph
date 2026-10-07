"""Compare observed DOC with the discharge-weighted upstream mixture."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from river_graph.analysis.river_flow_mixing import (
    attach_daily_flows,
    summarize_mixing_window,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_flow_mixing_v1")
PARENT = Path("experiments/phase4_transfer/doc_river_event_observations_v1")
KEY = ["source_a", "source_b", "receiver", "year"]


def gauge_table(manifest: dict, stations: pd.DataFrame) -> pd.DataFrame:
    project = Transformer.from_crs(4326, 3006, always_xy=True).transform
    rows = []
    for item in manifest["objects"]:
        if item["kind"] != "flow":
            continue
        preamble = Path(item["path"]).read_text().split("TIMESTAMP,")[0]
        area = re.search(r"Catchment area\s+([\d.]+)\s*km2", preamble)
        lat = re.search(r"LATITUDE:\s*([\d.]+)", preamble)
        lon = re.search(r"LONGITUDE:\s*([\d.]+)", preamble)
        methods = [line for line in preamble.splitlines() if line.lower().startswith("q -")]
        if not area or not lat or not lon or len(methods) != 1:
            raise ValueError(f"Unrecognized gauge metadata: {item['filename']}")
        site = f"C{item['site']}"
        laboratory = stations[stations.site.eq(site)].iloc[0]
        gx, gy = project(float(lon[1]), float(lat[1]))
        cx, cy = project(laboratory.longitude, laboratory.latitude)
        rows.append({"site": site, "catchment_area_km2": float(area[1]),
                     "gauge_latitude": float(lat[1]), "gauge_longitude": float(lon[1]),
                     "gauge_sample_separation_m": np.hypot(gx-cx, gy-cy),
                     "discharge_method": methods[0], "object_url": item["object_url"]})
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def build():
    analysis = PARENT / "analysis"
    windows = pd.read_csv(analysis / "campaign_variation.csv")
    connections = pd.read_csv(analysis / "monitored_confluences.csv")
    stations = pd.read_csv(analysis / "stations.csv")
    flow = pd.read_parquet(analysis / "daily_discharge.parquet")
    campaigns = pd.read_parquet(analysis / "matched_doc_campaigns.parquet")
    if windows.duplicated(KEY).any():
        raise ValueError("Parent comparison windows are not unique")
    selected = campaigns.merge(windows[KEY], on=KEY, how="inner", validate="many_to_one")
    selected = attach_daily_flows(selected, flow)
    gauges = gauge_table(json.loads((PARENT / "retrieval_manifest.json").read_text()), stations)
    area = gauges.set_index("site").catchment_area_km2
    connections["upstream_area_sum_km2"] = connections.source_a.map(area) + connections.source_b.map(area)
    connections["receiver_area_km2"] = connections.receiver.map(area)
    connections["upstream_area_share"] = connections.upstream_area_sum_km2 / connections.receiver_area_km2
    availability, summaries, detailed = [], [], []
    for row in windows.to_dict("records"):
        subset = selected
        for key in KEY:
            subset = subset[subset[key].eq(row[key])]
        configuration = connections
        for key in KEY[:-1]:
            configuration = configuration[configuration[key].eq(row[key])]
        if len(configuration) != 1:
            raise ValueError("Original window does not have one mapped configuration")
        context = configuration.iloc[0].to_dict()
        key = {k: row[k] for k in KEY}
        key["peak_date"] = row["peak_date"]
        ready = subset.complete_flow_campaign.sum() >= 5
        availability.append({**key, "n_original_campaigns": len(subset),
                             "n_upstream_flow_campaigns": int(subset.valid_upstream_flow.sum()),
                             "n_complete_flow_campaigns": int(subset.complete_flow_campaign.sum()),
                             "n_missing_q_a": int(subset.q_a_m3s.isna().sum()),
                             "n_missing_q_b": int(subset.q_b_m3s.isna().sum()),
                             "n_missing_q_receiver": int(subset.q_receiver_m3s.isna().sum()),
                             "variation_comparison_available": bool(ready)})
        if ready:
            metrics, complete = summarize_mixing_window(subset)
            summaries.append({**context, **key, **metrics})
            detailed.append(complete)
    availability = pd.DataFrame(availability)
    comparison = pd.DataFrame(summaries)
    if comparison.empty:
        raise ValueError("No window has enough matched daily flow")
    by_receiver = []
    for site, sub in comparison.groupby("receiver"):
        by_receiver.append({"receiver": site, "n_windows": len(sub),
                            "n_receiver_cv_below_upstream_average": int(sub.receiver_minus_mean_upstream_cv.lt(0).sum()),
                            "n_dynamic_mix_cv_below_upstream_average": int(sub.dynamic_mix_minus_mean_upstream_cv.lt(0).sum()),
                            "n_receiver_cv_below_dynamic_mix": int(sub.receiver_minus_dynamic_mix_cv.lt(0).sum()),
                            "n_receiver_sd_below_dynamic_mix": int(sub.receiver_minus_dynamic_mix_sd.lt(0).sum()),
                            "window_median_flow_share": float(sub.flow_share_median.median()),
                            "mean_receiver_minus_mix_cv": float(sub.receiver_minus_dynamic_mix_cv.mean()),
                            "mean_receiver_minus_mix_sd": float(sub.receiver_minus_dynamic_mix_sd.mean())})
    tables = {"gauge_metadata": gauges, "configuration_flow_coverage": connections,
              "window_availability": availability, "mixing_comparison": comparison,
              "receiver_summary": pd.DataFrame(by_receiver)}
    products = {"all_eligible_campaigns": selected, "complete_mixing_campaigns": pd.concat(detailed, ignore_index=True)}
    summary = {
        "n_original_windows": len(windows), "n_original_matched_campaigns": len(selected),
        "n_windows_with_matched_flow": len(comparison),
        "n_complete_flow_campaigns": int(availability.n_complete_flow_campaigns.sum()),
        "n_campaigns_in_variation_comparison": len(products["complete_mixing_campaigns"]),
        "n_receiver_cv_below_upstream_average": int(comparison.receiver_minus_mean_upstream_cv.lt(0).sum()),
        "n_dynamic_mix_cv_below_upstream_average": int(comparison.dynamic_mix_minus_mean_upstream_cv.lt(0).sum()),
        "n_receiver_cv_below_dynamic_mix": int(comparison.receiver_minus_dynamic_mix_cv.lt(0).sum()),
        "n_receiver_sd_below_dynamic_mix": int(comparison.receiver_minus_dynamic_mix_sd.lt(0).sum()),
        "n_receiver_configurations": int(comparison.receiver.nunique()),
        "n_research_catchments": 1, "flow_resolution": "daily source-calendar means",
        "doc_resolution": "observed laboratory campaigns; no interpolation",
        "comparison_scope": "partial observed branches, not complete downstream mass balance",
    }
    return tables, products, summary


def main():
    tables, products, summary = build()
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in products.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [PARENT / "analysis" / name for name in (
        "campaign_variation.csv", "monitored_confluences.csv", "stations.csv",
        "daily_discharge.parquet", "matched_doc_campaigns.parquet")]
    sources += [Path(item["path"]) for item in json.loads((PARENT / "retrieval_manifest.json").read_text())["objects"]
                if item["kind"] == "flow"]
    sources += [PARENT / "retrieval_manifest.json", ROOT / "study_plan.md",
                Path("scripts/analyze_doc_river_flow_mixing_v1.py"),
                Path("src/river_graph/analysis/river_flow_mixing.py")]
    for path in sources:
        if str(path).startswith(("scripts/", "src/")):
            target = ROOT / "code_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    outputs = [output / f"{k}.csv" for k in tables] + [output / f"{k}.parquet" for k in products] + [output / "summary.json"]
    receipt = {"source_hashes": {str(p): sha256_file(p) for p in sources},
               "output_hashes": {str(p): sha256_file(p) for p in outputs}}
    (ROOT / "analysis_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
