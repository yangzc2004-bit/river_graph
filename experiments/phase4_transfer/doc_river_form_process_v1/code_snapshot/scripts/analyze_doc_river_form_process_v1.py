"""Relate real river form to observed DOC integration and along-channel signals."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_process import (
    FOCAL_TERMS,
    MIX_CONTROLS,
    PATH_CONTROLS,
    aggregate_receivers,
    anomaly_diagnostics,
    cluster_mean,
    conditional_association,
)
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_form_process_v1")
OLD = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis")
MORPH = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
OUTCOMES = ("log_anomaly_sd_ratio", "signal_rho", "mean_log_departure")


def load_mixing():
    cases = pd.read_csv(OLD/"mixing_cases.csv", dtype={"target": str, "huc4": str})
    inventory = pd.read_csv(OLD/"confluence_inventory.csv", dtype={"target": str, "source_a": str, "source_b": str})
    records = pd.read_parquet(OLD/"mixing_records.parquet")
    rows = []
    for (pair, weighting), f in records[records.population.eq("monthly")].groupby(["pair_id", "weighting"]):
        summary = cases[cases.pair_id.eq(pair) & cases.weighting.eq(weighting) & cases.population.eq("monthly")]
        assert len(summary) == 1
        r = summary.iloc[0].to_dict()
        context = inventory[inventory.pair_id.eq(pair)].iloc[0]
        assert context.eligible_monthly and context.branch_status == "independent"
        assert len(f) == r["n_records"]
        r.update(anomaly_diagnostics(f.sort_values("date"), mixing=True))
        r.update(source_a=context.source_a, source_b=context.source_b)
        rows.append(r)
    return pd.DataFrame(rows)


def load_paths():
    edges = pd.read_csv(OLD/"pathway_edges.csv", dtype={"source": str, "target": str, "huc4": str})
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    visible = permitted_doc(data, np.load(OLD/"source_cells.npy"))
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    months = pd.DatetimeIndex(data["months"])
    x, mask = np.asarray(data["x"]), np.asarray(data["x_mask"], bool)
    qi, ti = (list(data["feature_channels"]).index(c) for c in ("discharge", "temperature"))
    rows = []
    for e in edges.itertuples():
        a, b = index[e.source], index[e.target]
        common = np.flatnonzero(np.isfinite(visible[a]) & np.isfinite(visible[b]))
        assert len(common) == e.n_common_months
        f = pd.DataFrame({"date": months[common], "doc_a": visible[a, common], "doc_target": visible[b, common]})
        row = e._asdict()
        row.update(anomaly_diagnostics(f), population="path_monthly")
        rows.append(row)
        good = (mask[a, common, qi] & mask[b, common, qi] & mask[a, common, ti] & mask[b, common, ti]
                & (x[a, common, qi] > 0) & (x[b, common, qi] > 0))
        assert good.sum() == e.n_hydro_months
        if good.sum() >= 24:
            chosen = common[good]
            hydro = [np.column_stack([np.log1p(x[i, chosen, qi]), x[i, chosen, ti]]) for i in (a, b)]
            c = e._asdict()
            c.update(anomaly_diagnostics(f[good]), population="path_hydro_common_calendar")
            c["log_flow_ratio"] = float(np.mean(np.log(x[b, chosen, qi]/x[a, chosen, qi])))
            c["temperature_change"] = float(np.mean(x[b, chosen, ti]-x[a, chosen, ti]))
            rows.append(c)
            r = e._asdict()
            r.update(anomaly_diagnostics(f[good], hydro=hydro), population="path_hydro_adjusted")
            r["log_flow_ratio"] = float(np.mean(np.log(x[b, chosen, qi]/x[a, chosen, qi])))
            r["temperature_change"] = float(np.mean(x[b, chosen, ti]-x[a, chosen, ti]))
            rows.append(r)
    return pd.DataFrame(rows)


def build_panels():
    morphology = pd.read_csv(MORPH, dtype={"station": str, "huc4": str})
    keep = ["station", "cluster", *FOCAL_TERMS, "log_polygon_area", "wetland", "forest",
            "agriculture", "urban", "precipitation", "climate_temperature", "latitude", "longitude"]
    shape = morphology[keep].copy()
    cases, paths = load_mixing(), load_paths()
    panels, ledger = {}, []
    for kind in ("area", "flow"):
        f = cases[cases.weighting.eq(kind)]
        metrics = (*OUTCOMES, "log_mixture_sd_ratio", "log_downstream_mixture_sd_ratio", "source_anomaly_pearson",
                   "source_drainage_coverage", "junction_receiver_km", "n_months", "within_source_range_fraction")
        panels[f"mixing_{kind}"] = aggregate_receivers(f, metrics)
    for kind in ("path_monthly", "path_hydro_common_calendar", "path_hydro_adjusted"):
        f = paths[paths.population.eq(kind)]
        metrics = (*OUTCOMES, "path_length_km", "storage_fraction", "added_drainage_fraction",
                   "source_area_km2", "wetland_change_pct", "forest_change_pct", "log_flow_ratio",
                   "temperature_change", "n_months")
        panels[kind] = aggregate_receivers(f, metrics)
    for name, f in panels.items():
        f = f.merge(shape, left_on="target", right_on="station", how="left", validate="one_to_one", indicator=True)
        f["inclusion_status"] = np.where(f._merge.eq("both"), "included", "no_eligible_morphology_panel")
        for row in f.itertuples():
            ledger.append({"analysis": name, "target": row.target, "huc4": row.huc4,
                           "component": row.component, "cluster": row.cluster, "status": row.inclusion_status})
        f = f[f.inclusion_status.eq("included")].drop(columns=["_merge", "inclusion_status"]).copy()
        f["log_n_months"] = np.log1p(f.n_months)
        if name.startswith("mixing"):
            f["log_junction_km"] = np.log1p(f.junction_receiver_km)
        else:
            f["log_path_km"] = np.log1p(f.path_length_km)
            f["log_source_area"] = np.log1p(f.source_area_km2)
        panels[name] = f.dropna(subset=list(FOCAL_TERMS)).reset_index(drop=True)
    return cases, paths, panels, pd.DataFrame(ledger)


def summarize(panels, draws):
    descriptors, contrasts, associations, loco = [], [], [], []
    for name, f in panels.items():
        metrics = [*OUTCOMES]
        if name.startswith("mixing"):
            metrics += ["log_mixture_sd_ratio", "log_downstream_mixture_sd_ratio", "source_anomaly_pearson"]
        for unit in ("component", "huc4"):
            for metric in metrics:
                for group in ("all", 1, 2, 3):
                    s = f if group == "all" else f[f.cluster.eq(group)]
                    result = cluster_mean(s, metric, unit, draws=draws)
                    if result is not None:
                        descriptors.append({"analysis": name, "group": str(group), **result})
                result = cluster_mean(f, metric, unit, draws=draws, contrast=True)
                if result is not None:
                    contrasts.append({"analysis": name, "comparison": "broad_minus_elongated", **result})
        models = {"connection_background": MIX_CONTROLS if name.startswith("mixing") else PATH_CONTROLS}
        if name.startswith("mixing"):
            models["plus_source_synchrony"] = (*MIX_CONTROLS, "source_anomaly_pearson")
        else:
            models["plus_hydro"] = (*PATH_CONTROLS, "log_flow_ratio", "temperature_change")
        for adjustment, controls in models.items():
            if name != "path_hydro_adjusted" and adjustment == "plus_hydro":
                continue
            for outcome in OUTCOMES:
                for focal in FOCAL_TERMS:
                    for unit in ("component", "huc4"):
                        result, sensitivity = conditional_association(f, outcome, focal, controls, unit=unit, draws=draws)
                        if result is None:
                            raise ValueError(f"non-identifiable design: {name}, {outcome}, {focal}, {adjustment}")
                        associations.append({"analysis": name, "adjustment": adjustment, **result})
                        if len(sensitivity):
                            sensitivity = sensitivity.assign(analysis=name, adjustment=adjustment,
                                                             outcome=outcome, focal=focal, unit=unit)
                            loco.append(sensitivity)
        print(f"Completed {name}: {len(f)} receivers, {f.component.nunique()} connected systems", flush=True)
    return {"class_descriptors": pd.DataFrame(descriptors), "class_contrasts": pd.DataFrame(contrasts),
            "morphology_associations": pd.DataFrame(associations),
            "leave_one_block_out": pd.concat(loco, ignore_index=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    cases, paths, panels, ledger = build_panels()
    cases.to_csv(out/"mixing_connections.csv", index=False)
    paths.to_csv(out/"path_connections.csv", index=False)
    ledger.to_csv(out/"inclusion_ledger.csv", index=False)
    for name, f in panels.items():
        f.to_csv(out/f"{name}_receivers.csv", index=False)
    summaries = summarize(panels, args.bootstrap_draws)
    for name, f in summaries.items():
        f.to_csv(out/f"{name}.csv", index=False)
    files = [MORPH, DATASET, ROOT/"study_plan.md", *[OLD/n for n in (
        "mixing_cases.csv", "mixing_records.parquet", "confluence_inventory.csv", "pathway_edges.csv", "source_cells.npy")]]
    code = [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
            Path("src/river_graph/analysis/river_form_process.py"), Path("src/river_graph/analysis/river_mechanisms.py")]
    for path in code:
        destination = ROOT/"code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    sources = {str(p): sha256_file(p) for p in [*files, *code]}
    record = {"source_hashes": sources, "bootstrap_draws": args.bootstrap_draws,
              "source_role": "deduplicated source-train union 142/143/144, exploratory",
              "receiver_counts": {name: {"receivers": len(f), "huc4": f.huc4.nunique(),
                                         "components": f.component.nunique(),
                                         "class_counts": f.cluster.value_counts().astype(int).to_dict()}
                                  for name, f in panels.items()},
              "models_retrained": False, "geographic_or_external_predictions_read": False,
              "unit": "equal receiver after connection averaging",
              "primary_bootstrap": "shared monitored-station connected components",
              "interpretation": "exploratory pointwise structure/process associations; not causal effects"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
