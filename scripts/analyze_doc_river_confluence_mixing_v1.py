"""Measure real mapped junctions while retaining the original network classes."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_storage_placement_v1 import COLS, ROUTING, VAA, real_paths

from river_graph.analysis.river_confluence_geometry import (
    METRICS,
    measure_junction,
    paired_form_contrasts,
)
from river_graph.analysis.river_storage_placement import selected_corridor
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_planform import project_geometry, read_cached_lines

ROOT = Path("experiments/phase4_transfer/doc_river_confluence_mixing_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_storage_placement_v1/analysis")
PAIRS = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis/covariate_selected_pairs.csv")
CACHE = Path("data/raw/river_planform_v1/flowlines.sqlite")
CLASSES = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/classes.csv")
FROZEN_PANEL = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis/station_morphology_doc_panel.csv")
CONFIG = {"scales_m": [100., 250., 500.], "primary_scale_m": 250., "gap_tolerance_m": 20.,
          "turning_step_m": 25., "bootstrap_draws": 5000, "bootstrap_seed": 42,
          "bootstrap_unit": "HUC4 groups of original matched pairs",
          "observed_mixing_data_available": False,
          "scope": "actual local centreline geometry in previously selected ST357 networks; no observed mixing outcomes"}
DTYPES = {k: str for k in ("station", "case_id", "station_a", "station_b", "huc4", "physical_junction_id")}
CODE = tuple(Path(p) for p in (
    "src/river_graph/analysis/river_confluence_geometry.py",
    "src/river_graph/analysis/river_monitored_footprint.py",
    "src/river_graph/analysis/river_storage_placement.py",
    "src/river_graph/analysis/river_whole_storage.py",
    "src/river_graph/topology/river_planform.py",
    "scripts/analyze_doc_river_storage_placement_v1.py",
    "scripts/analyze_doc_river_confluence_mixing_v1.py",
    "scripts/fetch_doc_river_confluence_mixing_v1.py",
    "scripts/plot_doc_river_confluence_mixing_v1.py",
    "scripts/verify_doc_river_confluence_mixing_v1.py",
    "tests/test_river_confluence_geometry.py",
))


def one_junction(row, vaa, connection):
    paths, _, _ = real_paths(row, vaa)
    selection = {k: int(getattr(row, k)) for k in ("junction_index", "source_a_index", "source_b_index")}
    for label in ("junction", "source_a", "source_b"):
        if int(paths.comids[selection[label+"_index"]]) != int(getattr(row, label+"_comid")):
            raise ValueError("saved selected junction indices no longer identify the same reaches")
    reaches = selected_corridor(paths, selection)
    cached = read_cached_lines(connection, reaches.comid.tolist())
    missing = sorted(set(reaches.comid)-set(cached))
    provenance = {"station": row.station, "receiving_comid": int(row.comid),
                  "n_corridor_reaches": len(reaches), "n_missing_mapped_reaches": len(missing),
                  "missing_comids": json.dumps(missing),
                  "routing_sha256": sha256_file(ROUTING/f"comid_{int(row.comid)}.npz")}
    import hashlib

    digest = hashlib.sha256()
    for cid in sorted(cached):
        digest.update(str(cid).encode()+b"\0"+cached[cid].wkb)
    provenance["used_geometry_sha256"] = digest.hexdigest()
    identifiers = {k: getattr(row, k) for k in ("station", "comid", "cluster", "huc4", "junction_comid",
                                              "source_a_comid", "source_b_comid", "weight_a")}
    identifiers["contributing_area_balance"] = 4*row.weight_a*(1-row.weight_a)
    for segment in ("branch_a", "branch_b"):
        identifiers[segment+"_inlet_comid"] = int(reaches[reaches.segment.eq(segment)].sort_values("sequence").comid.iloc[-1])
    identifiers["physical_junction_id"] = str(int(row.junction_comid))
    if missing:
        measured = pd.DataFrame([identifiers | {"scale_m": scale, "status": "missing_mapped_reaches"}
                                 | {m: np.nan for m in METRICS} for scale in CONFIG["scales_m"]])
        return measured, pd.DataFrame(), provenance
    projected = {cid: project_geometry(g) for cid, g in cached.items()}
    measured, local = measure_junction(reaches, projected, row.weight_a,
                                      scales=CONFIG["scales_m"], gap_tolerance=CONFIG["gap_tolerance_m"],
                                      turning_step=CONFIG["turning_step_m"])
    for k, value in identifiers.items():
        measured[k] = value
    vertices = []
    if CONFIG["primary_scale_m"] in local:
        for segment, line in local[CONFIG["primary_scale_m"]].items():
            for i, (x, y) in enumerate(np.asarray(line.coords)[:, :2]):
                vertices.append({"station": row.station, "junction_comid": int(row.junction_comid),
                                 "scale_m": CONFIG["primary_scale_m"], "segment": segment,
                                 "vertex": i, "x_m": x, "y_m": y})
    return measured, pd.DataFrame(vertices), provenance


def summarize(geometry):
    rows = []
    for (scale, cluster), group in geometry.groupby(["scale_m", "cluster"], sort=True):
        kept = group[group.status.eq("measured")]
        for metric in METRICS+("contributing_area_balance",):
            values = kept[metric]
            rows.append({"scale_m": scale, "cluster": cluster, "metric": metric,
                         "n_selected_networks": len(group), "n_measured": len(kept),
                         "n_physical_junctions": kept.junction_comid.nunique(), "n_huc4": kept.huc4.nunique(),
                         "mean": values.mean(), "median": values.median(),
                         "q25": values.quantile(.25), "q75": values.quantile(.75)})
    reuse = geometry[geometry.scale_m.eq(CONFIG["primary_scale_m"])].groupby("junction_comid", sort=True).agg(
        n_receiving_networks=("station", "size"), n_forms=("cluster", "nunique"),
        forms=("cluster", lambda x: ",".join(map(str, sorted(x.unique())))),
        receiving_stations=("station", lambda x: ",".join(sorted(x))),
        branch_a_inlets=("branch_a_inlet_comid", "nunique"), branch_b_inlets=("branch_b_inlet_comid", "nunique"),
    ).reset_index()
    sensitivity = []
    for metric in METRICS:
        wide = geometry.pivot(index="station", columns="scale_m", values=metric)
        for a, b in ((100., 250.), (250., 500.), (100., 500.)):
            matched = wide[[a, b]].dropna()
            sensitivity.append({"metric": metric, "scale_a_m": a, "scale_b_m": b,
                                "n_networks": len(matched), "spearman_rho": matched.corr(method="spearman").iloc[0, 1],
                                "median_absolute_difference": (matched[a]-matched[b]).abs().median()})
    return pd.DataFrame(rows), reuse, pd.DataFrame(sensitivity)


def build():
    cases = pd.read_csv(PREVIOUS/"selected_footprints.csv", dtype=DTYPES)
    cases = cases[cases.cohort.eq("whole_confluence")].copy()
    if len(cases) != 295 or cases.station.duplicated().any():
        raise ValueError("the original 295 selected real network junctions are required")
    classes = pd.read_csv(FROZEN_PANEL, dtype=DTYPES)
    joined = cases[["station", "cluster"]].merge(classes[["station", "cluster"]], on="station", validate="one_to_one")
    if len(classes) != 297 or len(joined) != 295 or not np.array_equal(joined.cluster_x.to_numpy(), joined.cluster_y.to_numpy()):
        raise ValueError("original whole-network form labels changed")
    pairs = pd.read_csv(PAIRS, dtype=DTYPES)
    pairs = pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].copy()
    if len(pairs) != 22:
        raise ValueError("the original 22 elongated/broad matched pairs are required")
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    frames, vertices, inventory = [], [], []
    started = time.monotonic()
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for i, row in enumerate(cases.itertuples()):
            measured, lines, receipt = one_junction(row, vaa, connection)
            frames.append(measured)
            if not lines.empty:
                vertices.append(lines)
            inventory.append(receipt)
            if i % 40 == 0 or i+1 == len(cases):
                print(f"Actual junction geometry {i+1}/{len(cases)}; {time.monotonic()-started:.1f}s", flush=True)
    geometry = pd.concat(frames, ignore_index=True)
    summary, reuse, sensitivity = summarize(geometry)
    contrasts, pair_detail = paired_form_contrasts(geometry, pairs, draws=CONFIG["bootstrap_draws"])
    status = geometry.groupby(["scale_m", "status"], as_index=False).size().rename(columns={"size": "n_networks"})
    outputs = {"local_junction_geometry": geometry, "form_geometry_summary": summary,
               "physical_junction_reuse": reuse, "measurement_scale_sensitivity": sensitivity,
               "matched_form_contrasts": contrasts, "matched_pair_geometry": pair_detail,
               "mapped_geometry_inventory": pd.DataFrame(inventory), "measurement_status": status}
    primary = geometry[geometry.scale_m.eq(CONFIG["primary_scale_m"]) & geometry.status.eq("measured")]
    result = {"n_original_classified_networks": len(classes), "n_selected_networks": len(cases),
              "n_measured_primary_scale": len(primary), "n_unique_primary_junctions": primary.junction_comid.nunique(),
              "n_junctions_reused_across_networks": int(reuse.n_receiving_networks.gt(1).sum()),
              "n_junctions_shared_between_forms": int(reuse.n_forms.gt(1).sum()),
              "n_original_matched_pairs": len(pairs), "primary_scale_m": CONFIG["primary_scale_m"],
              "scope": CONFIG["scope"], "observed_mixing_event_rows_analyzed": 0,
              "doc_reaction_or_removal_estimated": False}
    return outputs, pd.concat(vertices, ignore_index=True), result


def main():
    analysis = ROOT/"analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (ROOT/"config.json").write_text(json.dumps(CONFIG, indent=2)+"\n")
    tables, vertices, summary = build()
    for name, frame in tables.items():
        frame.to_csv(analysis/f"{name}.csv", index=False)
    vertices.to_parquet(analysis/"local_centerlines.parquet", index=False)
    (analysis/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    sources = [PREVIOUS/"selected_footprints.csv", PREVIOUS/"representatives.csv", PAIRS, CLASSES, FROZEN_PANEL, VAA,
               ROOT/"study_plan.md", ROOT/"config.json", ROOT/"access_records.json"]
    raw_metadata = Path("data/raw/river_confluence_mixing_v1/datacite.json")
    if raw_metadata.exists():
        sources.append(raw_metadata)
    for source in CODE:
        destination = ROOT/"code_snapshot"/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (ROOT/"analysis_sources.json").write_text(json.dumps({
        "scope": CONFIG["scope"], "geometry_crs": "EPSG:5070", "cached_geometry_store": str(CACHE),
        "used_geometry_hashes": "analysis/mapped_geometry_inventory.csv, per corridor WKB content",
        "source_hashes": {str(p): sha256_file(p) for p in sources},
        "code_hashes": {str(p): sha256_file(p) for p in CODE},
        "product_hashes": {str(p): sha256_file(p) for p in analysis.iterdir() if p.is_file()},
    }, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
