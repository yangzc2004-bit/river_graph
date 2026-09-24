"""Reproducible QC replay and availability audits for Stage 1 (no training)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.experiments.masks import (
    make_e1,
    make_e2_partial,
    make_e2_strict,
    make_e3,
)
from river_graph.experiments.transfer import (
    ANALYTES,
    UNITS,
    array,
    array_hash,
    availability_tasks,
    file_hash,
    object_hash,
)

RAW_COLUMNS = {
    "Location_Identifier": "station", "Activity_StartDate": "date",
    "Result_Characteristic": "characteristic", "Result_Measure": "value",
    "Result_MeasureUnit": "unit", "Result_SampleFraction": "fraction",
    "Result_ResultDetectionCondition": "detection", "USGSpcode": "pcode",
}


def qc_replay(raw_dir: Path, datasets: dict) -> tuple[dict, pd.DataFrame]:
    """Replay legacy target filters against the current cache, never rewrite datasets.

    This is retrospective label reproduction, not contemporaneous provenance.
    Bad files and missing columns are fatal to QC status and individually logged.
    """
    files = sorted(raw_dir.glob("*.csv"))
    records, frames, errors = [], [], []
    for path in files:
        records.append({"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size})
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, dtype=str)
            frame = frame.rename(columns=RAW_COLUMNS)
            if not set(RAW_COLUMNS.values()) <= set(frame):
                raise ValueError("missing WQP columns")
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - failed raw records must be enumerated
            errors.append({"file": str(path), "error": str(exc)})
    if not frames:
        return {"passed": False, "errors": errors or ["no raw files"]}, pd.DataFrame(records)
    raw = pd.concat(frames, ignore_index=True)
    raw["station"] = raw.station.str.removeprefix("USGS-")
    raw["value"] = pd.to_numeric(raw.value, errors="coerce")
    raw["month"] = pd.to_datetime(raw.date, errors="coerce", format="mixed").dt.strftime("%Y-%m")
    uncen = raw.detection.isna() | raw.detection.fillna("").eq("")
    rules = {
        "doc": raw.characteristic.eq("Organic carbon") & (
            raw.pcode.eq("00681") | (raw.pcode.isna() & raw.fraction.str.lower().isin(
                ["dissolved", "filtered field and/or lab"]))) & raw.unit.str.lower().eq("mg/l"),
        "ph": raw.characteristic.eq("pH") & raw.unit.eq("standard units") & raw.value.between(0, 14),
        "spec_conductance": raw.characteristic.eq("Specific conductance") & raw.unit.eq("uS/cm")
        & raw.value.between(0, 100000),
    }
    results = {}
    for a, valid in rules.items():
        ds = datasets[a]
        sites = list(ds["site_no"])
        months = [str(m)[:7] for m in ds["months"]]
        keep = valid & uncen & raw.value.notna() & raw.month.isin(months) & raw.station.isin(sites)
        sub = raw.loc[keep]
        means = sub.groupby(["station", "month"]).value.mean()
        grid = means.unstack().reindex(index=sites, columns=months)
        rebuilt_mask = grid.notna().to_numpy()
        rebuilt = grid.fillna(0).to_numpy(dtype=np.float32)
        mask, y = array(ds["y_mask"]), array(ds["y"])
        common = mask & rebuilt_mask
        diff = np.abs(y[common] - rebuilt[common])
        mismatch = int((mask != rebuilt_mask).sum())
        # Relative tolerance only accommodates alternative floating summation orders.
        values_match = bool(np.allclose(y[common], rebuilt[common], rtol=1e-6, atol=1e-6))
        results[a] = {
            "accepted_raw_rows": len(sub), "replayed_cells": int(rebuilt_mask.sum()),
            "mask_mismatches": mismatch, "values_match_at_1e6": values_match,
            "max_abs_difference": float(diff.max()) if len(diff) else None,
            "mask_bitwise_equal": bool(np.array_equal(mask, rebuilt_mask)),
            "values_bitwise_equal": bool(np.array_equal(y[common], rebuilt[common])),
            "passed": mismatch == 0 and values_match and bool(common.any()),
        }
    return {
        "passed": bool(files) and not errors and all(v["passed"] for v in results.values()),
        "identity_status": "retrospective_raw_cache_reproduction",
        "not_contemporaneous": True, "errors": errors, "analytes": results,
        "raw_manifest_sha256": object_hash(records), "raw_files": len(records),
        "rules": "original DOC pcode/fraction+unit+censor filters; original pH/EC exact units, "
                 "uncensored and physical ranges; monthly arithmetic mean; no score-based filtering",
    }, pd.DataFrame(records)


def availability_audit(datasets: dict, nodes: pd.DataFrame, regions_path: Path) -> tuple:
    """Inventory two distinct objects: reconstruction masks and K=5 transfer tasks."""
    ref = datasets["doc"]
    sites = list(ref["site_no"])
    edges = pd.DataFrame([(sites[i], sites[j]) for i, j in array(ref["edge_index"]).T],
                         columns=["source", "target"])
    regions = json.loads(regions_path.read_text())["primary"]
    mask_rows, task_rows, task_files, aliases = [], [], {}, []
    for a in ANALYTES:
        ds = datasets[a]
        mask = array(ds["y_mask"])
        months = pd.DatetimeIndex(ds["months"])
        splits = {"E1": make_e1(mask)["e1_r20_seed42"],
                  "E2a": make_e2_strict(mask, months),
                  "E2b": make_e2_partial(mask, months),
                  "E3": make_e3(mask, edges, sites)}
        for family, split in splits.items():
            sets = {k: set(map(int, split[k])) for k in ("train", "val", "test")}
            disjoint = not any(sets[x] & sets[y] for x, y in (("train", "val"), ("train", "test"), ("val", "test")))
            mask_rows.append({
                "analyte": a, "family": family,
                **{f"n_{k}": len(split[k]) for k in ("train", "val", "test")},
                "test_stations": len(set(np.asarray(split["test"], dtype=int) // mask.shape[1])),
                "test_months": len(set(np.asarray(split["test"], dtype=int) % mask.shape[1])),
                "roles_disjoint": disjoint,
                "available": disjoint and all(len(split[k]) > 0 for k in ("train", "val", "test")),
                "role": "reconstruction_control_availability_not_transfer_mask",
                "mask_content_hash": object_hash({k: np.asarray(v).tolist() for k, v in split.items()}),
            })
        for region in regions:
            rows = region["task_component_rows"]
            hide = region["hide_rows"]
            canonical = sorted(nodes.iloc[hide].huc6.unique())
            if len(canonical) != 1:
                raise ValueError(f"legacy task spans canonical HUC6 codes: {region['code']}")
            actual_hide = np.flatnonzero(nodes.huc6.to_numpy() == canonical[0]).tolist()
            if sorted(hide) != actual_hide:
                raise ValueError(f"legacy hide membership differs for {region['code']}")
            graph = nx.Graph()
            graph.add_nodes_from(hide)
            graph.add_edges_from((int(i), int(j)) for i, j in array(ref["edge_index"]).T
                                 if i in hide and j in hide)
            if set(rows) not in list(nx.connected_components(graph)) or len(rows) != max(
                len(c) for c in nx.connected_components(graph)
            ):
                raise ValueError("legacy component not a largest component on current graph")
            if a == "doc":
                aliases.append({"task_id": region["code"], "canonical_huc6": canonical[0],
                                "hide_rows": hide, "task_component_rows": rows})
            counts = None
            for seed in (42, 43, 44):
                tasks = availability_tasks(mask, rows, ds["months"], seed)
                task_files[f"{a}/{region['code']}/{seed}.json"] = {
                    "role": "availability_inventory_only_already_exposed_not_confirmatory",
                    "analyte": a, "task_id": region["code"], "canonical_huc6": canonical[0],
                    "seed": seed, "k_list": [0, 1, 3, 5], "tasks": tasks,
                }
                counts = {
                    "analyte": a, "task_id": region["code"], "canonical_huc6": canonical[0],
                    "task_months": len(tasks), "query_cells": sum(len(t["query_cells"]) for t in tasks),
                    "post2020_task_months": sum(t["month"] > "2020-12" for t in tasks),
                    "pre2021_task_months": sum(t["month"] <= "2020-12" for t in tasks),
                    "component_stations": len(rows), "ci_under20_clusters": len(tasks) < 20,
                    "available": bool(tasks),
                }
            task_rows.append(counts)
    return pd.DataFrame(mask_rows), pd.DataFrame(task_rows), task_files, aliases


def unified_labels(datasets: dict, summaries: dict, nodes: pd.DataFrame) -> pd.DataFrame:
    """Observed cells only. This is an audit table, never a training view."""
    frames = []
    for a in ANALYTES:
        ds = datasets[a]
        rows, cols = np.nonzero(array(ds["y_mask"]))
        frames.append(pd.DataFrame({
            "analyte": a, "station": np.asarray(ds["site_no"])[rows],
            "station_index": rows, "month_index": cols,
            "month": np.asarray(ds["months"])[cols], "huc6": nodes.huc6.to_numpy()[rows],
            "value": array(ds["y"])[rows, cols], "unit": UNITS[a],
            "dataset_hash": summaries[a]["sha256"], "visibility_role": "audit_only_observed",
        }))
    return pd.concat(frames, ignore_index=True)


def label_content_bindings(datasets: dict) -> dict:
    return {a: {"y": array_hash(ds["y"]), "y_mask": array_hash(ds["y_mask"])}
            for a, ds in datasets.items()}
