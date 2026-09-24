"""Reproducible QC replay and availability audits for Stage 1 (no training)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.data.wqp import (
    extract_covariate_obs,
    extract_doc_obs,
    load_station_results,
)
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


def _external_station_ids(path: Path, huc8: str) -> set[str]:
    """Return valid stream station IDs from one WQP station inventory.

    The WQP station endpoint can return diversions and other non-stream
    locations even when a stream filter is supplied.  We therefore apply the
    location type and coordinate checks again before a station can enter the
    external graph inventory.
    """
    try:
        frame = pd.read_csv(path, dtype=str, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"{path}: incomplete or malformed station CSV") from exc
    if "Location_Identifier" not in frame.columns:
        raise ValueError(f"{path}: station identifier header missing")
    required = {
        "Location_Identifier", "Location_HUCEightDigitCode", "Location_Type",
        "Location_LatitudeStandardized", "Location_LongitudeStandardized",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing station columns {sorted(missing)}")
    frame = frame[
        frame["Location_HUCEightDigitCode"].eq(huc8)
        & frame["Location_Type"].eq("Stream")
    ].copy()
    lat = pd.to_numeric(frame["Location_LatitudeStandardized"], errors="coerce")
    lon = pd.to_numeric(frame["Location_LongitudeStandardized"], errors="coerce")
    frame = frame[lat.between(-90, 90) & lon.between(-180, 180)]
    return set(frame["Location_Identifier"].str.removeprefix("USGS-").dropna())


def _wqp_complete(path: Path) -> bool:
    """Structural check only; a valid CSV cannot prove provider completeness."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False
    if not text.startswith("Org_Identifier,") or not text.endswith("\n"):
        return False
    if "INCOMPLETE DATA" in text.upper():
        return False
    try:
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows)
        required = {
            "Location_Identifier", "Activity_StartDate", "Result_Characteristic",
            "Result_Measure", "Result_MeasureUnit", "Result_ResultDetectionCondition",
        }
        return required.issubset(header) and all(len(row) == len(header) for row in rows)
    except (csv.Error, StopIteration):
        return False


def external_raw_inventory(
    huc8: str,
    station_paths: dict[str, Path],
    result_paths: dict[str, Path],
    limits: dict[str, float | int],
) -> dict:
    """Audit raw WQP evidence for one external candidate, without training.

    This is deliberately a screening artifact.  It never creates a graph or
    selects a basin from prediction results.  A result file with WQP's
    ``INCOMPLETE DATA`` marker is ineligible even if its partial rows happen
    to exceed a threshold.
    """
    missing = set(ANALYTES).difference(station_paths) | set(ANALYTES).difference(result_paths)
    if missing:
        raise ValueError(f"missing analyte paths: {sorted(missing)}")
    station_sets = {
        a: _external_station_ids(Path(station_paths[a]), huc8) for a in ANALYTES
    }
    common = sorted(set.intersection(*station_sets.values()))
    analyte_rows: dict[str, dict] = {}
    for analyte in ANALYTES:
        path = Path(result_paths[analyte])
        complete = path.exists() and _wqp_complete(path)
        summary = {
            "result_path": str(path), "result_sha256": file_hash(path) if path.exists() else None,
            "result_complete": complete, "metadata_stations": len(station_sets[analyte]),
            "common_metadata_stations": len(common), "active_stations": None,
            "common_active_stations": None, "common_station_months": None,
            "station_months": None, "months": None, "min_month": None, "max_month": None,
            "accepted_rows": None, "eligible": False,
            "completeness_scope": "structural_only_provider_total_unverified",
        }
        if complete:
            tidy = load_station_results(path)
            if analyte == "doc":
                obs = extract_doc_obs(tidy)
            else:
                obs = extract_covariate_obs(tidy, analyte)
            # The frozen data gate permits analyte-specific active masks on a
            # shared graph. Counts use each analyte's valid stream inventory;
            # the all-three intersection is reported as a mapping diagnostic.
            obs = obs[obs["site_no"].isin(station_sets[analyte])].copy()
            obs["month"] = obs["date"].dt.to_period("M")
            obs = obs.dropna(subset=["month"])
            cells = obs.drop_duplicates(["site_no", "month"])
            common_cells = cells[cells["site_no"].isin(common)]
            summary.update({
                "active_stations": int(cells["site_no"].nunique()),
                "common_active_stations": int(common_cells["site_no"].nunique()),
                "station_months": len(cells), "months": int(cells["month"].nunique()),
                "common_station_months": len(common_cells),
                "min_month": str(cells["month"].min()) if len(cells) else None,
                "max_month": str(cells["month"].max()) if len(cells) else None,
                "accepted_rows": len(obs),
            })
        summary["eligible"] = bool(
            summary["result_complete"]
            and summary["active_stations"] >= limits["minimum_stations"]
            and summary["months"] >= limits["minimum_months"]
            and summary["station_months"] >= limits["minimum_station_months_per_analyte"]
        )
        analyte_rows[analyte] = summary
    eligible = all(row["eligible"] for row in analyte_rows.values())
    return {
        "huc8": huc8, "station_type": "Stream", "common_metadata_stations": len(common),
        "common_station_ids_sha256": object_hash(common), "station_sets": {
            a: {"count": len(s), "sha256": object_hash(sorted(s)),
                "path": str(station_paths[a]), "file_sha256": file_hash(station_paths[a])}
            for a, s in station_sets.items()
        }, "analytes": analyte_rows, "eligible": eligible,
        "limits": limits, "selection_role": "availability_screen_only",
        "completeness_scope": "structural_only_provider_total_unverified",
        "stage1_passed": False,
        "pending_checks": ["graph_connectivity", "feature_pipeline", "temporal_roles",
                           "provider_completeness", "contemporaneous_provenance"],
        "identity_status": "retrospective_raw_file_binding_not_download_provenance",
        "qc_code": {str(p): file_hash(p) for p in [
            Path(__file__), Path("src/river_graph/data/wqp.py"),
            Path("src/river_graph/data/quality.py")
        ]},
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
