"""Compare three real river forms using identical inputs and mapped storage."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_morphology_effect import (
    group_descriptor_summary,
    paired_response_summary,
)
from river_graph.analysis.river_whole_storage import StoragePaths, network_responses
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_whole_storage_v1")
MORPH = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis")
ROUTING = Path("data/raw/river_source_placement_v1/routing")
CACHE = Path("data/processed/river_whole_storage_v1")
VAA = Path("cache/nldplus_vaa.parquet")
COLS = ["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm", "wbareatype", "wbareacomi"]
FRACTIONS = (0., .25, .5, 1.)
SIGMAS = (.075, .15, .30)
DT = .0025
METRICS = ("pulse_peak", "duration_80", "pulse_sd", "peak_reduction_pct", "duration_change_pct",
           "storage_variance", "storage_share_of_structural_variance")
CODE = tuple(Path(p) for p in (
    "src/river_graph/analysis/river_whole_storage.py",
    "scripts/analyze_doc_river_whole_storage_v1.py",
    "scripts/plot_doc_river_whole_storage_v1.py",
    "scripts/verify_doc_river_whole_storage_v1.py",
    "tests/test_river_whole_storage.py"))


def context():
    panel = pd.read_csv(MORPH/"station_morphology_doc_panel.csv", dtype={"station": str, "huc4": str})
    pairs = pd.read_csv(MORPH/"covariate_selected_pairs.csv", dtype={"station_a": str, "station_b": str, "huc4": str})
    reps = pd.read_csv(PREVIOUS/"network_representatives.csv", dtype={"station": str})
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    return panel, pairs, reps, vaa


def make_paths(row, vaa):
    cache = np.load(ROUTING/f"comid_{int(row.comid)}.npz")
    frame = vaa.loc[cache["comids"]]
    reachable = np.isfinite(cache["distance_km"])
    denominator = frame.areasqkm.sum()
    paths = StoragePaths(frame[reachable], int(row.comid), cache["distance_km"][reachable],
                         scale=np.sqrt(row.basin_area_km2))
    coverage = frame.loc[reachable, "areasqkm"].sum()/denominator
    return paths, coverage


def simulate(row, vaa, *, dt=DT, curves=False):
    paths, coverage = make_paths(row, vaa)
    rows, trace = network_responses(paths, fractions=FRACTIONS, sigmas=SIGMAS, dt=dt, keep_curves=curves)
    sensitive, _ = network_responses(paths, fractions=(0., .5), sigmas=(.15,), points=5, dt=dt)
    return paths.descriptors() | {"represented_area_fraction": coverage}, rows+sensitive, trace


def add_contrasts(frame):
    keys = ["station", "input_sd", "source_points"]
    baseline = frame[frame.storage_fraction.eq(0)].set_index(keys)
    out = frame.copy()
    index = pd.MultiIndex.from_frame(out[keys])
    out["peak_reduction_pct"] = 100*(1-out.pulse_peak.to_numpy()/baseline.loc[index, "pulse_peak"].to_numpy())
    out["duration_change_pct"] = 100*(out.duration_80.to_numpy()/baseline.loc[index, "duration_80"].to_numpy()-1)
    denominator = out.path_variance+out.storage_variance
    out["storage_share_of_structural_variance"] = np.divide(out.storage_variance, denominator,
        out=np.zeros(len(out)), where=denominator > 0)
    return out


def summaries(frame, descriptors, pairs, draws=5000):
    grouped, matched, evidence = [], [], []
    keys = ["input_sd", "storage_fraction", "source_points"]
    for setting, f in frame.groupby(keys):
        info = dict(zip(keys, setting))
        g = group_descriptor_summary(f, METRICS, draws=draws)
        for name, value in info.items():
            g[name] = value
        grouped.append(g)
        m, e = paired_response_summary(pairs, f, dict.fromkeys(METRICS, "raw"), draws=draws)
        for name, value in info.items():
            m[name], e[name] = value, value
        matched.append(m)
        evidence.append(e)
    descriptor_metrics = ["path_variance", "storage_mean_share", "storage_exposed_flow_share",
                          "serial_storage_variance", "mean_serial_elements"]
    geometry = group_descriptor_summary(descriptors, descriptor_metrics, draws=draws)
    # Within-class distributions are geometry distributions, not uncertainty intervals.
    within = frame.groupby(keys+["cluster"]).agg(n=("station", "size"),
        peak_reduction_q10=("peak_reduction_pct", lambda x: x.quantile(.1)),
        peak_reduction_median=("peak_reduction_pct", "median"),
        peak_reduction_q90=("peak_reduction_pct", lambda x: x.quantile(.9)),
        positive_peak_reduction_fraction=("peak_reduction_pct", lambda x: np.mean(x > 1e-7)),
        duration_change_q10=("duration_change_pct", lambda x: x.quantile(.1)),
        duration_change_median=("duration_change_pct", "median"),
        duration_change_q90=("duration_change_pct", lambda x: x.quantile(.9))).reset_index()
    return {"class_summary": pd.concat(grouped, ignore_index=True),
            "matched_contrasts": pd.concat(matched, ignore_index=True),
            "matched_evidence": pd.concat(evidence, ignore_index=True),
            "geometry_summary": geometry, "within_class": within}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    panel, pairs, reps, vaa = context()
    a = ROOT/"analysis"
    a.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    engine = sha256_file("src/river_graph/analysis/river_whole_storage.py")
    vaa_hash = sha256_file(VAA)
    started = time.monotonic()
    scenarios, descriptors, traces = [], [], []
    example_comids = set(reps.comid)
    input_hashes = {}
    for i, row in enumerate(panel.itertuples()):
        source = ROUTING/f"comid_{int(row.comid)}.npz"
        input_hashes[str(source)] = sha256_file(source)
        identity = {"engine": engine, "vaa": vaa_hash, "routes": input_hashes[str(source)],
                    "basin_area_km2": row.basin_area_km2, "dt": DT, "fractions": list(FRACTIONS), "sigmas": list(SIGMAS)}
        cache = CACHE/f"comid_{int(row.comid)}.json"
        saved = json.loads(cache.read_text()) if cache.exists() and not args.no_cache else None
        if saved is not None and saved["identity"] != identity:
            raise ValueError(f"derived cache identity changed: {cache}; use a new version or --no-cache with a repair record")
        if saved is None:
            descriptor, rows, trace = simulate(row, vaa, curves=row.comid in example_comids)
            saved = {"identity": identity, "descriptor": descriptor, "rows": rows, "trace": trace}
            cache.write_text(json.dumps(saved))
        info = {"station": row.station, "comid": int(row.comid), "cluster": int(row.cluster), "huc4": row.huc4}
        scenarios.extend(r | info for r in saved["rows"])
        descriptors.append(saved["descriptor"] | info)
        if row.station in set(reps.station):
            traces.extend(r | info for r in saved["trace"])
        if i % 15 == 0 or i+1 == len(panel):
            print(f"{i+1}/{len(panel)} station-network instances; elapsed {time.monotonic()-started:.1f}s", flush=True)
    scenario = add_contrasts(pd.DataFrame(scenarios))
    descriptor = pd.DataFrame(descriptors)
    tables = {"scenario_metrics": scenario, "network_descriptors": descriptor,
              **summaries(scenario, descriptor, pairs, draws=args.bootstrap_draws), "representatives": reps}
    # Half-step replay for fixed examples plus storage extremes selected by geometry alone.
    replay = sorted(set(reps.station) | set(descriptor.nlargest(1, "serial_storage_variance").station))
    numerical = []
    for station in replay:
        row = next(r for r in panel.itertuples() if r.station == station)
        _, fine, _ = simulate(row, vaa, dt=DT/2)
        f = pd.DataFrame(fine).set_index(["storage_fraction", "input_sd", "source_points"])
        c = scenario[scenario.station.eq(station)].set_index(f.index.names).loc[f.index]
        for metric in ("pulse_peak", "duration_80", "numerical_centroid", "numerical_sd", "anomaly_area_fraction"):
            numerical.append({"station": station, "metric": metric,
                "max_absolute_difference": abs(f[metric]-c[metric]).max(), "coarse_dt": DT, "fine_dt": DT/2})
    tables["numerical_convergence"] = pd.DataFrame(numerical)
    for name, frame in tables.items():
        frame.to_csv(a/f"{name}.csv", index=False)
    pd.DataFrame(traces).to_parquet(a/"representative_responses.parquet", index=False)
    config = {"fractions": FRACTIONS, "input_sds": SIGMAS, "dt": DT,
              "source_sensitivity_points": 5, "sensitivity_fraction": .5,
              "bootstrap_draws": args.bootstrap_draws, "cohort": "unchanged 297 instances / 295 receiving reaches",
              "storage_grouping": "contiguous mapped reaches sharing positive physical waterbody COMID",
              "time_unit": "channel km / sqrt basin area km2; conceptual common velocity"}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    summary = {"n_instances": len(panel), "n_distinct_outlets": panel.comid.nunique(), "n_scenarios": len(scenario),
        "class_counts": panel.groupby("cluster").size().to_dict(), "n_huc4": panel.huc4.nunique(),
        "max_centroid_error": abs(scenario.numerical_centroid-scenario.pulse_centroid).max(),
        "max_sd_error": abs(scenario.numerical_sd-scenario.pulse_sd).max(),
        "max_unit_gain_error": abs(scenario.anomaly_area_fraction-1).max(),
        "n_storage_exposed": int(descriptor.storage_exposed_flow_share.gt(0).sum()),
        "largest_route_error_km": descriptor.max_route_error_km.max()}
    (a/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    sources = [MORPH/"station_morphology_doc_panel.csv", MORPH/"covariate_selected_pairs.csv",
               PREVIOUS/"network_representatives.csv", VAA, ROOT/"study_plan.md", *CODE]
    manifest = {"source_hashes": input_hashes | {str(p): sha256_file(p) for p in sources},
                "product_hashes": {str(p): sha256_file(p) for p in sorted(a.iterdir())},
                "config_hash": sha256_file(ROOT/"config.json"),
                "role": "exploratory identical-input geometry experiment; not observed DOC effects"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
