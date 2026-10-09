"""Export DOC components and empirical intervals for a new station-month grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from portable_doc_ensemble_v1 import PortableDOCEnsemble
from run_unified_doc_spatial import digest, write_json

from river_graph.experiments.provenance import sha256_file

BUNDLE = Path("experiments/phase4_transfer/doc_portable_source_fit_v1/ensemble")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--inputs", type=Path, required=True, help="label-free NPZ using the saved feature semantics")
    parser.add_argument("--output", type=Path, required=True, help="new output parquet path")
    parser.add_argument("--k", type=int, choices=(0, 1, 3, 5), default=0)
    parser.add_argument("--support", type=Path, help="NPZ containing integer cells and DOC values")
    args = parser.parse_args()
    meta = args.output.with_suffix(".meta.json")
    diagnostics = args.output.with_suffix(".sources.json")
    if args.output.exists() or meta.exists() or diagnostics.exists():
        raise FileExistsError("retain the existing prediction and choose a new output path")
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {key: archive[key].copy() for key in ("site_no", "months", "x", "x_mask", "static", "regime")}
        inputs["daily_features"] = archive["daily_features" if "daily_features" in archive.files else "daily"].copy()
        if "river_edges" in archive.files:
            inputs["river_edges"] = archive["river_edges"].copy()
    if bool(args.k) != bool(args.support):
        raise ValueError("K1/3/5 require explicit supports; K0 does not accept support labels")
    members = PortableDOCEnsemble.load(args.bundle)
    components = members.predict_components(inputs)
    prediction = components["final_pred"]
    support_cells = np.empty(0, dtype=np.int64)
    if args.k:
        with np.load(args.support, allow_pickle=False) as archive:
            support_cells, support_values = archive["cells"].copy(), archive["values"].copy()
        adapted = members.predict_with_support(inputs, k=args.k,
            support_cells=support_cells, support_values=support_values)
        prediction = adapted["final_pred"]
        components["pi_lower"], components["pi_upper"] = adapted["pi_lower"], adapted["pi_upper"]
    n, t = prediction.shape
    roles = np.full(n*t, "unmonitored" if args.k == 0 else "support_adapted", dtype=object)
    roles[support_cells] = "support"
    frame = pd.DataFrame({"analyte": "doc", "station": np.repeat(inputs["site_no"].astype(str), t),
        "month": np.tile(inputs["months"].astype(str), n), "y_pred": prediction.ravel(),
        "visibility_role": roles, "model_name": "environmental_temporal_doc", "seed": "42-46 ensemble", "k": args.k})
    for key in ("environment_pred", "local_temporal_correction", "source_transfer_correction", "river_correction",
                "native_pred", "member_prediction_sd", "pi_lower", "pi_upper", "hydro_availability"):
        frame[key] = np.asarray(components[key]).ravel()
    frame["support_correction"] = (prediction-components["final_pred"]).ravel()
    frame["interval_width"] = frame.pi_upper-frame.pi_lower
    config = {"operation": "portable DOC full-grid inference", "analyte": "doc", "output_units": "mg/L",
        "bundle_path": str(args.bundle), "ensemble_manifest_hash": sha256_file(args.bundle/"ensemble.json"),
        "inputs_path": str(args.inputs), "input_hash": sha256_file(args.inputs), "k": args.k,
        "support_hash": None if args.support is None else sha256_file(args.support),
        "runner_sha256": sha256_file(__file__), "n_stations": n, "n_months": t,
        "water_quality_input": "none at K0; only designated DOC supports at K1/3/5",
        "interval": "source-validation empirical calibration", "support_correction": "retrospective output correction"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    write_json(diagnostics, {"retrieval_sources": components["retrieval_sources"]})
    write_json(meta, {"config": config, "config_hash": digest(config), "prediction_sha256": sha256_file(args.output),
        "source_diagnostics_sha256": sha256_file(diagnostics), "member_manifests": json.loads((args.bundle/"ensemble.json").read_text())["member_manifests"]})
    print(f"Saved {len(frame):,} predicted station-months to {args.output}", flush=True)


if __name__ == "__main__":
    main()
