"""Resolve actual monitored branches/trunks and reuse fixed DOC calibrators."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_form_process import (
    aggregate_receivers,
    cluster_mean,
    conditional_association,
)
from river_graph.analysis.river_monitored_footprint import (
    geometry_examples,
    matched_pulses,
    partition_footprint,
    timing_components,
)
from river_graph.analysis.river_observed_transport import calibrated_prediction
from river_graph.experiments.provenance import sha256_file
from river_graph.topology.river_structure import VAA_COLUMNS, ReachNetwork

ROOT = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1")
OLD = Path("experiments/phase4_transfer/doc_river_observed_transport_v1")
SIGNAL = Path("experiments/phase4_transfer/doc_river_signal_mechanisms_v1")
STRUCTURE = Path("experiments/phase4_transfer/doc_river_internal_structure_v1")
INVENTORY = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/confluence_inventory.csv")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
VAA = Path("cache/nldplus_vaa.parquet")
CODE = tuple(map(Path, ("scripts/analyze_doc_monitored_river_footprint_v1.py",
    "src/river_graph/analysis/river_monitored_footprint.py", "src/river_graph/topology/river_structure.py",
    "src/river_graph/analysis/river_form_process.py", "src/river_graph/analysis/river_observed_transport.py")))
TERMS = ("branch_balance", "independent_branch_cv", "common_fraction")
OUTCOMES = ("outlet_mixture_log_sd_ratio", "logscale_outlet_mixture_log_sd_ratio")
METRICS = ("mae", "log_mae", "mse", "bias", "q90_mae")
OPERATORS = {"same_month": "same_month_proxy", "common_trunk_only": "common_only_proxy",
             "equal_branches": "mean_delay_proxy", "actual_branches": "actual_branch_proxy"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_geometry():
    inputs = pd.read_parquet(OLD/"analysis/input_records.parquet")
    pairs = inputs.groupby("pair_id", as_index=False).first()
    columns = ["pair_id", "source_a", "source_b", "target", "huc4", "component", "cluster", "weight_a", "source_drainage_coverage"]
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "huc_cd": str}).set_index("site_no")
    net = ReachNetwork(pd.read_parquet(VAA, columns=list(VAA_COLUMNS)))
    inventory = pd.read_csv(INVENTORY, dtype={"source_a": str, "source_b": str, "target": str}).set_index("pair_id")
    rows, reaches = [], []
    for row in pairs.itertuples():
        endpoints = [(int(nodes.loc[s, "comid"]), float(nodes.loc[s, "measure"])) for s in (row.source_a, row.source_b, row.target)]
        m, r = partition_footprint(net, *endpoints, row.weight_a)
        old = inventory.loc[row.pair_id]
        np.testing.assert_allclose([m["path_a_km"], m["path_b_km"], m["common_km"]],
            [row.path_a_km, row.path_b_km, old.junction_receiver_km], atol=1e-8, rtol=1e-10)
        assert m["junction_comid"] == int(old.first_common_comid)
        assert str(row.huc4) == str(nodes.loc[row.target, "huc_cd"]).zfill(8)[:4]
        rows.append({name: getattr(row, name) for name in columns} | m)
        r["pair_id"] = row.pair_id
        reaches.append(r)
    frame = pd.DataFrame(rows)
    if len(frame) != 59 or frame.target.nunique() != 22 or frame.component.nunique() != 11:
        raise ValueError("retain the entire previous fixed observational cohort")
    return inputs, frame, pd.concat(reaches, ignore_index=True)


def predict_fixed(inputs, geometry, states, max_path):
    f = inputs.merge(geometry[["pair_id", "common_km"]], on="pair_id", validate="many_to_one")
    pieces = timing_components(f, max_path)
    if pieces.duplicated(["pair_id", "month_index"]).any():
        raise ValueError("unique connection/month records required")
    states = {int(s["held_component"]): s for s in states if s["operator"] == "mean_delay"}
    outputs = []
    for held, s in states.items():
        if held in s["training_components"] or s["fraction"] != 1.:
            raise ValueError("saved held-system mean-delay fit and fixed fraction one required")
        test = pieces[pieces.component.eq(held)]
        for operator, proxy in OPERATORS.items():
            product = test.copy()
            product["y_pred"] = calibrated_prediction(test, test[proxy].to_numpy(), s)
            product["operator"] = operator
            product["q90_threshold"] = s["q90_threshold"]
            product["high_doc"] = product.y_true >= s["q90_threshold"]
            product["visibility_role"] = "held_receiver_score_known_upstream_input"
            outputs.append(product)
    prediction = pd.concat(outputs, ignore_index=True)
    if not np.isfinite(prediction.y_pred).all() or len(prediction) != len(inputs)*len(OPERATORS):
        raise ValueError("complete finite predictions required")
    return pieces, prediction


def score_predictions(prediction):
    rows = []
    for (pair, operator), f in prediction.groupby(["pair_id", "operator"]):
        r = f.iloc[0]
        error = f.y_pred.to_numpy()-f.y_true.to_numpy()
        tail = f.high_doc.to_numpy(bool)
        rows.append({"pair_id": pair, "target": r.target, "huc4": r.huc4, "component": r.component,
            "operator": operator, "mae": np.mean(abs(error)), "log_mae": np.mean(abs(np.log1p(f.y_pred)-np.log1p(f.y_true))),
            "mse": np.mean(error**2), "bias": error.mean(), "q90_mae": np.mean(abs(error[tail])) if tail.any() else np.nan,
            "n_months": len(f), "n_tail_months": int(tail.sum())})
    connections = pd.DataFrame(rows)
    receiver = []
    for operator, s in connections.groupby("operator"):
        receiver.append(aggregate_receivers(s, METRICS).assign(operator=operator))
    return connections, pd.concat(receiver, ignore_index=True)


def summarize_errors(receiver, draws):
    ref = receiver[receiver.operator.eq("equal_branches")]
    summaries, comparisons, influence = [], [], []
    for operator, frame in receiver.groupby("operator"):
        for metric in METRICS:
            result = cluster_mean(frame, metric, "component", draws=draws)
            if result is not None:
                summaries.append({"operator": operator, **result})
        joined = frame.merge(ref, on=["target", "huc4", "component"], suffixes=("_candidate", "_reference"), validate="one_to_one")
        for metric in METRICS:
            pair = joined.dropna(subset=[metric+"_candidate", metric+"_reference"]).copy()
            pair["gain"] = pair[metric+"_reference"]-pair[metric+"_candidate"]
            if metric == "bias":
                # Positive here denotes reduced magnitude of the receiver mean bias.
                pair["gain"] = abs(pair[metric+"_reference"])-abs(pair[metric+"_candidate"])
            result = cluster_mean(pair, "gain", "component", draws=draws)
            if result is None:
                continue
            record = {"operator": operator, "reference": "equal_branches", **result, "metric": metric}
            _, index = np.unique(pair.component, return_inverse=True)
            n = int(index.max()+1)
            weight = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, index]
            base = pair[metric+"_reference"].to_numpy()
            candidate = pair[metric+"_candidate"].to_numpy()
            if metric == "bias":
                base, candidate = abs(base), abs(candidate)
            denominator = weight@base
            boot = np.divide(weight@(base-candidate), denominator, out=np.full(draws, np.nan), where=denominator > 0)
            finite = boot[np.isfinite(boot)]
            lo, hi = np.quantile(100*finite, [.025, .975]) if n >= 2 and len(finite) else (np.nan, np.nan)
            record.update(relative_gain_pct=100*(base.mean()-candidate.mean())/base.mean() if base.mean() > 0 else np.nan,
                          relative_ci_low=lo, relative_ci_high=hi)
            comparisons.append(record)
            for block in pair.component.unique():
                keep = pair.component.ne(block)
                influence.append({"operator": operator, "metric": metric, "omitted_component": int(block),
                    "gain": pair.loc[keep, "gain"].mean()})
    return {"model_summary": pd.DataFrame(summaries), "paired_error_comparisons": pd.DataFrame(comparisons),
            "error_omitted_systems": pd.DataFrame(influence)}


def observed_tables(geometry, pieces, draws):
    mix = pd.read_csv(SIGNAL/"analysis/mixing_connections.csv", dtype={"target": str, "huc4": str})
    connections = geometry.merge(mix[["pair_id", "n_months", "source_rho", "mixture_buffer_fraction", "equal_amplitude_buffer_fraction", *OUTCOMES]],
                                 on="pair_id", validate="one_to_one")
    metrics = [*TERMS, "total_path_cv", "mean_total_km", "source_drainage_coverage", "common_storage_fraction",
               "source_rho", "mixture_buffer_fraction", "equal_amplitude_buffer_fraction", *OUTCOMES]
    receiver = aggregate_receivers(connections, metrics)
    receiver["log_mean_path"] = np.log1p(receiver.mean_total_km)
    whole = pd.read_csv(STRUCTURE/"analysis/observed_receivers.csv", dtype={"target": str})
    receiver = receiver.merge(whole[["target", "tributary_balance", "path_cv", "basin_area_km2", "profile", "cluster"]], on="target", validate="one_to_one")
    associations, omitted = [], []
    for outcome in OUTCOMES:
        for focal in TERMS:
            controls = (*[t for t in TERMS if t != focal], "log_mean_path", "source_drainage_coverage")
            result, detail = conditional_association(receiver, outcome, focal, controls, draws=draws)
            if result is not None:
                associations.append(result)
                omitted.append(detail.assign(outcome=outcome, focal=focal))
    alignment = []
    for whole_term, matched in (("tributary_balance", "branch_balance"), ("path_cv", "total_path_cv"), ("path_cv", "independent_branch_cv")):
        alignment.append({"whole_descriptor": whole_term, "footprint_descriptor": matched,
            "spearman": receiver[whole_term].corr(receiver[matched], method="spearman"), "n_receivers": len(receiver)})
    piece_metrics = []
    for column in ("shared_delay_input", "equal_branch_delay_input", "differential_arrival_input"):
        name = column+"_abs"
        pieces[name] = abs(pieces[column])
        piece_metrics.append(name)
    contrasts = pieces.groupby(["pair_id", "target", "huc4", "component"], as_index=False)[piece_metrics].mean()
    input_receivers = aggregate_receivers(contrasts, piece_metrics)
    input_summary = [cluster_mean(input_receivers, metric, "component", draws=draws) for metric in piece_metrics]
    return {"observed_connections": connections, "observed_receivers": receiver,
        "structural_associations": pd.DataFrame(associations), "association_omitted_systems": pd.concat(omitted, ignore_index=True),
        "scale_alignment": pd.DataFrame(alignment), "input_connections": contrasts,
        "input_receivers": input_receivers, "input_summary": pd.DataFrame(input_summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    inputs, geometry, reaches = build_geometry()
    reps, cuts = geometry_examples(geometry)
    previous = json.loads((OLD/"config.json").read_text())
    states = json.loads((OLD/"analysis/fitted_states.json").read_text())
    config = {"bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42, "bootstrap_unit": "connected monitoring system",
        "aggregation": "dates within pair, pairs within receiver, receivers equal", "corridor": "primary downstream station-to-station route",
        "geometry_example_cuts": cuts, "pulse_sd": .15, "pulse_dt": .0025, "max_path_km": previous["max_path_km"],
        "prediction_reference": "equal_branches (saved complete mean-delay model)", "calibrators_refitted": False,
        "previous_results_seen": True, "neural_training": False, "structural_terms": TERMS, "observed_outcomes": OUTCOMES}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    geometry.to_csv(out/"footprints.csv", index=False)
    reaches.to_csv(out/"corridor_reaches.csv", index=False)
    reps.to_csv(out/"representatives.csv", index=False)
    pulses, traces = [], []
    for row in geometry.itertuples():
        p, trace = matched_pulses(row.path_a_km, row.path_b_km, row.common_km, row.weight_a)
        p["pair_id"], p["target"], p["component"] = row.pair_id, row.target, row.component
        pulses.append(p)
        if row.pair_id in set(reps.pair_id):
            trace["pair_id"] = row.pair_id
            traces.append(trace)
    pieces, prediction = predict_fixed(inputs, geometry, states, config["max_path_km"])
    cm, rm = score_predictions(prediction)
    tables = {"pulse_scenarios": pd.concat(pulses, ignore_index=True), "connection_errors": cm, "receiver_errors": rm,
              **summarize_errors(rm, args.bootstrap_draws), **observed_tables(geometry, pieces, args.bootstrap_draws)}
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    pieces.to_parquet(out/"timing_inputs.parquet", index=False)
    prediction.to_parquet(out/"fixed_predictions.parquet", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(out/"representative_pulses.parquet", index=False)
    files = [OLD/"analysis/input_records.parquet", OLD/"analysis/fitted_states.json", OLD/"analysis/connection_predictions.provenance.json",
        OLD/"analysis/connection_predictions.parquet", OLD/"config.json", SIGNAL/"analysis/mixing_connections.csv",
        SIGNAL/"analysis_sources.json", STRUCTURE/"analysis/observed_receivers.csv", INVENTORY, NODES, VAA, ROOT/"study_plan.md", *CODE]
    for path in CODE:
        dest = ROOT/"code_snapshot"/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    hashes = {str(p): sha256_file(p) for p in files}
    record = {"source_hashes": hashes, "config_hash": digest(config), "n_connections": len(geometry), "n_receivers": 22,
        "n_monitoring_systems": 11, "n_connection_months": len(inputs),
        "n_unique_receiver_months": len(inputs.drop_duplicates(["target", "month_index"])), "model_refitting": False}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    origin = json.loads((OLD/"analysis/connection_predictions.provenance.json").read_text())
    runtime = {str(p): hashes[str(p)] for p in CODE}
    sidecar = {"config": config, "config_hash": digest(config), "dataset_hash": origin["dataset_hash"], "mask_hash": origin["mask_hash"],
        "source_cells_sha256": origin["source_cells_sha256"], "runtime_sources": runtime, "runtime_snapshot_hash": digest(runtime),
        "prediction_sha256": sha256_file(out/"fixed_predictions.parquet"), "saved_states_sha256": hashes[str(OLD/"analysis/fitted_states.json")],
        "input_records_sha256": hashes[str(OLD/"analysis/input_records.parquet")], "prediction_grain": "connection x observed month x operator",
        "product_role": "fixed held-system model input sensitivity; no refitting", "visibility": "known current/previous source DOC; receiver truth for scores only"}
    (out/"fixed_predictions.provenance.json").write_text(json.dumps(sidecar, indent=2)+"\n")
    print(geometry[["branch_balance", "independent_branch_cv", "common_fraction", "common_storage_fraction"]].describe().to_string())
    print(tables["paired_error_comparisons"].query("metric == 'mae'").to_string(index=False))
    print(tables["structural_associations"].to_string(index=False))


if __name__ == "__main__":
    main()
