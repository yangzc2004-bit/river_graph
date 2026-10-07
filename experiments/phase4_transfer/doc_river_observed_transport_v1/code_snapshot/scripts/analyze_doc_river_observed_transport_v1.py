"""Relate real tributary arrival operators to held-out observed downstream DOC."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.analysis.river_observed_transport import (
    FRACTIONS,
    OPERATORS,
    connection_metrics,
    nested_predictions,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_observed_transport_v1")
OLD = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis")
CONTEXT = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis/network_context.csv")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
METRICS = ("mae", "log_mae", "mse", "bias", "q90_mae", "signal_rho")


def build_records(data=None):
    data = torch.load(DATASET, map_location="cpu", weights_only=False) if data is None else data
    visible = permitted_doc(data, np.load(OLD/"source_cells.npy"))
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    months = pd.DatetimeIndex(data["months"])
    if not months.to_period("M").equals(pd.period_range(months[0], months[-1], freq="M")):
        raise ValueError("dataset requires unique, continuous calendar months")
    nodes = pd.read_csv(CONTEXT, dtype={"station": str, "huc4": str})
    candidates = pd.read_csv(OLD/"confluence_inventory.csv", dtype={"target": str, "source_a": str, "source_b": str, "huc4": str})
    candidates = candidates[candidates.eligible_monthly & candidates.target.isin(nodes.station)].copy()
    candidates = candidates.merge(nodes[["station", "basin_area_km2"]], left_on="target", right_on="station", validate="many_to_one")
    max_path = float(candidates[["source_a_receiver_km", "source_b_receiver_km"]].max().max())
    x, mask = np.asarray(data["x"], float), np.asarray(data["x_mask"], bool)
    qi, ti = (list(data["feature_channels"]).index(c) for c in ("discharge", "temperature"))
    records, ledger, coverage = [], [], []
    for row in candidates.itertuples():
        a, b, target = (index[s] for s in (row.source_a, row.source_b, row.target))
        source_good = np.isfinite(visible[a]) & np.isfinite(visible[b])
        common = source_good & np.isfinite(visible[target])
        for lag in (0, 1, 2, 3, 6, 12):
            good = common.copy()
            good[:lag] = False
            for k in range(1, lag+1):
                good &= np.roll(source_good, k)
            coverage.append({"pair_id": row.pair_id, "target": row.target, "component": int(row.component),
                             "max_lag": lag, "n_months": int(good.sum()), "eligible_24": good.sum() >= 24})
            if lag == 1:
                chosen = np.flatnonzero(good)
        eligible = len(chosen) >= 24
        ledger.append({"pair_id": row.pair_id, "target": row.target, "huc4": row.huc4,
                       "component": int(row.component), "cluster": int(row.cluster), "n_months": len(chosen),
                       "included": eligible, "reason": "included" if eligible else "fewer_than_24_common_current_previous_months"})
        if not eligible:
            continue
        dates = months[chosen]
        angle = 2*np.pi*dates.month.to_numpy()/12
        flow = np.where(mask[target, chosen, qi] & (x[target, chosen, qi] > 0), x[target, chosen, qi], np.nan)
        temperature = np.where(mask[target, chosen, ti], x[target, chosen, ti], np.nan)
        f = pd.DataFrame({"date": dates, "month_index": chosen, "pair_id": row.pair_id,
                          "source_a": row.source_a, "source_b": row.source_b, "target": row.target,
                          "huc4": row.huc4, "component": int(row.component), "cluster": int(row.cluster),
                          "doc_a_now": visible[a, chosen], "doc_b_now": visible[b, chosen],
                          "doc_a_previous": visible[a, chosen-1], "doc_b_previous": visible[b, chosen-1],
                          "y_true": visible[target, chosen],
                          "weight_a": row.source_a_area_km2/(row.source_a_area_km2+row.source_b_area_km2),
                          "path_a_km": row.source_a_receiver_km, "path_b_km": row.source_b_receiver_km,
                          "path_difference_scaled": abs(row.source_a_receiver_km-row.source_b_receiver_km)/np.sqrt(row.basin_area_km2),
                          "source_drainage_coverage": row.source_drainage_coverage,
                          "month_sin": np.sin(angle), "month_cos": np.cos(angle),
                          "year_fraction": dates.year.to_numpy()+(dates.month.to_numpy()-1)/12,
                          "log_basin_area": np.log1p(row.basin_area_km2),
                          "log_discharge": np.log1p(flow), "temperature": temperature})
        records.append(f)
    return pd.concat(records, ignore_index=True), pd.DataFrame(ledger), pd.DataFrame(coverage), max_path


def receiver_metrics(connections):
    context = connections.groupby("target", as_index=False).agg(huc4=("huc4", "first"), component=("component", "first"),
                                                              cluster=("cluster", "first"))
    f = connections.groupby(["target", "operator"], as_index=False)[list(METRICS)].mean()
    f["rmse"] = np.sqrt(f.mse)
    return f.merge(context, on="target", validate="many_to_one")


def paired_gains(receivers, draws):
    choices = [(op, "background") for op in OPERATORS if op != "background"]
    choices += [(op, "same_month") for op in ("uniform_history", "mean_delay", "branch_arrival")]
    choices += [("branch_arrival", "mean_delay"), ("branch_arrival", "uniform_history"), ("mean_delay", "uniform_history")]
    rows = []
    for candidate, reference in choices:
        f = receivers[receivers.operator.eq(candidate)].merge(receivers[receivers.operator.eq(reference)],
            on=["target", "huc4", "component", "cluster"], suffixes=("_candidate", "_reference"), validate="one_to_one")
        for group in ("all", "1", "2", "3"):
            s = f if group == "all" else f[f.cluster.eq(int(group))]
            for unit in ("component", "huc4"):
                for metric in ("mae", "log_mae", "q90_mae"):
                    valid = s.dropna(subset=[metric+"_candidate", metric+"_reference"])
                    if valid.empty:
                        continue
                    base, c = (valid[metric+suffix].to_numpy() for suffix in ("_reference", "_candidate"))
                    _, gi = np.unique(valid[unit], return_inverse=True)
                    n = int(gi.max()+1)
                    w = np.random.default_rng(42).multinomial(n, np.full(n, 1/n), size=draws)[:, gi]
                    delta = (w@(base-c))/w.sum(axis=1)
                    gain = 100*(1-(w@c)/(w@base))
                    lo, hi = np.quantile(delta, [.025, .975]) if n > 1 else (np.nan, np.nan)
                    glo, ghi = np.quantile(gain, [.025, .975]) if n > 1 else (np.nan, np.nan)
                    rows.append({"candidate": candidate, "reference": reference, "group": group, "metric": metric,
                                 "reference_mean": base.mean(), "candidate_mean": c.mean(),
                                 "error_reduction": np.mean(base-c), "ci_low": lo, "ci_high": hi,
                                 "relative_reduction_pct": 100*(1-c.mean()/base.mean()), "gain_ci_low_pct": glo, "gain_ci_high_pct": ghi,
                                 "unit": unit, "n_receivers": len(valid), "n_blocks": n,
                                 "n_positive_receivers": int((c < base).sum()), "interval_estimable": n > 1})
    return pd.DataFrame(rows)


def summarize(receivers, draws):
    rows = []
    for operator, f in receivers.groupby("operator"):
        for group in ("all", "1", "2", "3"):
            s = f if group == "all" else f[f.cluster.eq(int(group))]
            for unit in ("component", "huc4"):
                for metric in (*METRICS, "rmse"):
                    r = cluster_mean(s, metric, unit, draws=draws)
                    if r is not None:
                        rows.append({"operator": operator, "group": group, **r})
    return pd.DataFrame(rows)


def system_sensitivity(receivers):
    """Influence of each connected system on receiver-equal paired effects."""
    rows = []
    for excluded in sorted(receivers.component.unique()):
        kept = receivers[receivers.component.ne(excluded)]
        for candidate, reference in (("same_month", "background"), ("uniform_history", "same_month"),
                                     ("branch_arrival", "same_month"), ("branch_arrival", "mean_delay"),
                                     ("branch_arrival", "uniform_history")):
            a = kept[kept.operator.eq(candidate)].set_index("target")
            b = kept[kept.operator.eq(reference)].set_index("target")
            for metric in ("mae", "log_mae", "q90_mae"):
                paired = pd.concat([a[metric], b[metric]], axis=1, keys=["candidate", "reference"]).dropna()
                rows.append({"excluded_component": int(excluded), "candidate": candidate, "reference": reference,
                             "metric": metric, "n_receivers": len(paired),
                             "error_reduction": (paired.reference-paired.candidate).mean()})
    return pd.DataFrame(rows)


def receiver_coverage(predictions):
    rows = []
    # Operators repeat identical truth cells. Connections may repeat them too.
    f = predictions[predictions.operator.eq("same_month")]
    for target, s in f.groupby("target"):
        unique = s.drop_duplicates("month_index")
        r = s.iloc[0]
        rows.append({"target": target, "huc4": r.huc4, "component": r.component, "cluster": r.cluster,
                     "n_pairs": s.pair_id.nunique(), "n_unique_months": len(unique),
                     "n_unique_q90": int(unique.high_doc.sum()), "tail_unstable": unique.high_doc.sum() < 20})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame, ledger, coverage, max_path = build_records()
    predictions, trials, states = nested_predictions(frame, max_path)
    connections = connection_metrics(predictions)
    receivers = receiver_metrics(connections)
    for name, f in {"inclusion_ledger": ledger, "window_availability": coverage, "selection_trials": trials,
                    "connection_metrics": connections, "receiver_metrics": receivers,
                    "receiver_coverage": receiver_coverage(predictions),
                    "system_sensitivity": system_sensitivity(receivers),
                    "operator_summary": summarize(receivers, args.bootstrap_draws),
                    "paired_gains": paired_gains(receivers, args.bootstrap_draws)}.items():
        f.to_csv(out/f"{name}.csv", index=False)
    frame.to_parquet(out/"input_records.parquet", index=False)
    predictions.to_parquet(out/"connection_predictions.parquet", index=False)
    (out/"fitted_states.json").write_text(json.dumps(states, indent=2)+"\n")
    code = [Path("scripts/analyze_doc_river_observed_transport_v1.py"), Path("src/river_graph/analysis/river_observed_transport.py"),
            Path("src/river_graph/analysis/river_mechanisms.py"), Path("src/river_graph/analysis/river_form_process.py")]
    for p in code:
        dest = ROOT/"code_snapshot"/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    config = {"operators": OPERATORS, "fractions": FRACTIONS, "min_common_months": 24, "lookback": 2,
              "max_path_km": max_path, "ridge_alpha": 1., "inner_folds": 3,
              "outer": "leave_connected_monitoring_system_out", "source_role": "union142/143/144",
              "training_weight": "equal receiver, then equal pair, then equal date", "target_transform": "log1p"}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    mask_payload = frame[["pair_id", "target", "source_a", "source_b", "month_index", "component"]].to_csv(index=False).encode()
    runtime = {str(p): sha256_file(p) for p in code}
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    sidecar = {"config_hash": digest(config), "dataset_hash": sha256_file(DATASET),
               "mask_hash": hashlib.sha256(mask_payload).hexdigest(), "runtime_snapshot_hash": digest(runtime),
               "prediction_sha256": sha256_file(out/"connection_predictions.parquet"),
               "config": config, "runtime_sources": runtime, "study_plan_sha256": sha256_file(ROOT/"study_plan.md"),
               "source_cells_sha256": sha256_file(OLD/"source_cells.npy"),
               "visibility": "held receiving DOC never used to select/fill/scale/fit its operator; observed upstream current/previous DOC is an allowed input",
               "prediction_grain": "connection x observed receiver month x operator; shared receiver observations repeat across connections",
               "calibrators_fitted": True, "existing_neural_models_retrained": False}
    (out/"connection_predictions.provenance.json").write_text(json.dumps(sidecar, indent=2)+"\n")
    files = [DATASET, CONTEXT, OLD/"source_cells.npy", OLD/"confluence_inventory.csv", ROOT/"study_plan.md", *code]
    record = {"source_hashes": {str(p): sha256_file(p) for p in files}, "bootstrap_draws": args.bootstrap_draws,
              "receivers": frame.target.nunique(), "pairs": frame.pair_id.nunique(), "components": frame.component.nunique(),
              "pair_months": len(frame), "unique_receiver_months": len(frame.drop_duplicates(["target", "month_index"])),
              "class_receiver_counts": frame.drop_duplicates("target").cluster.value_counts().to_dict(),
              "max_path_km": max_path, "existing_neural_models_retrained": False,
              "interpretation": "monthly geometry-guided information diagnostic; not physical travel-time estimation or unmonitored K0 validation"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
