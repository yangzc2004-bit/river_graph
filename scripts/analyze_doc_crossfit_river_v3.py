"""Supplementary station resampling and evidence-linked cross-fit report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_crossfit_river_v2 import ATLAS, DATA, ROOT, digest, write

from river_graph.experiments.provenance import sha256_file
from river_graph.models.crossfit_river import RELATIONS, WAYS
from river_graph.models.river_structure_residual import upstream_paths


def resampling(frame, first, second, metric_name, rng):
    a = frame.query("model == @first").copy()
    b = frame.query("model == @second").copy()
    key = ["split", "seed", "station", "cell"]
    paired = a.merge(b[key+["y_pred"]], on=key, validate="one_to_one", suffixes=("", "_b"))
    if metric_name == "mae":
        paired["difference"] = abs(paired.y_pred-paired.y_true)-abs(paired.y_pred_b-paired.y_true)
    else:
        paired["difference"] = abs(np.log1p(paired.y_pred)-np.log1p(paired.y_true))-abs(np.log1p(paired.y_pred_b)-np.log1p(paired.y_true))
    paired["run"] = paired["split"].astype(str)+"_"+paired["seed"].astype(str)
    grouped = paired.groupby(["station", "run"])["difference"].agg(["sum", "count"])
    errors = grouped["sum"].unstack("run", fill_value=0).to_numpy()
    counts = grouped["count"].unstack("run", fill_value=0).to_numpy()
    n = len(errors)
    weights = rng.multinomial(n, np.full(n, 1/n), size=5000)
    den = weights@counts
    assert (den > 0).all()
    draws = ((weights@errors)/den).mean(1)
    return {"first": first, "second": second, "metric": metric_name,
        "mean_difference": float((errors.sum(0)/counts.sum(0)).mean()),
        "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)),
        "stations": n, "draws": 5000,
        "scope": "conditional station-resampling interval; repeated partition/seed cells kept together; exploratory"}


def audit_all_matching(root):
    """Verify all source-fold and evaluation controls against uncapped paths."""
    data = torch.load(DATA, weights_only=False)
    names = np.asarray(data["site_no"], str)
    edges = pd.read_csv(ATLAS/"station_edge_paths.csv", dtype={"source": str, "target": str})
    physical = pd.read_csv(ATLAS/"station_structure.csv", dtype={"station": str})
    reverse = edges.rename(columns={"source": "target", "target": "source"})
    attrs = physical.set_index("station")
    attrs.index = attrs.index.astype(str)
    count = 0
    for run in sorted((root/"runs").iterdir()):
        groups = json.loads((run/"source_roles.json").read_text())
        evaluation = json.loads((run/"evaluation_roles.json").read_text())
        groups += [{"held_stations": row["receiving"], "eligible_donors":
            np.setdiff1d(np.arange(len(names)), row["receiving"]).tolist(), "matching": row["matching"]} for row in evaluation]
        for group in groups:
            held = group["held_stations"]
            allowed = group["eligible_donors"]
            up = upstream_paths(edges, physical, names[held], names[allowed], candidates=len(allowed), max_km=np.inf)
            down = upstream_paths(reverse, physical, names[held], names[allowed], candidates=len(allowed), max_km=np.inf)
            ups = {q: {p[0] for p in up[i]} for i, q in enumerate(held)}
            downs = {q: {p[0] for p in down[i]} for i, q in enumerate(held)}
            for record in group["matching"]:
                receiver = record["receiver"]
                directed, null = record["directed"], record["nonconnected"]
                assert names[directed] in (ups[receiver] if record["direction"] == 1 else downs[receiver])
                assert names[null] not in ups[receiver] | downs[receiver]
                assert attrs.loc[names[null], "comid"] != attrs.loc[names[receiver], "comid"]
                assert {directed, null, record["generic"]} <= set(allowed)
                assert record["path_km"] <= 3000
                count += 1
    return count


def table(summary, role):
    subset = summary.query("role == @role")
    endpoints = ["mae", "log1p_mae", "rmse", "r2", "q90_mae"]
    optimum = {e: (subset[e].max() if e == "r2" else subset[e].min()) for e in endpoints}
    lines = ["| Training | Relationship | MAE | log1p MAE | RMSE | R2 | Q90 MAE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("local", *(f"{way}__{arm}" for way in WAYS for arm in RELATIONS)):
        row = subset.loc[subset.model == name].iloc[0]
        if name == "local":
            way, arm = "Local only", "No message"
        else:
            way, arm = name.split("__")
        cells = []
        for e in endpoints:
            cell = f"{row[e]:.4f}"
            if row[e] == optimum[e]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| "+" | ".join([way, arm, *cells])+" |")
    return "\n".join(lines)


def main():
    root = ROOT
    analysis = root/"analysis"
    audit = json.loads((analysis/"independent_audit.json").read_text())
    assert audit["passed"]
    protocol = json.loads((root/"protocol.json").read_text())
    pred = pd.read_parquet(analysis/"all_predictions.parquet")
    rows = []
    rng = np.random.default_rng(20261011)
    for role in ("val", "test"):
        frame = pred.query("role == @role")
        pairs = [(f"crossfit__{a}", f"fitted__{a}") for a in RELATIONS]
        pairs += [(f"{w}__directed", f"{w}__{c}") for w in WAYS for c in ("generic", "nonconnected")]
        pairs += [(f"{w}__{a}", "local") for w in WAYS for a in RELATIONS]
        for first, second in pairs:
            for endpoint in ("mae", "log1p_mae"):
                rows.append({"role": role, **resampling(frame, first, second, endpoint, rng)})
    intervals = pd.DataFrame(rows)
    intervals.to_csv(analysis/"station_resampling_intervals.csv", index=False)
    comparisons = pd.read_csv(analysis/"paired_comparisons.csv")
    interactions = []
    for role in ("val", "test"):
        for endpoint in ("mae", "log1p_mae"):
            for control in ("generic", "nonconnected"):
                a = comparisons.query("role == @role and metric == @endpoint and first == 'crossfit__directed' and second == @name", local_dict={"name": f"crossfit__{control}", "role": role, "endpoint": endpoint}).iloc[0]
                b = comparisons.query("role == @role and metric == @endpoint and first == 'fitted__directed' and second == @name", local_dict={"name": f"fitted__{control}", "role": role, "endpoint": endpoint}).iloc[0]
                interactions.append({"role": role, "metric": endpoint, "control": control,
                                     "crossfit_minus_fitted_relation_advantage": float(a.mean_difference-b.mean_difference)})
    pd.DataFrame(interactions).to_csv(analysis/"training_relationship_interactions.csv", index=False)
    checked = audit_all_matching(root)
    summary = pd.read_csv(analysis/"summary.csv")
    anchors = pd.read_csv(analysis/"source_anchor_scores.csv").groupby("way").mean(numeric_only=True)
    fitted_mae, oof_mae = anchors.loc["fitted", "mae"], anchors.loc["crossfit", "mae"]
    body = f"""# DOC whole-station cross-fitting and matched relationship controls

Protocol `{digest(protocol)}`. Nine fixed packages: original source-station partitions 142/143/144 by seeds 42/43/44. All 54 local fits and 54 correction fits completed and bound before evaluation. Checkpoints were replayed with zero numerical difference. {checked} real/negative-control matching records were independently verified, including every source fold, against uncapped directed paths. All six correction heads have identical seeded initial tensors and parameter counts. Full query coverage is 100%; the support column denotes the subset with a usable matched message.

This experiment isolates two correction-training strategies on a **new, clean common backbone**: covariate-only ecological ExtraTrees plus MLP/12-month GRU, then environmental-state attention. It does not reproduce the entire historical observed-DOC/retrieval chain. No historical supervised weights, chemistry, DOC-valued inputs or station embeddings are used. Input data and outer partitions match the selected branch; the absolute errors must not be described as a direct upgrade to a different historical pipeline.

The fitted strategy trains corrections on the full-source local predictor's source predictions and states. Crossfit uses five local models trained without any receiving-fold DOC, including all supervised encoders; both its anchor and latent state are fold isolated. At deployment every correction uses the **same full-source local model**, with no fold ensemble. Source-anchor training MAE averaged {fitted_mae:.4f} for fitted vs {oof_mae:.4f} for whole-station-held-out predictions; this quantifies the distribution mismatch, not test performance.

## Untouched outer-test stations: primary evaluation

{table(summary, 'test')}

All numbers are equal means of nine complete per-run metrics, including per-run R2; no seed ensemble. MAE/RMSE/Q90 MAE have the dataset's DOC concentration units (mg/L); log1p MAE and R2 are dimensionless. Q90 uses each run's source-training threshold. Independent observed-cell queries only, not scoring an unobserved continuous field.

## Source-validation stations: diagnostic evaluation

{table(summary, 'val')}

These labels selected no epoch or arm. Every neural component uses the pre-fixed final epoch (30). The outer validation query population is retained from the original branch, but the local predictor is freshly fitted; historical development metrics belong to their own pipeline.

## Paired effects, support and limits

`paired_comparisons.csv` contains crossfit-minus-fitted within all three relationships, genuine-minus-control within each training strategy, and every correction-minus-local effect. Negative difference means lower error. `training_relationship_interactions.csv` reports whether crossfit changes the directed relationship advantage. `support_strata.csv` retains all predictions and separates cells with/without a common usable candidate. Unsupported predictions reproduce the shared local baseline exactly; no model is evaluated only on supported cells.

`station_resampling_intervals.csv` provides supplementary 5000-draw paired station-resampling intervals for both primary errors. Repeated split/seed occurrences of a station travel together; run means are recomputed within each draw. These are conditional exploratory intervals for the fixed models and masks. They do not propagate refitting uncertainty, station-network dependence, or establish population-level significance. Three overlapping partitions and three seeds are not nine independent rivers. Inspect paired run signs and split-average ranges as well.

Real attention uses upstream and downstream paths as statistical context, including 0/1/3-month lagged environmental states. Downstream context does not imply backward physical DOC transport. Generic attention remains learned but has no direction/path input; donor count and date support are matched. Nonconnected attention keeps pseudo-path slot attributes while replacing donors with unrelated matched nodes; those copied attributes are a negative-control device, not asserted physical links. Generic sparse attention is not a dense all-node Transformer. The known 357-node atlas supplies only covariates and paths outside training roles; this is internal K0 transfer, not a genuinely unseen river or virtual-site validation.

Native and log1p MAE are jointly primary. Improvements in one error, RMSE or R2 alone do not justify an overall superiority claim. River-specific benefit requires consistent directed advantages over **both** generic and nonconnected controls. All six outcomes are retained. No released architecture, paper endpoint or default was changed based on this experiment.

Evidence: `experiments/phase4_transfer/doc_crossfit_river_v2/` protocol, manifest, scoring gate, run completion records, prediction/provenance pairs, independent audit and analysis CSVs. Runnable entry: `uv run python scripts/start_doc_crossfit_river_v2.py`; audit: `uv run python scripts/verify_doc_crossfit_river_v3.py`; supplementary report: `uv run python scripts/analyze_doc_crossfit_river_v3.py`.
"""
    report = Path("docs/doc_crossfit_river_results_20261011.md")
    report.write_text(body)
    write(analysis/"supplementary_audit.json", {"passed": True, "all_role_matching_records": checked,
        "bootstrap_draws": 5000, "protocol_hash": digest(protocol), "analysis_script_sha256": sha256_file(__file__),
        "report_sha256": sha256_file(report), "intervals_sha256": sha256_file(analysis/"station_resampling_intervals.csv")})
    print(summary.to_string(index=False))
    print(intervals.query("role == 'test'").to_string(index=False))


if __name__ == "__main__":
    main()
