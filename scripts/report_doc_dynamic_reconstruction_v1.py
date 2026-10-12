"""Post-run paired diagnostics; does not select or train any model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1")
ARMS = ("monthly_log", "daily_log", "monthly_joint", "daily_joint",
        "rf_monthly", "rf_daily", "xgb_monthly", "xgb_daily")
NAMES = {"monthly_log": "RiverDOCNet: original inputs and log loss",
         "daily_log": "RiverDOCNet: daily-flow descriptors and log loss",
         "monthly_joint": "RiverDOCNet: original inputs and joint loss",
         "daily_joint": "RiverDOCNet: daily-flow descriptors and joint loss",
         "rf_monthly": "History random forest: original inputs",
         "rf_daily": "History random forest: daily-flow descriptors",
         "xgb_monthly": "History XGBoost: original inputs",
         "xgb_daily": "History XGBoost: daily-flow descriptors"}
METRICS = ("mae", "log1p_mae", "rmse", "r2", "station_centered_r2",
           "log1p_station_centered_r2", "within_station_sd_ratio",
           "monthly_delta_mae", "monthly_delta_correlation")


def sufficient_statistics(part):
    y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
    residual = p-y
    centered = pd.DataFrame({"station": part.station, "y": y, "p": p})
    c = centered[["y", "p"]] - centered.groupby("station")[["y", "p"]].transform("mean")
    log = pd.DataFrame({"station": part.station, "y": np.log1p(y), "p": np.log1p(p)})
    lc = log[["y", "p"]] - log.groupby("station")[["y", "p"]].transform("mean")
    ordered = part.sort_values(["station", "month"])
    d = ordered.groupby("station")[["month", "y_true", "y_pred"]].diff()
    d = d[d.month == 1]
    a, b = d.y_true.to_numpy(float), d.y_pred.to_numpy(float)
    return {"n": len(y), "ae": abs(residual).sum(),
            "logae": abs(np.log1p(p)-np.log1p(y)).sum(), "sse": np.square(residual).sum(),
            "ysum": y.sum(), "y2sum": np.square(y).sum(),
            "within_y2": np.square(c.y).sum(), "within_p2": np.square(c.p).sum(),
            "within_sse": np.square(c.p-c.y).sum(),
            "log_within_y2": np.square(lc.y).sum(), "log_within_sse": np.square(lc.p-lc.y).sum(),
            "dn": len(a), "dae": abs(b-a).sum(), "dzero": abs(a).sum(),
            "dasum": a.sum(), "dbsum": b.sum(), "da2": np.square(a).sum(),
            "db2": np.square(b).sum(), "dab": (a*b).sum()}


def metrics_from_statistics(s):
    n = s["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sst = s["y2sum"] - s["ysum"]**2/n
        av = s["da2"] - s["dasum"]**2/s["dn"]
        bv = s["db2"] - s["dbsum"]**2/s["dn"]
        dc = (s["dab"]-s["dasum"]*s["dbsum"]/s["dn"])/np.sqrt(av*bv)
        return {"mae": s["ae"]/n, "log1p_mae": s["logae"]/n,
                "rmse": np.sqrt(s["sse"]/n), "r2": 1-s["sse"]/sst,
                "station_centered_r2": 1-s["within_sse"]/s["within_y2"],
                "log1p_station_centered_r2": 1-s["log_within_sse"]/s["log_within_y2"],
                "within_station_sd_ratio": np.sqrt(s["within_p2"]/s["within_y2"]),
                "monthly_delta_mae": s["dae"]/s["dn"],
                "monthly_delta_correlation": dc, "zero_change_delta_mae": s["dzero"]/s["dn"]}


def spatial_bootstrap(frame, summary):
    regions = sorted(frame.target_huc4.astype(str).unique())
    seeds = sorted(frame.seed.unique())
    rows = [{"region": str(region), "seed": seed, "arm": arm, **sufficient_statistics(part)}
            for (region, seed, arm), part in frame.groupby(["target_huc4", "seed", "model_name"])]
    stats = pd.DataFrame(rows)
    keys = [c for c in stats if c not in ("region", "seed", "arm")]
    draws = np.random.default_rng(20261012).multinomial(len(regions), np.repeat(1/len(regions), len(regions)), size=10000)
    estimated = {}
    for arm in ARMS:
        sampled = []
        for seed in seeds:
            a = stats[(stats.arm == arm) & (stats.seed == seed)].set_index("region").loc[regions, keys]
            totals = draws @ a.to_numpy(float)
            sampled.append(metrics_from_statistics({key: totals[:, i] for i, key in enumerate(keys)}))
        estimated[arm] = {metric: np.mean([s[metric] for s in sampled], axis=0) for metric in METRICS}
        exact = []
        for seed in seeds:
            a = stats[(stats.arm == arm) & (stats.seed == seed)][keys].sum()
            exact.append(metrics_from_statistics(a.to_dict()))
        reference = summary[(summary.track == "space") & (summary.model_name == arm)].iloc[0]
        for metric in METRICS:
            value = np.mean([s[metric] for s in exact])
            # Native float32 delta means and float64 sufficient-statistic sums
            # differ at rounding precision; all concentration metrics use float64.
            assert abs(value-reference[metric]) < 1e-6, (arm, metric, value, reference[metric])
    intervals = []
    for arm in ARMS:
        for metric in METRICS:
            delta = estimated[arm][metric]-estimated["monthly_log"][metric]
            lower, upper = np.quantile(delta, [.025, .975])
            exact = summary[(summary.track == "space") & (summary.model_name == arm)].iloc[0][metric]
            base = summary[(summary.track == "space") & (summary.model_name == "monthly_log")].iloc[0][metric]
            intervals.append({"arm": arm, "metric": metric, "difference_from_matched_control": exact-base,
                              "lower_95": lower, "upper_95": upper, "bootstrap_units": len(regions), "draws": len(draws)})
    return stats, pd.DataFrame(intervals)


def markdown_table(frame, columns):
    headers = ["Model", *columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    ordered = frame.set_index("model_name")
    for arm in ARMS:
        row = ordered.loc[arm]
        lines.append("| " + " | ".join([NAMES[arm], *[f"{row[c]:.4f}" for c in columns]]) + " |")
    return "\n".join(lines)


def station_diagnostics(frame):
    rows = []
    for (seed, arm, station), part in frame.groupby(["seed", "model_name", "station"]):
        if len(part) < 12:
            continue
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        a, b = np.log1p(y), np.log1p(p)
        a, b = a-a.mean(), b-b.mean()
        denominator = np.square(a).sum()
        if denominator <= 0:
            continue
        correlation = np.corrcoef(a, b)[0, 1] if np.std(b) > 1e-6 else np.nan
        rows.append({"seed": seed, "model_name": arm, "station": station,
                     "observed_months": len(part), "mae": abs(p-y).mean(),
                     "log_centered_r2": 1-np.square(b-a).sum()/denominator,
                     "log_sd_ratio": np.sqrt(np.square(b).sum()/denominator),
                     "log_temporal_correlation": correlation})
    station = pd.DataFrame(rows)
    summaries = []
    for (seed, arm), part in station.groupby(["seed", "model_name"]):
        summaries.append({"seed": seed, "model_name": arm, "eligible_stations": len(part),
                          "median_log_centered_r2": part.log_centered_r2.median(),
                          "mean_log_centered_r2": part.log_centered_r2.mean(),
                          "fraction_positive_log_centered_r2": (part.log_centered_r2 > 0).mean(),
                          "median_log_sd_ratio": part.log_sd_ratio.median(),
                          "median_log_temporal_correlation": part.log_temporal_correlation.median(),
                          "correlation_eligible_stations": int(part.log_temporal_correlation.notna().sum())})
    by_seed = pd.DataFrame(summaries)
    return station, by_seed.groupby("model_name", as_index=False).mean(numeric_only=True)


def plot_curves(frame, choice, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    control = frame[(frame.model_name == "monthly_log") & (frame.seed == 42)]
    density = control.groupby("station").size().sort_values(ascending=False, kind="stable")
    sites = list(density.head(4).index)
    comparison = choice if choice != "monthly_log" else "daily_joint"
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)
    selections = []
    for ax, station in zip(axes, sites, strict=True):
        observations = control[control.station == station].sort_values("month")
        available = np.zeros(len(data["months"]), int)
        available[observations.month.to_numpy()] = 1
        counts = np.convolve(available, np.ones(48, int), mode="valid")
        start = int(np.argmax(counts)); end = start+48
        times = pd.DatetimeIndex(data["months"])[start:end]
        visible = observations[observations.month.between(start, end-1)]
        ax.scatter(pd.DatetimeIndex(data["months"])[visible.month], np.log1p(visible.y_true), s=15, c="black", label="Observed available-sample month mean", zorder=3)
        for arm, color, label in (("monthly_log", "#3575B2", "Matched original-input control"),
                                  (comparison, "#D3633E", "Validation-selected recipe" if comparison == choice else "Joint/daily variant (not selected)")):
            part = frame[(frame.station == station) & (frame.seed == 42) & (frame.model_name == arm)].set_index("month")
            curve = np.full(48, np.nan)
            indices = np.intersect1d(part.index, np.arange(start, end))
            curve[indices-start] = np.log1p(part.loc[indices, "y_pred"])
            ax.plot(times, curve, marker=".", ms=4, lw=1.2, color=color, label=label)
        ax.set_title(f"Station {data['site_no'][station]}: densest observed 48-month window", fontsize=10)
        ax.set_ylabel("log1p(DOC in mg/L)")
        ax.grid(alpha=.18)
        selections.append({"station_index": int(station), "site_no": str(data["site_no"][station]),
                           "window_start": str(times[0].date()), "window_end": str(times[-1].date()),
                           "observed_points": len(visible)})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=8, ncol=3)
    fig.savefig(out / "observed_temporal_examples.png", dpi=180)
    fig.savefig(out / "observed_temporal_examples.pdf")
    plt.close(fig)
    return {"selection_rule": "Four most densely observed spatial-query stations; earliest densest 48-month window per station, selected using label availability only. Seed 42 is illustrative, not a seed ensemble. Lines break across unobserved months.",
            "comparison_arm": comparison, "stations": selections}


def main():
    assert (ROOT / "batch_complete.json").exists(), "Training/scoring must finish first"
    audit = json.loads((ROOT / "analysis/independent_audit.json").read_text())
    assert audit["passed"], "Independent checks must pass first"
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert sha256_file(protocol["daily"]) == protocol["daily_sha256"]
    assert sha256_file("data/processed/graph_nodes_graphfix_st357.csv") == protocol["nodes_sha256"]
    assert sha256_file(ROOT / "preparation.json") == protocol["preparation_sha256"]
    for name, expected in protocol["mask_sha256"].items():
        assert sha256_file(ROOT / "masks" / f"{name}.npz") == expected, name
    choice = json.loads((ROOT / "selected_recipe.json").read_text())
    amendment = json.loads((ROOT / "resource_amendment.json").read_text())
    gate = json.loads((ROOT / "all_selections.json").read_text())
    assert not amendment["scientific_settings_changed"] and not amendment["target_scoring_started"]
    assert gate["resource_amendment_sha256"] == sha256_file(ROOT / "resource_amendment.json")
    assert amendment["script_sha256"] == sha256_file("scripts/resume_doc_dynamic_resources_v1.py")
    assert amendment["test_sha256"] == sha256_file("tests/test_doc_dynamic_resources_v1.py")
    summary = pd.read_csv(ROOT / "analysis/summary.csv")
    spatial = pd.concat([pd.read_parquet(ROOT / "fits" / t["id"] / "predictions.parquet")
                         for t in protocol["tasks"] if t["track"] == "space"], ignore_index=True)
    out = ROOT / "analysis"
    stats, intervals = spatial_bootstrap(spatial, summary)
    stats.to_csv(out / "spatial_sufficient_statistics.csv", index=False)
    intervals.to_csv(out / "spatial_paired_bootstrap.csv", index=False)
    station, station_summary = station_diagnostics(spatial)
    station.to_csv(out / "station_temporal_diagnostics.csv", index=False)
    station_summary.to_csv(out / "station_temporal_summary.csv", index=False)
    examples = plot_curves(spatial, choice["selected_arm"], out)
    (out / "temporal_example_selection.json").write_text(json.dumps(examples, indent=2)+"\n")
    content = ["# RiverDOCNet dynamic reconstruction development results", "",
               "The environment MLP, twelve-month GRU and upstream/downstream/global graph Transformer are retained. Four neural recipes are matched at 12,853 parameters. The original-input control is retrained under the new matched batching and runtime, so historical paper scores are not a causal control for this experiment.", "",
               "A separately frozen execution amendment increased independent fit concurrency from four to eight with at most four simultaneous neural fits. Every task remains single-threaded; no scientific setting, seed, candidate, mask or selection rule changed. Existing running workers were adopted rather than restarted. The original protocol and prediction files were retained. See resource_amendment.json.", "",
               f"The source-validation rule selected **{NAMES[choice['selected_arm']]}** before any target scoring. Native MAE and log-MAE remain primary. Temporal diagnostics do not change model selection.", "",
               "## Spatial queries", "", markdown_table(summary[summary.track == "space"], list(METRICS)), "",
               "Each seed pools the same five held HUC4 regions (104 stations, 7,262 observed-query cells); reported metrics average three seeds, without prediction ensembling.", ""]
    for gap in (3, 6, 12):
        content += [f"## {gap}-month temporal gaps", "", markdown_table(summary[(summary.track == "time") & (summary.gap_months == gap)], list(METRICS)), ""]
    content += ["## Paired uncertainty and limits", "",
                "spatial_paired_bootstrap.csv reports recipe-minus-control differences using 10,000 paired resamples of the five HUC4 regions, with seeds retained and pooled-cell metrics recomputed. Five clusters give limited independent replication. These exploratory intervals are not adjusted for multiple comparisons. Time-gap repetitions are overlapping development masks, so their spread is not an independent basin confidence interval.", "",
                "Station-centered R-squared removes each station's observed and predicted evaluation-period means separately and assesses fluctuations only. It is a diagnostic, not ordinary predictive R-squared, and uses evaluation labels only for scoring. SD ratios closer to one alone do not establish skill; delta error and concentration errors must also improve.", "",
                "Station-balanced exploratory summaries are also reported in station_temporal_summary.csv: each eligible station has at least twelve query observations and positive observed log variance; per-station log-centered scores are summarized for each seed and then averaged across seeds. Constant predictions remain in centered-score and variance summaries; undefined correlations are reported separately. This checks whether pooled fluctuation scores are dominated by a few stations. It is not an additional primary endpoint or a selection criterion.", "",
                "Temporal queries retain the parent protocol's fixed three-month scored cores while surrounding gaps increase. Results do not score every missing month in a twelve-month gap. All queries have observed DOC truth; naturally missing cells and arbitrary virtual river locations remain unvalidated.", "",
                "The eight additional descriptors enrich within-month daily discharge variability on the existing monitoring footprint. They do not supply missing hydrology, meteorology or independent physical states. The target is the mean of available DOC samples in a calendar month. Raw-record support and rare high concentrations are audited without deleting frozen labels. Sampling support alone does not establish measurement error.", "",
                "The protocol reuses previously inspected Mississippi development regions and masks. It is an optimization study, not an independent confirmation or a cross-river transfer claim.", "",
                "## Illustrative observed curves", "", "![Temporal examples](../experiments/phase4_transfer/doc_dynamic_reconstruction_v1/analysis/observed_temporal_examples.png)", "",
                examples["selection_rule"], "",
                "Independent verification replays every selected checkpoint on both validation and query cells and confirms frozen labels, identical queries, source-only selection and metric recomputation. See analysis/independent_audit.json."]
    report = Path("docs/doc_dynamic_reconstruction_results_20261012.md")
    report.write_text("\n".join(content)+"\n")
    outputs = [out / "spatial_sufficient_statistics.csv", out / "spatial_paired_bootstrap.csv",
               out / "station_temporal_diagnostics.csv", out / "station_temporal_summary.csv",
               out / "temporal_example_selection.json", out / "observed_temporal_examples.png",
               out / "observed_temporal_examples.pdf", report]
    provenance = {"script_sha256": sha256_file(__file__), "protocol_sha256": sha256_file(ROOT / "protocol.json"),
                  "summary_sha256": sha256_file(out / "summary.csv"), "audit_sha256": sha256_file(out / "independent_audit.json"),
                  "selected_recipe_sha256": sha256_file(ROOT / "selected_recipe.json"),
                  "resource_amendment_sha256": sha256_file(ROOT / "resource_amendment.json"),
                  "outputs": {str(p): sha256_file(p) for p in outputs},
                  "training_and_selection_modified": False}
    (out / "report_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"report": str(report), "selected_arm": choice["selected_arm"], "independent_audit_passed": True}))


if __name__ == "__main__":
    main()
