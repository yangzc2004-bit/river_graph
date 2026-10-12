"""Combine completed DOC comparators with matched-query and arithmetic checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

NETWORK = Path("experiments/phase4_transfer/doc_network_work_v2")
INTERPOLATION = Path("experiments/phase4_transfer/doc_interpolation_v2")
OUT = Path("experiments/phase4_transfer/doc_model_comparison_v1")
METRICS = ("mae", "log1p_mae", "rmse", "r2")
METHODS = {
    "fixed_fusion": ("Proposed", "Environmental MLP + GRU + graph Transformer"),
    "doc_xgboost": ("Reference", "XGBoost with 12-month history"),
    "ssn_covariance_adapt": ("Reference", "SSN covariance adaptation"),
    "doc_hydro_gam": ("Reference", "Hydrological/environmental GAM adaptation"),
    "rf_history_control": ("Reference/control", "RF with 12-month history"),
    "doc_ecological_rf": ("Reference", "Environmental/current-driver RF adaptation"),
    "physt_architecture_adapt": ("Reference", "PhyST-HC official architecture adaptation"),
    "spatiotemporal_idw": ("Traditional", "Bounded space-time IDW"),
    "ordinary_kriging": ("Traditional", "Exponential ordinary kriging"),
    "euclidean_idw": ("Traditional", "Same-month Euclidean IDW"),
    "spatiotemporal_kriging": ("Traditional", "Exponential space-time ordinary kriging"),
    "river_idw": ("Traditional", "Same-month NHD river-distance IDW"),
    "seasonal_median": ("Traditional", "Station/source seasonal median"),
    "nearest_spatial": ("Traditional", "Same-month spatial nearest neighbour"),
    "linear_time": ("Traditional", "Within-station linear time interpolation"),
    "pchip_time": ("Traditional", "Within-station PCHIP time interpolation"),
}


def calculate(frame):
    y = frame.y_true.to_numpy(dtype=float)
    p = frame.y_pred.to_numpy(dtype=float)
    assert np.isfinite(p).all()
    e = p-y
    return {"mae": np.abs(e).mean(),
            "log1p_mae": np.abs(np.log1p(p)-np.log1p(y)).mean(),
            "rmse": np.sqrt(np.square(e).mean()),
            "r2": 1-np.square(e).sum()/np.square(y-y.mean()).sum()}


def main():
    protocols = [json.loads((root/"protocol.json").read_text()) for root in (NETWORK, INTERPOLATION)]
    for field in ("dataset_sha256", "nodes_sha256", "seeds", "regions"):
        assert protocols[0][field] == protocols[1][field], field
    endpoint_aliases = {"native_mae": "mae", "mae": "mae", "log1p_mae": "log1p_mae"}
    endpoints = [[endpoint_aliases[name] for name in protocol["joint_primary"]] for protocol in protocols]
    assert endpoints[0] == endpoints[1] == ["mae", "log1p_mae"]
    for region in protocols[0]["regions"]:
        assert protocols[0]["mask_sha256"][region] == protocols[1]["mask_sha256"][f"space_{region}"]
    for root in (NETWORK, INTERPOLATION):
        assert json.loads((root/"analysis/independent_audit.json").read_text())["status"] == "passed"
    related = pd.read_csv(NETWORK/"analysis/summary.csv").set_index("model_name")
    interpolation = pd.read_csv(INTERPOLATION/"analysis/summary.csv")
    spatial = interpolation[interpolation.track == "space"].set_index("model_name")
    temporal = interpolation[interpolation.track == "time"]
    maximum = 0.
    source_hashes = {}
    reference_keys = None
    fusions = []
    comparisons = 0
    for root, folder, filename, summary in (
        (NETWORK, "regions/huc4_", "predictions.parquet", related),
        (INTERPOLATION, "tasks/space_", "predictions.parquet", spatial),
    ):
        paths = [root/f"{folder}{region}"/filename for region in protocols[0]["regions"]]
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        for path in paths:
            source_hashes[str(path)] = sha256_file(path)
        assert set(frame.model_name) == set(summary.index)
        rows = []
        for (model, seed), group in frame.groupby(["model_name", "seed"]):
            group = group.sort_values("cell")
            assert len(group) == 7262 and not group.cell.duplicated().any()
            assert group.station.nunique() == 104
            keys = group[["cell", "y_true"]].to_numpy()
            if reference_keys is None:
                reference_keys = keys
            np.testing.assert_array_equal(keys, reference_keys)
            result = calculate(group)
            rows.append({"model_name": model, "seed": seed, **result})
            if model == "fixed_fusion":
                fusions.append((root.name, seed, group.y_pred.to_numpy()))
            comparisons += 1
        independent = pd.DataFrame(rows).groupby("model_name")[list(METRICS)].mean()
        for model in summary.index:
            for metric in METRICS:
                delta = abs(independent.loc[model, metric]-summary.loc[model, metric])
                maximum = max(maximum, delta)
                assert delta < 1e-10
    for seed in protocols[0]["seeds"]:
        copies = [values for _, s, values in fusions if s == seed]
        assert len(copies) == 2
        np.testing.assert_array_equal(copies[0], copies[1])

    rows = []
    for model, (category, label) in METHODS.items():
        row = {"category": category, "model_name": model, "method": label}
        if model in spatial.index:
            selected = spatial.loc[model]
            row["intrinsic_spatial_coverage"] = selected.intrinsic_coverage
        elif model in related.index:
            selected = related.loc[model]
            row["intrinsic_spatial_coverage"] = selected.coverage
        else:
            selected = None
            row["intrinsic_spatial_coverage"] = np.nan
        for metric in METRICS:
            row[f"space_{metric}"] = selected[metric] if selected is not None else np.nan
        for gap in (3, 6, 12):
            values = temporal[(temporal.model_name == model) & (temporal.gap_months == gap)]
            assert len(values) <= 1
            row[f"time_mae_{gap}"] = values.iloc[0].mae if len(values) else np.nan
        rows.append(row)
    combined = pd.DataFrame(rows)
    assert len(combined) == 16 and combined.model_name.nunique() == 16
    scored = [f"space_{metric}" for metric in METRICS] + [f"time_mae_{gap}" for gap in (3, 6, 12)]
    best = {field: combined[field].max() if field == "space_r2" else combined[field].min() for field in scored}

    def fmt(value, field):
        if pd.isna(value):
            return "—"
        text = f"{value:.4f}"
        return f"**{text}**" if value == best[field] else text

    lines = ["| Category | Method | Spatial MAE ↓ | Spatial log1p MAE ↓ | Spatial RMSE ↓ | Spatial R² ↑ | Intrinsic spatial coverage | Time MAE:3 / 6 / 12 months ↓ |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = [row["category"], row["method"]]
        values += [fmt(row[f"space_{metric}"], f"space_{metric}") for metric in METRICS]
        cover = row["intrinsic_spatial_coverage"]
        values.append("—" if pd.isna(cover) else f"{cover*100:.2f}%")
        values.append(" / ".join(fmt(row[f"time_mae_{gap}"], f"time_mae_{gap}") for gap in (3, 6, 12)))
        lines.append("| " + " | ".join(values) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT/"summary.csv", index=False)
    for path in [NETWORK/"analysis/summary.csv", INTERPOLATION/"analysis/summary.csv"]:
        source_hashes[str(path)] = sha256_file(path)
    audit = {"status": "passed", "spatial_population": {"queries": 7262, "stations": 104, "regions": 5},
             "matched_spatial_model_seed_groups": comparisons, "unique_spatial_models": 14,
             "unique_combined_models": 16, "maximum_independent_metric_difference": maximum,
             "identical_fusion_prediction_copies": True, "source_hashes": source_hashes,
             "script_sha256": sha256_file(Path(__file__)), "summary_sha256": sha256_file(OUT/"summary.csv")}
    (OUT/"aggregation_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    report = """# Consolidated DOC model comparison

Date: 2026-10-11. The environmental MLP + GRU + graph Transformer has the lowest spatial native MAE and log1p MAE point estimates among the tested methods. The hydrological/environmental GAM has the lowest spatial RMSE and highest predictive R². These are fixed-budget internal development results, not global SOTA or universal endpoint superiority.

The spatial columns use identical 7,262 observed queries at 104 whole-DOC-held-out stations in five Mississippi HUC4 regions. Native errors are mg C/L; R² is 1−SSE/SST. Scores are computed per seed then averaged, not from an ensemble. The proposed model's duplicate predictions were checked for exact equality across the two parent comparisons.

The time column separately uses 266 eligible stations and three masking seeds, with the same scored cores across 3/6/12-month gaps within each seed. It reports native MAE only; it must not be ranked against spatial scores. A dash denotes unavailable or inapplicable results: reference models have not run this temporal protocol; within-station time interpolation cannot support a station with no DOC. All methods from both rounds are included exactly once. Bold marks the best full-precision point estimate in each scored column; coverage is descriptive.

""" + "\n".join(lines) + """

Every spatial score includes 100% final prediction coverage. Unsupported interpolation outputs use the declared training-only station/seasonal/source median fallback. Intrinsic coverage describes direct output before fallback. The independent common-covered-subset comparison in the interpolation report retained the fusion versus traditional spatial ranking.

The spatial MAE point estimate is 27.40% lower than bounded space-time IDW, 3.05% lower than historical XGBoost, and 6.13% lower than SSN. Existing exploratory five-basin confidence intervals for native-MAE differences against XGBoost, SSN and GAM include zero; they are unadjusted and use previously examined development regions. The small numerical advantages do not establish population significance. No new uncertainty estimate is inferred from this consolidation.

Temporal native MAE is lowest for fusion at all three tested gap lengths, but the advantage is only 0.48% over linear interpolation at three months and 1.29% over seasonal median at twelve months. At three months linear interpolation has lower log1p MAE/RMSE and higher R²; at twelve months seasonal median has lower log1p MAE and bounded space-time IDW has lower RMSE/higher R². Therefore temporal joint-endpoint superiority is not established.

SSN, GAM, ecological RF and PhyST-HC rows retain their adaptation labels. PhyST-HC executes the author's architecture with our inputs/target/training and is not a complete reproduction of the original process-state workflow. Input organization and budgets differ between families; this comparison alone cannot attribute gains to additional input sources or architectural components. Matched-input ablations and external-river evaluation address those separate claims. INCA-C is excluded because only its public native example, not this matched Mississippi comparison, has been executed.

Sources: [related-model comparison](doc_network_work_results_20261011.md), [traditional comparison](doc_interpolation_results_20261011.md), [combined metrics](../experiments/phase4_transfer/doc_model_comparison_v1/summary.csv), [matched-query aggregation checks](../experiments/phase4_transfer/doc_model_comparison_v1/aggregation_audit.json). The combined checks independently recomputed all four spatial metrics for both parent comparisons, reconciled matching masks/dataset/seeds, and verified identical query/truth membership for every spatial model and seed.
"""
    destination = Path("docs/doc_model_comparison_20261011.md")
    destination.write_text(report)
    print(json.dumps({"report": str(destination), **{key: audit[key] for key in (
        "status", "unique_spatial_models", "unique_combined_models", "maximum_independent_metric_difference")}}))


if __name__ == "__main__":
    main()
