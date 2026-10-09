"""Report source-validation-only K0 reconstruction and donor mechanism diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_source_retrieval_v1 import ROOT


def load_panel(root):
    frames, thresholds = [], {}
    for completion in sorted((root / "runs").glob("*/complete.json")):
        run = completion.parent
        config = json.loads((run / "config.json").read_text())
        frame = pd.read_parquet(run / "predictions.parquet")
        if not frame.visibility_role.eq("val").all() or set(frame.model_name) != set(config["models"]):
            raise ValueError("analysis requires the complete source-validation model panel")
        frames.append(frame)
        thresholds[(config["split_seed"], config["seed"])] = config["q90_threshold_train"]
    if not frames:
        raise ValueError("no completed source-validation runs")
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["split_seed", "seed", "model_name", "cell"]).any():
        raise ValueError("duplicate query cells")
    return panel, thresholds


def paired(panel, candidate, reference):
    columns = ["split_seed", "seed", "station", "cell"]
    a = panel[panel.model_name.eq(candidate)][[*columns, "y_true", "y_pred"]]
    b = panel[panel.model_name.eq(reference)][[*columns, "y_true", "y_pred"]]
    pair = a.merge(b, on=columns, validate="one_to_one", suffixes=("_candidate", "_reference"))
    if len(pair) != len(a) or len(pair) != len(b):
        raise ValueError("comparison loses query rows")
    np.testing.assert_array_equal(pair.y_true_candidate, pair.y_true_reference)
    pair["candidate_error"] = np.abs(pair.y_pred_candidate-pair.y_true_candidate)
    pair["reference_error"] = np.abs(pair.y_pred_reference-pair.y_true_reference)
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    panel, thresholds = load_panel(args.root)
    runs, parts, curves = metric_summary(panel, thresholds)
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "run_metrics.csv", index=False)
    parts.to_csv(output / "partition_metrics.csv", index=False)
    curves.to_csv(output / "summary.csv", index=False)
    contrasts = [(arm, reference) for arm in ("retrieval", "hydro_pretrained", "hydro_pretrained_retrieval")
                 for reference in ("current_model", "matched_daily_trees", "static_memory")]
    contrasts += [(arm, f"{arm}_{ablation}") for arm in ("retrieval", "hydro_pretrained_retrieval")
                  for ablation in ("uniform", "zero_source_values")]
    contrasts.append(("hydro_pretrained_retrieval", "retrieval"))
    results, station_rows = [], []
    for candidate, reference in contrasts:
        pair = paired(panel, candidate, reference)
        directions = pair.groupby(["split_seed", "seed"]).agg(a=("candidate_error", "mean"), b=("reference_error", "mean"))
        part_direction = directions.groupby("split_seed").mean()
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                q = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= q]
            result = {"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((directions.a < directions.b).sum()),
                "improved_partitions": int((part_direction.a < part_direction.b).sum()),
                "evaluation_role": "source_validation_selected_not_independent_confirmation"}
            results.append(result)
        station = pair.groupby(["split_seed", "station"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"), n_rows=("cell", "nunique"))
        station["candidate"], station["reference"] = candidate, reference
        station["delta_mae"] = station.candidate_mae-station.reference_mae
        station_rows.append(station)
    effects = pd.DataFrame(results)
    effects.to_csv(output / "paired_effects.csv", index=False)
    pd.concat(station_rows, ignore_index=True).to_csv(output / "station_effects.csv", index=False)
    records = []
    for (_, _, model), group in panel.groupby(["split_seed", "seed", "model_name"]):
        if "retrieval_entropy" in group and group.retrieval_entropy.notna().any():
            records.append({"model_name": model, "entropy": group.retrieval_entropy.mean(),
                            "effective_donors": group.effective_donors.mean(), "max_weight": group.max_donor_weight.mean()})
    pd.DataFrame(records).to_csv(output / "retrieval_diagnostics.csv", index=False)
    lines = ["# DOC source-retrieval development", "", "These results use selected source-validation stations only.",
             "No target DOC evaluation or geographical/external confirmation has been performed.", "",
             "## All-observation K0 MAE", "", "| Model | MAE mg/L | Q90 MAE |", "|---|---:|---:|"]
    for row in curves.itertuples():
        lines.append(f"| {row.model_name} | {row.mae:.6f} | {row.q90_mae:.6f} |")
    lines += ["", "## Paired effects", "", "| Candidate vs reference | MAE gain % [95% CI] |",
              "|---|---:|"]
    for row in effects[effects.region.eq("overall")].itertuples():
        lines.append(f"| {row.candidate} vs {row.reference} | {row.relative_gain_pct:.3f} [{row.gain_ci_low_pct:.3f}, {row.gain_ci_high_pct:.3f}] |")
    lines += ["", "Intervals describe the selected development panel; selection optimism remains.",
              "Geographical confirmation is required before spatial-performance claims."]
    (output / "findings.md").write_text("\n".join(lines)+"\n")
    print(curves[["model_name", "mae", "q90_mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
