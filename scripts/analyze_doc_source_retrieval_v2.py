"""Verify donor-contrast replay and report development-only K0 comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_source_retrieval_v2 import ROOT, predict_contrast
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.doc_source_retrieval import SourceRetrievalAttention


def verify(run):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    meta = json.loads((run / "predictions.meta.json").read_text())
    if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run / "predictions.parquet")
            or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])):
        raise ValueError("changed product identity")
    table = pd.read_parquet(run / "predictions.parquet")
    with np.load(config["mask_path"], allow_pickle=False) as split:
        expected = np.sort(split["val"])
        for _, rows in table.groupby("model_name"):
            np.testing.assert_array_equal(np.sort(rows.cell), expected)
            assert not np.isin(rows.cell, split["test"]).any()
    model = SourceRetrievalAttention.from_payload(torch.load(run / "retrieval_contrast.pt", weights_only=False))
    selection = json.loads((run / "retrieval_contrast.json").read_text())
    with np.load(run / "validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    for ablation in (None, "uniform", "zero_source_values"):
        delta, *_ = predict_contrast(model, inputs, scale=selection["residual_scale"], ablation=ablation)
        name = "retrieval_contrast"+("_"+ablation if ablation else "")
        np.testing.assert_array_equal(np.maximum(0, inputs["base"]+delta),
                                      table[table.model_name.eq(name)].sort_values("cell").y_pred.to_numpy())
    return {"run": run.name, "identity": True, "bitwise_contrast_replay": True, "target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    runs = sorted((args.root / "runs").glob("*/complete.json"))
    replay = [verify(path.parent) for path in runs]
    write_json(args.root / "verification" / "replay.json", replay)
    panel, thresholds = load_panel(args.root)
    per_run, parts, summary = metric_summary(panel, thresholds)
    for metric in ("q90_mae", "r2", "log_r2"):
        means = parts.groupby(["model_name", "k"])[metric].mean()
        for index, row in summary.iterrows():
            summary.loc[index, metric] = means[(row.model_name, row.k)]
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    for filename, table in (("run_metrics", per_run), ("partition_metrics", parts), ("summary", summary)):
        table.to_csv(output / f"{filename}.csv", index=False)
    results = []
    references = ("current_model", "matched_daily_trees", "static_memory",
                  "retrieval_contrast_uniform", "retrieval_contrast_zero_source_values")
    for reference in references:
        pair = paired(panel, "retrieval_contrast", reference)
        directions = pair.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                limit = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= limit]
            results.append({"candidate": "retrieval_contrast", "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((directions.candidate_error < directions.reference_error).sum()),
                "improved_partitions": int((directions.groupby("split_seed").mean().candidate_error
                                           < directions.groupby("split_seed").mean().reference_error).sum()),
                "evaluation_role": "source_validation_selected_not_confirmation"})
    effects = pd.DataFrame(results)
    effects.to_csv(output / "paired_effects.csv", index=False)
    sources = {str(path.parent / "complete.json"): sha256_file(path.parent / "complete.json") for path in runs}
    write_json(output / "sources.json", {"runs": sources, "packages": len(runs), "bootstrap_draws": args.bootstrap_draws})
    print(summary[["model_name", "mae", "q90_mae"]].to_string(index=False))
    print(effects[effects.region.eq("overall")][["reference", "relative_gain_pct", "gain_ci_low_pct",
          "gain_ci_high_pct", "improved_partitions"]].to_string(index=False))


if __name__ == "__main__":
    main()
