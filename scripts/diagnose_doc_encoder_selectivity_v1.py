"""Diagnose encoder selectivity using source training and fixed validation only.

The Parquet reader selects only fixed validation query cells. Target-test
labels, target-test predictions, and prior endpoint analyses are never read.
This script performs no model fitting or parameter selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule

ARMS = ("frozen", "last_self", "last_self_ecology")
ROOTS = (Path("experiments/phase4_transfer/doc_encoder_residual_v1"),
         Path("experiments/phase4_transfer/doc_encoder_budget_v1"))
OUTPUT = Path("experiments/phase4_transfer/doc_selective_residual_v1/diagnostics")
MODELS = {"context": "context_pred", **{arm: f"{arm}_pred" for arm in ARMS},
          **{f"{arm}_integrated": f"{arm}_integrated_k0_pred" for arm in ARMS}}
METRICS = ("mae", "tail_mae", "ordinary_mae", "bias", "tail_bias", "ordinary_bias",
           "ordinary_positive_error", "ordinary_negative_error", "false_q90_rate",
           "tail_recall", "tail_precision", "station_equal_mae", "tail_absolute_error_share")


def selectivity_metrics(truth, prediction, stations, threshold):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    stations = np.asarray(stations)
    if (truth.ndim != 1 or prediction.shape != truth.shape or stations.shape != truth.shape
            or not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
            or (truth < 0).any() or (prediction < 0).any() or not np.isfinite(threshold)):
        raise ValueError("selected validation truth/predictions must be finite aligned native vectors")
    error = prediction - truth
    tail = truth >= threshold
    predicted_tail = prediction >= threshold
    ordinary = ~tail
    false = predicted_tail & ordinary

    def mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    total_error = np.abs(error).sum()
    return {
        "n_query": len(truth), "n_query_stations": len(np.unique(stations)),
        "n_tail": int(tail.sum()), "n_ordinary": int(ordinary.sum()),
        "n_false_q90": int(false.sum()), "tail_unstable": int(tail.sum()) < 20,
        "mae": mean(np.abs(error)), "tail_mae": mean(np.abs(error[tail])),
        "ordinary_mae": mean(np.abs(error[ordinary])), "bias": mean(error),
        "tail_bias": mean(error[tail]), "ordinary_bias": mean(error[ordinary]),
        "ordinary_positive_error": mean(np.maximum(error[ordinary], 0)),
        "ordinary_negative_error": mean(np.maximum(-error[ordinary], 0)),
        "false_q90_rate": float(false.sum() / ordinary.sum()) if ordinary.any() else float("nan"),
        "tail_recall": mean(predicted_tail[tail]),
        "tail_precision": mean(tail[predicted_tail]),
        "station_equal_mae": float(pd.DataFrame({"station": stations, "error": np.abs(error)})
                                    .groupby("station").error.mean().mean()),
        "tail_absolute_error_share": float(np.abs(error[tail]).sum() / total_error)
        if total_error else 0.0,
    }


def selected_labels(dataset, cells):
    """Index labels before converting to NumPy; never materialize other roles."""
    return np.asarray(dataset["y"].reshape(-1)[cells], dtype=np.float64)


def source_concentration(partition, cells, truth, threshold, months):
    stations, counts = np.unique(cells // months, return_counts=True)
    tail_counts = np.array([np.sum(truth[cells // months == station] >= threshold)
                            for station in stations])
    weight_two = counts + tail_counts
    order = np.argsort(counts)[::-1]
    n_top = int(np.ceil(.1 * len(stations)))
    shares, weighted_shares = counts / counts.sum(), weight_two / weight_two.sum()
    row = {
        "partition": partition, "n_source_stations": len(stations), "n_source_cells": len(cells),
        "source_q90": threshold, "n_source_tail": int(tail_counts.sum()),
        "station_count_min": int(counts.min()), "station_count_median": float(np.median(counts)),
        "station_count_max": int(counts.max()), "top_decile_station_count": n_top,
        "top_decile_cell_share": float(shares[order[:n_top]].sum()),
        "largest_station_cell_share": float(shares.max()),
        "effective_source_stations_cell_loss": float(1 / np.square(shares).sum()),
        "effective_source_stations_tail_weight_two": float(1 / np.square(weighted_shares).sum()),
        "n_source_stations_with_tail": int(np.count_nonzero(tail_counts)),
    }
    detail = pd.DataFrame({"partition": partition, "station_index": stations,
                           "n_source_cells": counts, "n_source_tail": tail_counts,
                           "cell_loss_station_share": shares,
                           "tail_weight_two_station_share": weighted_shares})
    return row, detail


def write_report(output, by_run, partitions, aggregate, concentration):
    def table(frame, columns):
        head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
        lines = []
        for row in frame[columns].to_dict("records"):
            lines.append("| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float)
                                            else str(row[k]) for k in columns) + " |")
        return head + "\n" + "\n".join(lines)

    current = aggregate[aggregate.budget == 60]
    old = aggregate[aggregate.budget == 30]
    current_table = table(current, ["model", "mae", "tail_mae", "ordinary_mae", "tail_bias",
                                   "ordinary_bias", "false_q90_rate", "station_equal_mae"])
    old_table = table(old, ["model", "mae", "tail_mae", "ordinary_mae", "ordinary_bias", "false_q90_rate"])
    source_table = table(concentration, ["partition", "n_source_stations", "n_source_cells",
                                         "station_count_median", "station_count_max",
                                         "top_decile_cell_share", "effective_source_stations_cell_loss",
                                         "effective_source_stations_tail_weight_two"])
    report = f"""# Source-validation selectivity: native DOC encoder residual

## Scope and definitions

This diagnostic uses the existing 30- and 60-epoch encoder products only at the
fixed source-validation query cells. The five scheduled support months remain
reserved at K=0. Parquet reads have an explicit `cell in validation_query` filter;
the selected rows are checked against the query vector and the `val` role.
Dataset labels are indexed only at source-training cells and validation queries.
No target-test predictions, labels or endpoint analyses were read; no model was
fitted. Dataset/mask hashes and saved validation MAEs are checked.

Q90 is the frozen source-training quantile, with a tail defined by truth >= Q90.
Signed bias is prediction minus truth. False-Q90 rate is false high predictions
divided by the number of ordinary query cells (truth < Q90). Ordinary positive
error is mean(max(prediction - truth, 0)) over **all** ordinary cells, not only
overpredictions. Ordinary MAE equals positive plus negative error mass.

Scores pool query cells within each run, average three seeds within partition,
then give each partition equal weight. Station-equal MAE additionally averages
station MAEs within each run. Seeds repeat the same ecological observations;
counts in the run CSV are not summed across seeds. Integrated models use the
already selected K0 ecological mixture, without fitting new mixture weights.

## Current 60-epoch panel

{current_table}

All errors and biases are in mg/L; false-Q90 rates are proportions.

Relative to context, self-plus-ecology lowers ordinary MAE from 1.290994 to
1.204062, ordinary bias from +0.362010 to +0.146393, and ordinary positive-error
mass from 0.826502 to 0.675227. Tail MAE falls from 9.984876 to 9.634572 and tail
bias improves from -9.462878 to -8.841819. False-Q90 rate nonetheless rises from
0.009296 to 0.014050. The model therefore reduces ordinary overprediction in
aggregate while making more errors around the high-value decision threshold.
Those are different properties and should not be conflated.

## Earlier 30-epoch panel

{old_table}

Longer training improves both ordinary and tail MAE in the equal-partition
summary. For the ecology arm, ordinary positive-error mass falls from 0.709574
to 0.675227 and ordinary bias falls from +0.192333 to +0.146393, while false-Q90
rate rises modestly from 0.013186 to 0.014050. Thus the duration benefit is not a
uniform upward shift of all predictions.

## Partition differences

- **Partition 142:** extra ecology tuning versus frozen improves tail MAE
  (6.054286 to 5.970752), but slightly worsens ordinary MAE (1.214397 to
  1.219168), ordinary bias (+0.087630 to +0.126039), and false-Q90 rate
  (0.016003 to 0.020882). This is the clearest local sensitivity–specificity
  tradeoff.
- **Partition 143:** ecology tuning improves both ordinary and tail MAE over
  frozen. Ordinary positive-error mass falls (0.986841 to 0.918200), although
  false-Q90 rate rises slightly (0.017795 to 0.019084). Ecological integration
  further reduces ordinary error while sacrificing some tail accuracy.
- **Partition 144:** all residual arms improve ordinary MAE over context, and
  ordinary signed bias is already negative. Tail MAE remains about 15.65 mg/L.
  Adding a universal ordinary-overprediction penalty would not directly address
  this partition's dominant high-value underprediction.

Full per-partition metrics and counts are saved separately. These validation
labels already selected checkpoints and mixture strengths, so the decomposition
supports the next development experiment rather than a new confirmatory claim.

## Source-station sampling concentration

{source_table}

The top decile rounds upward to 24 of 232 stations. Those stations contribute
38–40% of training cells; median station counts are only 36–37. Cell-weighted
loss therefore gives substantially greater influence to densely sampled source
stations. The effective station count here is the inverse squared sum of each
station's objective-weight share, not an inferential sample size.

Tail weight two increases the effective station count modestly, rather than
making station concentration worse. It should not be blamed for this sampling
imbalance without a matched loss comparison. Station-equal validation MAE is
also appreciably larger than pooled MAE, which shows that station-level errors
are distributed unevenly; it does not alone establish that reweighting will help.

## Recommended next experiment

**Test source-station-balanced training against the current cell-weighted loss,
keeping tail weight two, encoder mode, 60-epoch cap, patience and learning rates
fixed.** Continue choosing checkpoint and scale by the same pooled validation
MAE, and retain station-equal/tail/ordinary diagnostics as explanatory outputs.
A direct implementation gives each source station equal total loss weight,
with the existing tail weights normalized within station; its objective is the
mean of station-wise tail-weighted MAEs. Use a fixed whole-source normalization
so minibatch composition does not silently change the objective.

This tests a concrete mismatch between uneven training frequency and prediction
at new stations. It is not a claim that balancing must win under the existing
pooled endpoint.

The alternatives are less directly supported by these source-validation data:

1. **Extra ordinary-overprediction pressure:** useful as a focused follow-up
   if a remaining false-high cost matters, but broad overprediction already
   decreases and partition 144 ordinary predictions are biased downward.
2. **Tail weight one:** a clean future ablation of sensitivity against false
   positives, but current tail bias remains strongly negative and source
   station diversity slightly increases under weight two. Removing the tail
   emphasis is not the clearest first response to the observed imbalance.

No new model or training is launched by this diagnostic.
"""
    (output / "source_validation_selectivity_diagnostic.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs=2, default=ROOTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, hashes, mask_hashes = {}, {}, {}
    rows, concentration, counts, selected_sources = [], [], [], []
    seen_source = {}
    for root in args.roots:
        for partition in (142, 143, 144):
            for seed in (42, 43, 44):
                run = root / "runs" / f"split{partition}_seed{seed}"
                config = json.loads((run / "config.json").read_text())
                data_path, mask_path = config["dataset_path"], config["mask_path"]
                if data_path not in datasets:
                    hashes[data_path] = sha256_file(data_path)
                    datasets[data_path] = torch.load(data_path, map_location="cpu", weights_only=False)
                if mask_path not in mask_hashes:
                    mask_hashes[mask_path] = sha256_file(mask_path)
                if hashes[data_path] != config["dataset_hash"] or mask_hashes[mask_path] != config["mask_hash"]:
                    raise ValueError("dataset or mask differs from the bound configuration")
                dataset = datasets[data_path]
                months = dataset["y"].shape[1]
                # Deliberately do not access the test role in the mask archive.
                with np.load(mask_path, allow_pickle=False) as archive:
                    train, validation = archive["train"].copy(), archive["val"].copy()
                _, query = support_schedule(validation, months)
                truth, source_truth = selected_labels(dataset, query), selected_labels(dataset, train)
                threshold = float(config["q90_threshold_train"])
                # Saved training quantiles were computed on float32 labels;
                # the diagnostic promotes selected values to float64 first.
                if not np.isclose(np.quantile(source_truth, .9), threshold, rtol=0, atol=1e-6):
                    raise ValueError("source-only Q90 disagrees with the saved threshold")
                columns = ["cell", "visibility_role", *MODELS.values()]
                frame = pq.read_table(run / "full_grid.parquet", columns=columns,
                                      filters=[("cell", "in", query.tolist())]).to_pandas().sort_values("cell")
                if not np.array_equal(frame.cell.to_numpy(), query) or not frame.visibility_role.eq("val").all():
                    raise ValueError("Parquet rows must exactly match the fixed validation query")
                selected_sources.append({"run": str(run), "prediction_rows_read": len(frame),
                                         "role": "source_validation_query", "budget": config["epochs"]})
                mixers = json.loads((run / "mixers.json").read_text())
                for model, column in MODELS.items():
                    metrics = selectivity_metrics(truth, frame[column].to_numpy(), query // months, threshold)
                    if model in ARMS:
                        saved = json.loads((run / f"{model}.json").read_text())
                        expected = saved["validation_metrics"]["validation_mae"]
                    elif model.endswith("_integrated"):
                        expected = mixers[f"{model}_constant"]["selection_by_k"]["0"]["mae"]
                    else:
                        expected = metrics["mae"]
                    if not np.isclose(metrics["mae"], expected, rtol=0, atol=1e-12):
                        raise ValueError("recomputed validation MAE differs from saved model selection")
                    rows.append({"budget": config["epochs"], "partition": partition, "seed": seed,
                                 "model": model, "q90_threshold_train": threshold, **metrics})
                current_source = (train.tolist(), source_truth.tolist(), threshold)
                if partition not in seen_source:
                    seen_source[partition] = current_source
                    summary, detail = source_concentration(partition, train, source_truth, threshold, months)
                    concentration.append(summary)
                    counts.append(detail)
                elif current_source != seen_source[partition]:
                    raise ValueError("source labels or cells differ across matched run settings")
    by_run = pd.DataFrame(rows)
    partitions = by_run.groupby(["budget", "partition", "model"], as_index=False)[list(METRICS)].mean()
    aggregate = partitions.groupby(["budget", "model"], as_index=False)[list(METRICS)].mean()
    concentration = pd.DataFrame(concentration)
    by_run.to_csv(args.output / "selectivity_by_run.csv", index=False)
    partitions.to_csv(args.output / "selectivity_by_partition.csv", index=False)
    aggregate.to_csv(args.output / "selectivity_summary.csv", index=False)
    concentration.to_csv(args.output / "source_count_concentration.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(args.output / "source_station_counts.csv", index=False)
    (args.output / "read_scope.json").write_text(json.dumps({
        "prediction_filter": "fixed validation queries only; five support months reserved",
        "labels_indexed": ["train", "validation_query"], "outer_test_accessed": False,
        "n_model_run_rows": len(by_run), "sources": selected_sources,
    }, indent=2) + "\n")
    write_report(args.output, by_run, partitions, aggregate, concentration)
    print(aggregate[["budget", "model", "mae", "tail_mae", "ordinary_mae", "false_q90_rate"]].to_string(index=False))
    print(f"Saved source-validation diagnostic to {args.output}")


if __name__ == "__main__":
    main()
