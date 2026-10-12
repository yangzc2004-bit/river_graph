"""Replay ablation checkpoints and independently verify matching and metrics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import verify
from run_doc_information_ablation_v1 import (
    ARMS,
    BENCHMARK,
    DATASET,
    NODES,
    ROOT,
    SEEDS,
    cells_prediction,
    configuration,
    load_model,
    parent,
    task_context,
)
from run_doc_river_architecture_comparison_v1 import digest, role_labels, write_json
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import sha256_file


def independent_metrics(part):
    y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
    error = p-y
    sst = np.square(y-y.mean()).sum()
    return {"mae": float(np.abs(error).mean()),
        "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1-np.square(error).sum()/sst) if sst else np.nan,
        "bias": float(error.mean())}


def main():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    manifest = json.loads((ROOT/"manifest.json").read_text())
    batch = json.loads((ROOT/"batch_complete.json").read_text())
    assert manifest["protocol_hash"] == batch["protocol_hash"] == digest(protocol)
    assert batch["new_fits"] == 102 and batch["reused_fits"] == 39
    assert sha256_file(DATASET) == protocol["dataset_sha256"]
    assert sha256_file(NODES) == protocol["nodes_sha256"]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    for name, expected in protocol["runtime_snapshot"].items():
        assert sha256_file(ROOT/"code_snapshot"/name) == expected
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == expected
    for name, expected in protocol["dependencies"].items():
        assert sha256_file(name) == expected
    for name, expected in protocol["mask_sha256"].items():
        assert sha256_file(ROOT/"masks"/f"{name}.npz") == expected
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    truth = np.asarray(data["y"]).ravel()
    gate = json.loads((ROOT/"all_selections.json").read_text())
    assert gate["protocol_hash"] == digest(protocol) and gate["test_used"] is False
    assert len(gate["fit_receipts"]) == 141
    frames, maximum_replay, new_fits, reused_fits, parameters = [], 0., 0, 0, set()
    for task in protocol["tasks"]:
        directory = ROOT/"fits"/task["id"]
        config = configuration(protocol, task)
        assert sha256_file(directory/"selection_complete.json") == gate["fit_receipts"][task["id"]]
        verify(directory, "selection_complete.json", config)
        verify(directory, "test_complete.json", config)
        ctx = task_context(ROOT, task, data)
        assert json.loads((directory/"normalization.json").read_text()) == ctx["original"]["normalization"]
        if task["arm"] == "no_environment":
            assert not ctx["inputs"]["environment"].any()
        if task["arm"] == "no_time":
            assert not ctx["inputs"]["windows"].any() and not ctx["inputs"]["season"].any()
        if task["arm"] == "no_river":
            assert not ctx["inputs"]["environment"][:, :4].any()
            assert ctx["graphs"]["test"]["edge_count"] == 0
        if task["arm"] == "current_only":
            assert not ctx["inputs"]["windows"][:, :, :-1].any()
            assert not ctx["inputs"]["windows"][:, :, -1, 4:6].any()
        if task["arm"] == "no_relations":
            relation = ctx["graphs"]["test"]["relation"]
            assert torch.all(relation[~torch.eye(len(relation), dtype=torch.bool)] == 5)
        settings = protocol["settings"][task["track"]]
        torch.set_num_threads(settings["torch_threads"])
        train_y = role_labels(data, ctx["split"], "train", ctx["graphs"]["train"]["rows"].numpy())
        z = torch.log1p(train_y[torch.isfinite(train_y)])
        expected_norm = {"mean": float(z.mean()), "std": max(float(z.std()), .1)}
        with threadpool_limits(limits=settings["torch_threads"]):
            model, info = load_model(ROOT, task, protocol)
            assert info["target_normalization"] == expected_norm
            parameters.add(info["parameters"])
            choice = json.loads((directory/"selection.json").read_text())
            assert choice["test_used"] is False and choice["best_epoch"] == info["best_epoch"]
            if parent(task) is None:
                trace = json.loads((directory/"trace.json").read_text())
                running_best, selected = float("inf"), None
                for item in trace:
                    if item["validation_mae"] < running_best-1e-6:
                        running_best, selected = item["validation_mae"], item["epoch"]
                assert selected == info["best_epoch"]
                assert settings["min_epochs"] <= info["epochs_run"] <= settings["max_epochs"]
                new_fits += 1
            else:
                reused_fits += 1
            for role, filename in (("val", "validation_predictions.parquet"), ("test", "predictions.parquet")):
                saved = pd.read_parquet(directory/filename)
                cells = ctx["split"][role]
                np.testing.assert_array_equal(saved.cell, cells)
                np.testing.assert_array_equal(saved.y_true, truth[cells])
                assert np.isfinite(saved.y_pred).all() and (saved.y_pred >= 0).all()
                assert not saved.cell.duplicated().any()
                predicted = cells_prediction(model, ctx, cells, role, info["target_normalization"], settings["batch_months"])
                difference = float(np.abs(saved.y_pred.to_numpy()-predicted).max())
                maximum_replay = max(maximum_replay, difference)
                assert difference < 1e-6
                if role == "val":
                    for field, value in independent_metrics(saved).items():
                        assert abs(value-choice["validation_metrics"][field]) < 1e-10
                else:
                    frames.append(saved)
    assert new_fits == 102 and reused_fits == 39 and len(parameters) == 1
    joined = pd.concat(frames, ignore_index=True)
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    seed_table = pd.read_csv(ROOT/"analysis/by_seed.csv")
    maximum = 0.
    independent_rows = []
    for (track, gap, seed, arm), part in joined.groupby(["track", "gap_months", "seed", "model_name"]):
        assert not part.cell.duplicated().any()
        recorded = seed_table[(seed_table.track == track) & (seed_table.gap_months == gap) & (seed_table.seed == seed) & (seed_table.model_name == arm)].iloc[0]
        computed = independent_metrics(part)
        for field, value in computed.items():
            maximum = max(maximum, abs(value-recorded[field]))
        assert recorded.query_count == len(part) and recorded.coverage == 1.
        if track == "space":
            assert len(part) == 7262 and part.station.nunique() == 104
        independent_rows.append({"track": track, "gap_months": gap, "seed": seed, "model_name": arm, **computed})
    means = pd.DataFrame(independent_rows).groupby(["track", "gap_months", "model_name"])[["mae", "log1p_mae", "rmse", "r2", "bias"]].mean()
    for row in summary.itertuples():
        result = means.loc[(row.track, row.gap_months, row.model_name)]
        for field, value in result.items():
            maximum = max(maximum, abs(value-getattr(row, field)))
        baseline = means.loc[(row.track, row.gap_months, "fixed_fusion"), "mae"]
        maximum = max(maximum, abs(row.mae_increase-(result.mae-baseline)))
        assert abs(row.mae_increase_percent-100*(result.mae-baseline)/baseline) < 1e-10
    reference = pd.read_csv(BENCHMARK/"analysis/summary.csv").set_index("model_name").loc["fixed_fusion"]
    own = summary[(summary.track == "space") & (summary.model_name == "fixed_fusion")].iloc[0]
    for field in ("mae", "log1p_mae", "rmse", "r2"):
        assert abs(own[field]-reference[field]) < 1e-6
    intervals = json.loads((ROOT/"analysis/paired_comparisons.json").read_text())
    spatial = joined[joined.track == "space"]
    for item in intervals:
        endpoint = item["endpoint"]
        errors = []
        for arm in (item["comparator"], "fixed_fusion"):
            selected = spatial[spatial.model_name == arm].copy()
            y, p = selected.y_true.to_numpy(float), selected.y_pred.to_numpy(float)
            selected["error"] = np.abs(y-p) if endpoint == "mae" else np.abs(np.log1p(y)-np.log1p(p))
            groups = selected.groupby(["target_huc4", "cell"]).error.mean().groupby(level=0).agg(["sum", "count"])
            errors.append(groups)
        base, full = errors
        counts = base["count"].to_numpy()
        draw = np.random.default_rng(20261011).integers(0, 5, (5000, 5))
        values = (full["sum"].to_numpy()[draw].sum(1)-base["sum"].to_numpy()[draw].sum(1))/counts[draw].sum(1)
        np.testing.assert_allclose(np.quantile(values, [.025, .975]), item["difference_interval"], atol=1e-12)
    assert maximum < 1e-10
    verify(ROOT/"analysis", "complete.json", {"protocol_hash": digest(protocol)})
    result = {"status": "passed", "new_checkpoint_replays": new_fits, "reused_checkpoint_replays": reused_fits,
        "maximum_prediction_replay_difference": maximum_replay, "maximum_independent_metric_difference": maximum,
        "identical_parameter_count": next(iter(parameters)), "summary_rows": len(summary),
        "unique_spatial_queries": 7262, "spatial_stations": 104, "spatial_models": len(ARMS),
        "query_truth_and_masks_verified": True, "paired_spatial_intervals_recomputed": True,
        "source_only_selection": True, "protocol_hash": digest(protocol), "seeds": list(SEEDS)}
    write_json(ROOT/"analysis/independent_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
