"""Independent arithmetic, topology, role-isolation and checkpoint replay audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_crossfit_river_v2 import (
    ATLAS,
    ROOT,
    check,
    digest,
    fit_local,
    load_inputs,
    predict_river,
    write,
)

from river_graph.experiments.provenance import sha256_file
from river_graph.models.crossfit_river import RELATIONS, WAYS, new_river, relation_banks
from river_graph.models.river_structure_residual import upstream_paths


def metric(frame):
    truth, pred = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    error = pred-truth
    tail = truth >= frame.q90_threshold.to_numpy(float)
    return {"mae": float(np.abs(error).mean()),
        "log1p_mae": float(np.abs(np.log1p(pred)-np.log1p(truth)).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1-np.sum(error**2)/np.sum((truth-truth.mean())**2)),
        "bias": float(error.mean()), "q90_mae": float(np.abs(error[tail]).mean()),
        "coverage": float(frame.support.mean()), "n": len(frame)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(2)
    protocol = json.loads((root/"protocol.json").read_text())
    for name, expected in {**protocol["dependencies"], **protocol["runtime"]}.items():
        assert sha256_file(name) == expected, name
    gate = json.loads((root/"scoring_gate.json").read_text())
    assert gate["protocol_hash"] == digest(protocol)
    metrics, frames, anchor_scores, deltas, strata = [], [], [], [], []
    max_replay, matching_count = 0., 0
    for split in protocol["splits"]:
        for seed in protocol["seeds"]:
            run = root/"runs"/f"split{split}_seed{seed}"
            config = json.loads((run/"config.json").read_text())
            check(run, "complete.json", config)
            check(run, "predictions_complete.json", config)
            assert gate["complete_sha256"][str(run.resolve())] == sha256_file(run/"complete.json")
            sidecar = json.loads((run/"predictions.meta.json").read_text())
            assert sidecar["prediction_sha256"] == sha256_file(run/"predictions.parquet")
            data, masks, inputs, _ = load_inputs(split, root)
            y, t = np.asarray(data["y"]), data["y"].shape[1]
            # Deliberately destroy outer DOC labels during base replay; results must persist.
            forbidden = np.unique(np.concatenate([masks["val"]//t, masks["test"]//t]))
            altered = y.copy()
            altered[forbidden] = np.nan
            full = fit_local(run, "local_full", config, inputs, masks["train"], altered, seed, lambda *_: None)
            roles = json.loads((run/"source_roles.json").read_text())
            train_ids = set(masks["train"]//t)
            for i, record in enumerate(roles):
                held, fitted = set(record["held_stations"]), set(record["fitted_stations"])
                assert not held & fitted and held | fitted == train_ids
                local_record = json.loads((run/f"local_fold{i}.json").read_text())
                assert fitted == set(local_record["fitted_stations"])
                assert set(local_record["fitted_cells"]) == set(masks["train"][~np.isin(masks["train"]//t, list(held))])
                assert not held & set(record["eligible_donors"])
                assert not local_record["pretrained"]
            init, count = set(), set()
            for way in WAYS:
                for arm in RELATIONS:
                    info = json.loads((run/f"{way}__{arm}.json").read_text())
                    init.add(info["init_hash"])
                    count.add(info["parameters"])
                    assert not info["validation_labels_used"]
                    assert info["complete_local_anchor_oof"] == (way == "crossfit")
            assert len(init) == len(count) == 1
            physical = pd.read_csv(ATLAS/"station_structure.csv", dtype={"station": str})
            edges = pd.read_csv(ATLAS/"station_edge_paths.csv", dtype={"source": str, "target": str})
            rev = edges.rename(columns={"source": "target", "target": "source"})
            names = np.asarray(data["site_no"], str)
            attrs = physical.set_index("station")
            attrs.index = attrs.index.astype(str)
            pred = pd.read_parquet(run/"predictions.parquet")
            assert not pred.duplicated(["role", "model", "cell"]).any()
            assert np.isfinite(pred.y_pred).all() and (pred.y_pred >= 0).all()
            for role in ("val", "test"):
                cells = masks[role]
                receiving = np.unique(cells//t)
                eligible = np.setdiff1d(np.arange(len(y)), receiving)
                bank = relation_banks(edges, physical, names, receiving, eligible, inputs["available"], seed+700)
                up = upstream_paths(edges, physical, names[receiving], names[eligible], candidates=len(eligible), max_km=np.inf)
                down = upstream_paths(rev, physical, names[receiving], names[eligible], candidates=len(eligible), max_km=np.inf)
                related = {int(receiver): {p[0] for p in (*up[i], *down[i])} for i, receiver in enumerate(receiving)}
                for record in bank["records"]:
                    receiver, donor = record["receiver"], record["nonconnected"]
                    assert names[donor] not in related[receiver]
                    assert attrs.loc[names[receiver], "comid"] != attrs.loc[names[donor], "comid"]
                    assert not set(receiving) & {record["directed"], record["generic"], donor}
                    matching_count += 1
                for way in WAYS:
                    for arm in RELATIONS:
                        model = new_river(seed)
                        model.load_state_dict(torch.load(run/f"{way}__{arm}.pt", weights_only=True))
                        values, support = predict_river(model, inputs, bank, full, cells, arm)
                        saved = pred.loc[(pred.role == role) & (pred.model == f"{way}__{arm}")]
                        np.testing.assert_array_equal(saved.cell, cells)
                        max_replay = max(max_replay, float(abs(saved.y_pred.to_numpy()-values).max()))
                        np.testing.assert_array_equal(saved.support, support)
                for model_name, frame in pred.query("role == @role").groupby("model"):
                    np.testing.assert_array_equal(frame.cell, cells)
                    np.testing.assert_array_equal(frame.y_true, y.ravel()[cells])
                    metrics.append({"split": split, "seed": seed, "role": role, "model": model_name, **metric(frame)})
                    for supported in (False, True):
                        subset = frame.loc[frame.support == supported]
                        if len(subset):
                            strata.append({"split": split, "seed": seed, "role": role, "model": model_name,
                                           "supported": supported, **metric(subset)})
            anchor = pd.read_parquet(run/"source_anchor_predictions.parquet")
            for way, frame in anchor.groupby("way"):
                err = frame.y_pred.to_numpy(float)-frame.y_true.to_numpy(float)
                anchor_scores.append({"split": split, "seed": seed, "way": way, "mae": float(abs(err).mean()),
                                      "bias": float(err.mean()), "n": len(err)})
            frames.append(pred)
    assert max_replay == 0, max_replay
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(analysis/"per_run_metrics.csv", index=False)
    frame.groupby(["role", "model"]).mean(numeric_only=True).drop(columns=["split", "seed"]).to_csv(analysis/"summary.csv")
    pd.DataFrame(strata).to_csv(analysis/"support_strata.csv", index=False)
    pd.DataFrame(anchor_scores).to_csv(analysis/"source_anchor_scores.csv", index=False)
    indexed = frame.set_index(["role", "model", "split", "seed"])
    for role in ("val", "test"):
        comparisons = [(f"crossfit__{arm}", f"fitted__{arm}") for arm in RELATIONS]
        comparisons += [(f"{way}__directed", f"{way}__{control}") for way in WAYS for control in ("generic", "nonconnected")]
        comparisons += [(f"{way}__{arm}", "local") for way in WAYS for arm in RELATIONS]
        for a, b in comparisons:
            first, second = indexed.loc[(role, a)], indexed.loc[(role, b)]
            for endpoint in ("mae", "log1p_mae", "rmse", "q90_mae"):
                difference = first[endpoint]-second[endpoint]
                deltas.append({"role": role, "first": a, "second": b, "metric": endpoint,
                    "mean_difference": difference.mean(), "relative_percent": 100*difference.mean()/second[endpoint].mean(),
                    "improved_runs": int((difference < 0).sum()), "runs": len(difference),
                    "split_min": float(difference.groupby("split").mean().min()),
                    "split_max": float(difference.groupby("split").mean().max())})
    pd.DataFrame(deltas).to_csv(analysis/"paired_comparisons.csv", index=False)
    pd.concat(frames).to_parquet(analysis/"all_predictions.parquet", index=False)
    write(analysis/"all_predictions.meta.json", {"protocol_hash": digest(protocol),
        "runtime_hash": digest(protocol["runtime"]), "dataset_hash": protocol["dependencies"][str(Path("data/processed/mississippi_graph_graphfix_st357.pt"))],
        "mask_hashes": protocol["mask_hashes"], "prediction_sha256": sha256_file(analysis/"all_predictions.parquet"),
        "source_predictions": {str(p): sha256_file(p) for p in sorted((root/"runs").glob("*/predictions.parquet"))}})
    write(analysis/"independent_audit.json", {"passed": True, "auditor_sha256": sha256_file(__file__), "max_checkpoint_replay_difference": max_replay,
        "all_query_populations_verified": True, "matching_records_checked": matching_count,
        "all_supervised_receiving_fold_labels_excluded": True, "equal_initialization_and_capacity": True,
        "no_validation_or_test_labels_used_in_training": True, "runs": 9, "model_role_metrics": len(frame),
        "protocol_hash": digest(protocol), "summary_sha256": sha256_file(analysis/"summary.csv")})
    print(frame.groupby(["role", "model"])[["mae", "log1p_mae", "rmse", "r2", "q90_mae", "coverage"]].mean().to_string())


if __name__ == "__main__":
    main()
