"""Prospective 2x3 DOC cross-fitting/river controls, isolated run_ladder entry."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.provenance import sha256_file
from river_graph.models.crossfit_river import (
    RELATIONS,
    WAYS,
    LocalEnvironmentGRU,
    covariates,
    local_batch,
    native,
    new_river,
    relation_banks,
    river_batch,
)
from river_graph.models.hydro_river_expansion import FrozenHydroPreprocessing
from river_graph.models.kgml_local_transport import station_folds

ROOT = Path("experiments/phase4_transfer/doc_crossfit_river_v1")
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DAILY = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1/daily_features.npz")
DAILY_META = DAILY.with_suffix(".meta.json")
ATLAS = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1/analysis")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
CODE = ("scripts/run_doc_crossfit_river_v1.py", "scripts/start_doc_crossfit_river_v1.py",
        "scripts/verify_doc_crossfit_river_v1.py", "src/river_graph/models/crossfit_river.py",
        "src/river_graph/models/dynamic_river_state.py", "src/river_graph/models/dynamic_river_residual.py",
        "src/river_graph/models/hydro_river_expansion.py", "src/river_graph/models/river_structure_residual.py",
        "src/river_graph/models/kgml_local_transport.py", "tests/test_doc_crossfit_river_v1.py")


def digest(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def bind(run, name, files, config):
    write(run/name, {"config_hash": digest(config), "files": {x: sha256_file(run/x) for x in files}})


def check(run, name, config):
    payload = json.loads((run/name).read_text())
    assert payload["config_hash"] == digest(config)
    for path, expected in payload["files"].items():
        assert sha256_file(run/path) == expected, path


def freeze(root, repository):
    dependencies = {str(p): sha256_file(p) for p in (DATA, DAILY, DAILY_META,
        ATLAS/"station_structure.csv", ATLAS/"station_edge_paths.csv",
        Path("docs/doc_crossfit_river_plan_20261011.md"))}
    root.mkdir(parents=True, exist_ok=True)
    mask_hashes = {}
    for split in (142, 143, 144):
        destination = root/"masks"/f"split{split}.npz"
        destination.parent.mkdir(exist_ok=True)
        original = MASKS/destination.name
        if destination.exists() and destination.read_bytes() != original.read_bytes():
            raise ValueError("split changed")
        destination.write_bytes(original.read_bytes())
        mask_hashes[str(split)] = sha256_file(destination)
    protocol = {"experiment": ROOT.name, "splits": [142, 143, 144], "seeds": [42, 43, 44],
        "relations": list(RELATIONS), "ways": list(WAYS), "dependencies": dependencies,
        "mask_hashes": mask_hashes, "runtime": {str(name): sha256_file(name) for name in
            [*sorted(Path("src/river_graph").rglob("*.py")), *map(Path, CODE)]},
        "forest": {"trees": 200, "leaf": 4, "max_features": 1., "jobs": 2},
        "local": {"epochs": 30, "batch": 512, "lr": .001, "environment": [15, 24],
                  "gru": [14, 32], "lookback": 12, "penalty": .001},
        "river": {"epochs": 30, "batch": 512, "lr": .001, "heads": 2, "dimensions": 32,
                  "lags": [0, 1, 3], "slots": [10, 10], "max_km": 3000},
        "loss": "native MAE; source Q90 weight 2; fixed final epoch, no validation selection",
        "evaluation": "source-validation diagnostic, untouched outer-test primary; all nine packages before either scoring",
        "primary": ["mae", "log1p_mae"], "secondary": ["rmse", "r2", "bias", "q90_mae"],
        "aggregation": "equal split-seed mean; per-run R2 on identical complete queries; no prediction ensemble",
        "uncertainty": "descriptive paired differences across three overlapping partitions, all nine runs; station-cluster bootstrap supplementary, not external population proof",
        "crossfit": "five whole-source-station folds; forest AND every supervised neural layer retrained excluding receiving fold",
        "anchor": "fresh ecological ExtraTrees + MLP/causal GRU; no legacy fitted weights, DOC history, station IDs or water quality",
        "matching": "same seeded correction tensors and optimizer permutations; common edge-date support for all three donor banks",
        "scope": "K0 internal known atlas; full covariate nodes except complete receiving role/fold; not external river",
        "prior_paper_endpoints_modified": False, "local_fits": 54, "river_fits": 54}
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text()) != protocol:
            raise ValueError("frozen execution changed: use new version")
    else:
        write(root/"protocol.json", protocol)
        write(root/"manifest.json", {"protocol_hash": digest(protocol),
            "code_commit": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
            "workspace_dirty": bool(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)),
            "generator_script_sha256": sha256_file(CODE[0]), "dependencies": dependencies,
            "mask_hashes": mask_hashes, "runtime": protocol["runtime"]})
        for name in CODE:
            destination = root/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    return protocol


def load_inputs(split, root):
    data = torch.load(DATA, weights_only=False)
    with np.load(root/"masks"/f"split{split}.npz") as saved:
        masks = {k: saved[k].copy() for k in saved.files}
    with np.load(DAILY) as saved:
        daily = saved["full"].astype(np.float32)
    metadata = json.loads(DAILY_META.read_text())
    assert metadata["dataset_hash"] == sha256_file(DATA)
    prep = FrozenHydroPreprocessing.fit(data, masks["train"])
    return data, masks, covariates(prep.transform(data, daily)), prep.statistics


def fit_local(run, name, config, inputs, cells, y, seed, progress):
    targets = np.asarray(y, np.float64).ravel()[cells]
    threshold = float(np.quantile(targets, .9))
    model = LocalEnvironmentGRU(seed)
    if (run/f"{name}_complete.json").exists():
        check(run, f"{name}_complete.json", config)
        forest = joblib.load(run/f"{name}.joblib")
        model.load_state_dict(torch.load(run/f"{name}.pt", weights_only=True))
    else:
        forest = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=4,
            max_features=1., n_jobs=2, random_state=seed).fit(inputs["forest"][cells], np.log1p(targets))
        base = torch.tensor(forest.predict(inputs["forest"][cells]), dtype=torch.float32)
        truth = torch.tensor(targets, dtype=torch.float32)
        weights = torch.where(truth >= threshold, 2., 1.)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rng = np.random.default_rng(seed+101)
        trace = []
        for epoch in range(30):
            model.train()
            total = 0.
            for batch in np.array_split(rng.permutation(len(cells)), max(1, int(np.ceil(len(cells)/512)))):
                optimizer.zero_grad()
                delta, _ = model(*local_batch(inputs, cells[batch]))
                loss = ((native(base[batch]+delta)-truth[batch]).abs()*weights[batch]).sum()/weights[batch].sum()+.001*delta.square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach())*len(batch)
            trace.append(total/len(cells))
            if epoch % 5 == 0 or epoch == 29:
                progress(name, {"epoch": epoch+1, "train_loss": trace[-1]})
        joblib.dump(forest, run/f"{name}.joblib", compress=3)
        torch.save(model.state_dict(), run/f"{name}.pt")
        write(run/f"{name}.json", {"fitted_cells": cells.tolist(), "fitted_stations": np.unique(cells//y.shape[1]).tolist(),
            "q90": threshold, "trace": trace, "init_hash": digest({"seed": seed}), "pretrained": False})
        bind(run, f"{name}_complete.json", [f"{name}.joblib", f"{name}.pt", f"{name}.json"], config)
    model.eval()
    n, t = y.shape
    logpred = forest.predict(inputs["forest"]).astype(np.float32)
    hidden = np.empty((n*t, 56), np.float32)
    with torch.no_grad():
        for start in range(0, n*t, 4096):
            selected = np.arange(start, min(start+4096, n*t))
            delta, state = model(*local_batch(inputs, selected))
            logpred[selected] += delta.numpy()
            hidden[selected] = state.numpy()
    logpred = np.maximum(logpred.reshape(n, t), 0)
    states = np.concatenate([hidden.reshape(n, t, 56), np.broadcast_to(inputs["env"][:, None], (n, t, 15)),
                             inputs["daily"], inputs["hydro"]], -1).astype(np.float32)
    return {"logpred": logpred, "states": states}


def fit_river(run, name, config, inputs, groups, y, seed, progress):
    model = new_river(seed)
    init_hash = digest({k: v.detach().numpy().tolist() for k, v in model.state_dict().items()})
    if (run/f"{name}_complete.json").exists():
        check(run, f"{name}_complete.json", config)
        model.load_state_dict(torch.load(run/f"{name}.pt", weights_only=True))
        return model
    arm = name.split("__")[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    targets = np.asarray(y, np.float32).ravel()
    train_cells = np.concatenate([g["cells"] for g in groups])
    threshold = np.quantile(targets[train_cells], .9)
    # Both ways and all controls receive identical cell permutations.
    ordered_cells = np.sort(train_cells)
    group_id = {int(c): i for i, g in enumerate(groups) for c in g["cells"]}
    rng = np.random.default_rng(seed+2201)
    trace = []
    for epoch in range(30):
        total = 0.
        for batch in np.array_split(rng.permutation(len(ordered_cells)), int(np.ceil(len(ordered_cells)/512))):
            cells = ordered_cells[batch]
            optimizer.zero_grad()
            numerator, denominator = None, 0.
            for i, group in enumerate(groups):
                take = np.asarray([group_id[int(c)] == i for c in cells])
                if not take.any():
                    continue
                queried = cells[take]
                arrays, base = river_batch(inputs, group["bank"], group["anchor"], queried, arm)
                delta, _ = model(**arrays)
                truth = torch.tensor(targets[queried])
                weights = torch.where(truth >= threshold, 2., 1.)
                part = ((native(base+delta)-truth).abs()*weights).sum()
                numerator = part if numerator is None else numerator+part
                denominator += float(weights.sum())
            loss = numerator/denominator
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(cells)
        trace.append(total/len(ordered_cells))
        if epoch % 5 == 0 or epoch == 29:
            progress(name, {"epoch": epoch+1, "train_loss": trace[-1]})
    torch.save(model.state_dict(), run/f"{name}.pt")
    write(run/f"{name}.json", {"trace": trace, "init_hash": init_hash,
        "parameters": sum(p.numel() for p in model.parameters()), "fixed_epoch": 30,
        "complete_local_anchor_oof": name.startswith("crossfit"), "validation_labels_used": False})
    bind(run, f"{name}_complete.json", [f"{name}.pt", f"{name}.json"], config)
    return model


def predict_river(model, inputs, bank, anchor, cells, arm):
    pieces, support = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cells), 512):
            selected = cells[start:start+512]
            arrays, base = river_batch(inputs, bank, anchor, selected, arm)
            delta, diagnostic = model(**arrays)
            prediction = native(base+delta).double().numpy()
            pieces.append(prediction)
            support.append(diagnostic["support"].numpy())
    return np.concatenate(pieces), np.concatenate(support)


def run_one(task):
    root, split, seed = task
    root = Path(root)
    torch.set_num_threads(2)
    start = time.monotonic()
    protocol = json.loads((root/"protocol.json").read_text())
    run = root/"runs"/f"split{split}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    config = {"split": split, "seed": seed, "protocol_hash": digest(protocol),
        "dataset_hash": protocol["dependencies"][str(DATA)], "mask_hash": protocol["mask_hashes"][str(split)]}
    if (run/"complete.json").exists():
        check(run, "complete.json", config)
        return
    write(run/"config.json", config)

    def progress(stage, row):
        record = {"run": run.name, "stage": stage, "elapsed": time.monotonic()-start, **row}
        write(run/"progress.json", record)
        print(json.dumps(record), flush=True)

    data, masks, inputs, statistics = load_inputs(split, root)
    y, t = np.asarray(data["y"]), data["y"].shape[1]
    names = np.asarray(data["site_no"], str)
    physical = pd.read_csv(ATLAS/"station_structure.csv", dtype={"station": str})
    edges = pd.read_csv(ATLAS/"station_edge_paths.csv", dtype={"source": str, "target": str})
    full = fit_local(run, "local_full", config, inputs, masks["train"], y, seed, progress)
    groups_fitted, groups_crossfit, roles = [], [], []
    for i, held in enumerate(station_folds(masks["train"], t, seed)):
        cells = masks["train"][np.isin(masks["train"]//t, held)]
        fit_cells = masks["train"][~np.isin(masks["train"]//t, held)]
        anchor = fit_local(run, f"local_fold{i}", config, inputs, fit_cells, y, seed, progress)
        eligible = np.setdiff1d(np.arange(len(names)), held)
        bank = relation_banks(edges, physical, names, held, eligible, inputs["available"], seed+i)
        groups_fitted.append({"cells": cells, "bank": bank, "anchor": full})
        groups_crossfit.append({"cells": cells, "bank": bank, "anchor": anchor})
        roles.append({"fold": i, "held_stations": held.tolist(), "fitted_stations": np.unique(fit_cells//t).tolist(),
                      "eligible_donors": eligible.tolist(), "matching": bank["records"]})
    write(run/"source_roles.json", roles)
    write(run/"normalization.json", statistics)
    # Training mismatch diagnostics only; no held-out truth is consumed.
    diagnostic = []
    for way, groups in (("fitted", groups_fitted), ("crossfit", groups_crossfit)):
        for group in groups:
            cells = group["cells"]
            diagnostic.append(pd.DataFrame({"way": way, "cell": cells,
                "y_true": y.ravel()[cells], "y_pred": np.expm1(group["anchor"]["logpred"].ravel()[cells])}))
    pd.concat(diagnostic).to_parquet(run/"source_anchor_predictions.parquet", index=False)
    for way, groups in (("fitted", groups_fitted), ("crossfit", groups_crossfit)):
        for arm in RELATIONS:
            fit_river(run, f"{way}__{arm}", config, inputs, groups, y, seed, progress)
    files = ["config.json", "source_roles.json", "normalization.json", "source_anchor_predictions.parquet"]
    for name in ("local_full", *(f"local_fold{i}" for i in range(5))):
        files.extend(f"{name}{suffix}" for suffix in (".pt", ".joblib", ".json", "_complete.json"))
    for way in WAYS:
        for arm in RELATIONS:
            files.extend(f"{way}__{arm}{suffix}" for suffix in (".pt", ".json", "_complete.json"))
    bind(run, "complete.json", files, config)
    progress("complete", {"local_fits": 6, "river_fits": 6})
    del groups_fitted, groups_crossfit, full
    gc.collect()


def score(root):
    # Global gate: every supervised package must be bound before any val/test scoring.
    protocol = json.loads((root/"protocol.json").read_text())
    runs = [root/"runs"/f"split{s}_seed{k}" for s in protocol["splits"] for k in protocol["seeds"]]
    for run in runs:
        check(run, "complete.json", json.loads((run/"config.json").read_text()))
    gate = {"protocol_hash": digest(protocol), "complete_sha256": {str(r): sha256_file(r/"complete.json") for r in runs}}
    write(root/"scoring_gate.json", gate)
    for run in runs:
        config = json.loads((run/"config.json").read_text())
        if (run/"predictions_complete.json").exists():
            check(run, "predictions_complete.json", config)
            continue
        data, masks, inputs, _ = load_inputs(config["split"], root)
        y, t = np.asarray(data["y"]), data["y"].shape[1]
        full = fit_local(run, "local_full", config, inputs, masks["train"], y, config["seed"], lambda *_: None)
        physical = pd.read_csv(ATLAS/"station_structure.csv", dtype={"station": str})
        edges = pd.read_csv(ATLAS/"station_edge_paths.csv", dtype={"source": str, "target": str})
        frames, records = [], []
        for role in ("val", "test"):
            cells = masks[role]
            receiving = np.unique(cells//t)
            bank = relation_banks(edges, physical, data["site_no"], receiving,
                np.setdiff1d(np.arange(len(y)), receiving), inputs["available"], config["seed"]+700)
            records.append({"role": role, "receiving": receiving.tolist(), "matching": bank["records"]})
            base = native(torch.tensor(full["logpred"].ravel()[cells])).double().numpy()
            template = pd.DataFrame({"split": config["split"], "seed": config["seed"], "role": role,
                "cell": cells, "station": np.asarray(data["site_no"])[cells//t], "month": np.asarray(data["months"])[cells % t],
                "y_true": y.ravel()[cells], "q90_threshold": np.quantile(y.ravel()[masks["train"]], .9)})
            frame = template.copy()
            frame["model"], frame["y_pred"], frame["support"] = "local", base, bank["valid"][cells//t, cells % t].any((1, 2))
            frames.append(frame)
            for way in WAYS:
                for arm in RELATIONS:
                    model = new_river(config["seed"])
                    model.load_state_dict(torch.load(run/f"{way}__{arm}.pt", weights_only=True))
                    prediction, support = predict_river(model, inputs, bank, full, cells, arm)
                    np.testing.assert_array_equal(prediction[~support], base[~support])
                    frame = template.copy()
                    frame["model"], frame["y_pred"], frame["support"] = f"{way}__{arm}", prediction, support
                    frames.append(frame)
        write(run/"evaluation_roles.json", records)
        pd.concat(frames).to_parquet(run/"predictions.parquet", index=False)
        write(run/"predictions.meta.json", {**config, "runtime_hash": digest(protocol["runtime"]),
            "scoring_gate_sha256": sha256_file(root/"scoring_gate.json"),
            "prediction_sha256": sha256_file(run/"predictions.parquet"), "training_complete_sha256": sha256_file(run/"complete.json")})
        bind(run, "predictions_complete.json", ["predictions.parquet", "predictions.meta.json", "evaluation_roles.json"], config)
        print(json.dumps({"stage": "scored", "run": run.name}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    protocol = freeze(args.root, args.repository)
    if args.freeze_only:
        print(json.dumps({"frozen": digest(protocol)}))
        return
    torch.set_num_threads(2)
    tasks = [(str(args.root), s, k) for s in protocol["splits"] for k in protocol["seeds"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, tasks))
    score(args.root)
    write(args.root/"training_finished.json", {"pid": os.getpid(), "local_fits": 54, "river_fits": 54,
                                             "scored_packages": len(tasks)})


if __name__ == "__main__":
    main()
