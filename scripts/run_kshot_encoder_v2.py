#!/usr/bin/env python3
"""Step 4: train H2X bases + support-set encoder v2, evaluate on frozen v2 tasks.

Spec: experiments/kshot_protocol_v2/support_encoder_spec_v2.md

Fold structure (per target basin r, model seed m):
  base_{-r}     H2X excluding r                    — evaluation base
  base_{r,s}    H2X excluding r ∪ s                 — episode base for source s
                                                    (inner exclusion; skip with
                                                    --inner-exclusion approximate)
  encoder_r     trained on episodes from sources s ≠ r, then scored on the
                frozen v2 tasks of r.

Base predictions are cached as .npy under --out-dir/base_preds so a re-run
does not retrain H2X. First run with --seeds 42 to confirm the residual head
moves in the right direction before scaling to 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import load_dataset, metrics
from river_graph.experiments.kshot import label_tasks, make_support_query_tasks
from river_graph.models.gcn import GCNDocModel
from river_graph.models.support_set_encoder import (
    HOP_CAP,
    QUERY_DIM,
    REGIME_DIM,
    SUPPORT_DIM,
    SupportSetEncoder,
    fit_residual,
)

PRIMARY = ("103001", "510020", "102701", "101302", "101900")


def h2x_config(seed: int) -> GCNDocModel:
    """Frozen H2X base config (protocol.json → base_model.config)."""
    return GCNDocModel(
        architecture="transport_enc",
        variant="river",
        edge_set="river",
        edge_direction="both",
        hidden=64,
        layers=2,
        dropout=0.1,
        lr=0.001,
        weight_decay=0.0,
        edge_dropout=0.0,
        share_weights=False,
        max_epochs=200,
        patience=20,
        env_groups=None,
        env_encoder=True,
        seed=seed,
    )


def hide_basins_split(y_mask: np.ndarray, hide_rows: list[int], seed: int) -> dict:
    """v2-style split with the union of hide_rows kept out of train/val."""
    from river_graph.experiments.kshot import make_region_split

    return make_region_split(y_mask, hide_rows, seed=seed)


class GraphGeometry:
    """Hops, signed direction, stream order, drainage area for all nodes."""

    def __init__(self, dataset: dict, reach_attributes: Path):
        n = dataset["y"].shape[0]
        self.n = n
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(dataset["edge_index"].T.tolist())
        self.length = dict(nx.all_pairs_shortest_path_length(g))
        dg = nx.DiGraph()
        dg.add_nodes_from(range(n))
        dg.add_edges_from(dataset["edge_index"].T.tolist())
        self.dg = dg
        self._dir_cache: dict[tuple[int, int], int] = {}

        sites = [str(s) for s in dataset["site_no"]]
        site_to_row = {}
        for i, s in enumerate(sites):
            for key in (s, s.zfill(8), s.lstrip("0")):
                site_to_row[key] = i
        ra = pd.read_csv(reach_attributes, dtype={"site_no": str})
        self.order = np.zeros(n, dtype=np.float64)
        self.area = np.zeros(n, dtype=np.float64)
        found = 0
        for site, so, ta in zip(
            ra.site_no.astype(str), ra["streamorde"], ra["totdasqkm"]
        ):
            for key in (site, site.zfill(8), site.lstrip("0")):
                if key in site_to_row:
                    r = site_to_row[key]
                    self.order[r] = float(so)
                    self.area[r] = float(np.log1p(max(float(ta), 0.0)))
                    found += 1
                    break
        if found < n:
            print(f"[geom] stream order matched {found}/{n} nodes", flush=True)

    def hops(self, a: int, b: int) -> int:
        h = self.length.get(a, {}).get(b)
        return int(h) if h is not None else HOP_CAP + 1

    def direction(self, s: int, q: int) -> int:
        key = (s, q)
        if key in self._dir_cache:
            return self._dir_cache[key]
        if nx.has_path(self.dg, s, q):
            val = 0  # support upstream of query → query downstream of support
        elif nx.has_path(self.dg, q, s):
            val = 1
        else:
            val = 2
        self._dir_cache[key] = val
        return val


def basin_rows(sites: list[str], nodes: pd.DataFrame, huc6: str) -> list[int]:
    meta = pd.DataFrame({"station": [str(s) for s in sites]})
    meta = meta.merge(
        nodes[["site_no", "huc_cd"]].rename(columns={"site_no": "station"}),
        on="station",
        how="left",
        validate="one_to_one",
    )
    prefix = meta["huc_cd"].astype(str).str[:6]
    return [int(i) for i in np.flatnonzero(prefix.eq(str(huc6)).to_numpy())]


class Standardizer:
    def __init__(self, x: np.ndarray):
        self.mu = x.mean(axis=0, keepdims=True)
        self.sd = x.std(axis=0, keepdims=True) + 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd


def build_episode_tensors(
    support_cells: list[int],
    query_cells: list[int],
    y: np.ndarray,
    base_pred: np.ndarray,
    regime: np.ndarray,
    geom: GraphGeometry,
    months: list[str],
    std: dict,
) -> dict:
    """Pack one same-month episode (spec §2) as torch tensors."""
    t = y.shape[1]
    y_flat, b_flat = y.ravel(), base_pred.ravel()
    reg_env = regime[:, 4 : 4 + REGIME_DIM]
    k, qn = len(support_cells), len(query_cells)

    s_tokens = np.zeros((k, SUPPORT_DIM), dtype=np.float32)
    s_dir = np.zeros((k, qn), dtype=np.int64)
    s_hop = np.zeros((k, qn), dtype=np.int64)
    s_ord = np.zeros(k, dtype=np.float32)
    s_area = np.zeros(k, dtype=np.float32)
    s_reg = np.zeros((k, REGIME_DIM), dtype=np.float32)
    q_tokens = np.zeros((qn, QUERY_DIM), dtype=np.float32)
    q_ord = np.zeros(qn, dtype=np.float32)
    q_area = np.zeros(qn, dtype=np.float32)
    q_reg = np.zeros((qn, REGIME_DIM), dtype=np.float32)
    base_log = np.zeros(qn, dtype=np.float32)
    y_log = np.zeros(qn, dtype=np.float32)

    month_num = int(months[query_cells[0] % t][5:7]) if query_cells else 1

    for i, sc in enumerate(support_cells):
        sr = sc // t
        log_y = float(np.log1p(max(y_flat[sc], 0.0)))
        log_b = float(np.log1p(max(b_flat[sc], 0.0)))
        s_ord[i] = geom.order[sr]
        s_area[i] = geom.area[sr]
        s_reg[i] = reg_env[sr]
        s_tokens[i, 0] = log_y
        s_tokens[i, 1] = log_y - log_b
        s_tokens[i, 2] = s_ord[i]
        s_tokens[i, 3] = s_area[i]
        s_tokens[i, 4] = 0.0  # time_gap reserved for Case C
        s_tokens[i, 5 : 5 + REGIME_DIM] = s_reg[i]

    for j, qc in enumerate(query_cells):
        qr = qc // t
        q_ord[j] = geom.order[qr]
        q_area[j] = geom.area[qr]
        q_reg[j] = reg_env[qr]
        log_b = float(np.log1p(max(b_flat[qc], 0.0)))
        q_tokens[j, 0] = log_b
        q_tokens[j, 1 : 1 + REGIME_DIM] = q_reg[j]
        q_tokens[j, 1 + REGIME_DIM] = np.sin(2 * np.pi * month_num / 12)
        q_tokens[j, 2 + REGIME_DIM] = np.cos(2 * np.pi * month_num / 12)
        q_tokens[j, 3 + REGIME_DIM] = float(k)
        # target_meta (4 cols) left at 0 — reserved for leave-one-analyte-out
        base_log[j] = log_b
        y_log[j] = float(np.log1p(max(y_flat[qc], 0.0)))
        for i, sc in enumerate(support_cells):
            sr = sc // t
            s_hop[i, j] = geom.hops(sr, qr)
            s_dir[i, j] = geom.direction(sr, qr)

    s_tokens[:, 0:4] = std["sup"](s_tokens[:, 0:4])
    s_tokens[:, 5:] = std["reg"](s_tokens[:, 5:])
    q_tokens[:, 0:1] = std["qbase"](q_tokens[:, 0:1])
    q_tokens[:, 1 : 1 + REGIME_DIM] = std["reg"](
        q_tokens[:, 1 : 1 + REGIME_DIM]
    )
    q_tokens[:, 3 + REGIME_DIM : 4 + REGIME_DIM] = std["kk"](
        q_tokens[:, 3 + REGIME_DIM : 4 + REGIME_DIM]
    )
    s_ord[:] = std["ord"](s_ord.reshape(-1, 1)).ravel()
    q_ord[:] = std["ord"](q_ord.reshape(-1, 1)).ravel()
    s_area[:] = std["area"](s_area.reshape(-1, 1)).ravel()
    q_area[:] = std["area"](q_area.reshape(-1, 1)).ravel()
    s_reg = std["reg"](s_reg)
    q_reg = std["reg"](q_reg)

    return {
        "support_tokens": torch.tensor(s_tokens, dtype=torch.float32),
        "support_dir": torch.tensor(s_dir, dtype=torch.long),
        "support_hop": torch.tensor(s_hop, dtype=torch.long),
        "support_order": torch.tensor(s_ord, dtype=torch.float32),
        "support_area": torch.tensor(s_area, dtype=torch.float32),
        "support_regime": torch.tensor(s_reg, dtype=torch.float32),
        "query_tokens": torch.tensor(q_tokens, dtype=torch.float32),
        "query_order": torch.tensor(q_ord, dtype=torch.float32),
        "query_area": torch.tensor(q_area, dtype=torch.float32),
        "query_regime": torch.tensor(q_reg, dtype=torch.float32),
        "base_log": torch.tensor(base_log, dtype=torch.float32),
        "y_log": torch.tensor(y_log, dtype=torch.float32),
    }


def fit_standardizers(
    y: np.ndarray,
    base_pred: np.ndarray,
    regime: np.ndarray,
    cells: list[int],
    geom: GraphGeometry,
) -> dict:
    """Train-fold statistics only (spec §2.3)."""
    reg_env = regime[:, 4 : 4 + REGIME_DIM]
    t = y.shape[1]
    log_y = np.log1p(np.clip(y.ravel()[cells], 0, None)).reshape(-1, 1)
    log_b = np.log1p(np.clip(base_pred.ravel()[cells], 0, None)).reshape(-1, 1)
    rows = np.array([c // t for c in cells], dtype=int)
    resid = log_y - log_b
    return {
        "sup": Standardizer(np.hstack([log_y, resid, geom.order[rows, None], geom.area[rows, None]])),
        "qbase": Standardizer(log_b),
        "reg": Standardizer(reg_env[rows]),
        "kk": Standardizer(np.array([[1.0], [3.0], [5.0]])),
        "ord": Standardizer(geom.order[rows, None]),
        "area": Standardizer(geom.area[rows, None]),
    }


def train_encoder(
    episodes: list[dict],
    seed: int,
    hidden: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
) -> SupportSetEncoder:
    torch.manual_seed(seed)
    model = SupportSetEncoder(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_state = None
    bad = 0
    for _ in range(max_epochs):
        order = np.random.default_rng(seed).permutation(len(episodes))
        model.train()
        total, count = 0.0, 0
        for idx in order:
            ep = episodes[int(idx)]
            if ep["support_tokens"].shape[0] == 0 or ep["query_tokens"].shape[0] == 0:
                continue
            loss = fit_residual(
                model,
                ep["support_tokens"], ep["support_dir"], ep["support_hop"],
                ep["support_order"], ep["support_area"], ep["support_regime"],
                ep["query_tokens"], ep["query_order"], ep["query_area"],
                ep["query_regime"],
                ep["base_log"], ep["y_log"],
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * ep["query_tokens"].shape[0]
            count += ep["query_tokens"].shape[0]
        epoch_loss = total / max(count, 1)
        if epoch_loss < best - 1e-6:
            best = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predict_delta(model: SupportSetEncoder, ep: dict) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(
            ep["support_tokens"], ep["support_dir"], ep["support_hop"],
            ep["support_order"], ep["support_area"], ep["support_regime"],
            ep["query_tokens"], ep["query_order"], ep["query_area"],
            ep["query_regime"],
        ).numpy()


def load_frozen_tasks(proto_dir: Path, region: str, task_seed: int) -> list[dict]:
    path = proto_dir / "tasks" / region / f"{task_seed}.json"
    payload = json.loads(path.read_text())
    return payload["tasks"]


def get_base_pred(
    cache_dir: Path,
    tag: str,
    dataset: dict,
    hide_rows: list[int],
    seed: int,
    max_epochs: int,
    patience: int,
    smoke: bool,
) -> np.ndarray:
    """H2X base prediction (N, T) mg/L, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}.npy"
    if path.exists():
        return np.load(path)
    if smoke:
        max_epochs, patience = 2, 2
    split = hide_basins_split(dataset["y_mask"], hide_rows, seed=seed)
    model = h2x_config(seed)
    model.max_epochs = max_epochs
    model.patience = patience
    t0 = time.perf_counter()
    model.fit(dataset, split)
    pred = model.predict()
    np.save(path, pred)
    print(f"[base {tag}] trained in {time.perf_counter() - t0:.1f}s -> {path}", flush=True)
    return pred


def episodes_from_tasks(
    tasks: list,
    y: np.ndarray,
    base_pred: np.ndarray,
    regime: np.ndarray,
    geom: GraphGeometry,
    months: list[str],
    std: dict,
    k_list: tuple[int, ...] = (1, 3, 5),
) -> list[dict]:
    out = []
    for task in tasks:
        if isinstance(task, dict):
            query = list(task["query_cells"])
            support_by_k = {int(k): list(v) for k, v in task["support_cells_by_k"].items()}
        else:  # KShotTask
            query = list(task.query)
            support_by_k = {int(k): list(v) for k, v in task.support_by_k.items()}
        for k in k_list:
            support = support_by_k.get(k, [])
            if not support or not query:
                continue
            out.append(
                build_episode_tensors(
                    support, query, y, base_pred, regime, geom, months, std
                )
            )
    return out


def score_methods(y_true_mg: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, yp in preds.items():
        m = metrics(y_true_mg, yp)
        m["method"] = name
        rows.append(m)
    return pd.DataFrame(rows)


def simple_baselines(
    support: list[int],
    query: list[int],
    y: np.ndarray,
    base: np.ndarray,
    geom: GraphGeometry,
) -> dict[str, np.ndarray]:
    """local_mean / mean_bias / residual_idw / nearest on one episode."""
    t = y.shape[1]
    y_flat, b_flat = y.ravel(), base.ravel()
    if not support:
        vals = b_flat[query]
        return {n: vals for n in ("local_mean", "mean_bias", "idw", "nearest")}
    svals = y_flat[support]
    local_mean = float(svals.mean())
    resid_mean = float((svals - b_flat[support]).mean())
    out = {n: np.zeros(len(query)) for n in ("local_mean", "mean_bias", "idw", "nearest")}
    for j, qc in enumerate(query):
        qr = qc // t
        out["local_mean"][j] = local_mean
        out["mean_bias"][j] = b_flat[qc] + resid_mean
        wsum = rsum = 0.0
        best_h, best_v = None, local_mean
        for sc in support:
            sr = sc // t
            h = geom.hops(sr, qr)
            w = 1.0 / (h + 1.0)
            wsum += w
            rsum += w * (y_flat[sc] - b_flat[sc])
            if best_h is None or h < best_h:
                best_h, best_v = h, y_flat[sc]
        out["idw"][j] = b_flat[qc] + (rsum / wsum if wsum > 0 else 0.0)
        out["nearest"][j] = best_v
    return out


def gate_direction_report(
    delta: np.ndarray,
    base_log: np.ndarray,
    y_log: np.ndarray,
) -> dict[str, float]:
    """Does Δ move toward the truth? (Step 4 gate-direction check.)"""
    true_resid = y_log - base_log
    if len(delta) < 2 or float(np.std(delta)) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(delta, true_resid)[0, 1])
    before = float(np.mean(np.abs(base_log - y_log)))
    after = float(np.mean(np.abs(base_log + delta - y_log)))
    return {
        "corr_delta_true_resid": corr,
        "mean_delta": float(np.mean(delta)),
        "mean_true_resid": float(np.mean(true_resid)),
        "log_mae_before": before,
        "log_mae_after": after,
        "sign_agree": float(np.mean(np.sign(delta) == np.sign(true_resid))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--nodes", default="data/processed/graph_nodes_graphfix_st357.csv")
    ap.add_argument("--reach-attributes", default="data/processed/reach_attributes.csv")
    ap.add_argument("--protocol-dir", default="experiments/kshot_protocol_v2")
    ap.add_argument("--targets", nargs="+", default=list(PRIMARY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--task-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--inner-exclusion", choices=("clean", "approximate"), default="clean")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--encoder-epochs", type=int, default=200)
    ap.add_argument("--out-dir", default="experiments/kshot_encoder_v2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    cache_dir = out / "base_preds"

    dataset = load_dataset(args.dataset)
    y = np.asarray(dataset["y"])
    y_mask = np.asarray(dataset["y_mask"])
    regime = np.asarray(dataset["regime"], dtype=np.float64)
    months = [str(m) for m in dataset["months"]]
    sites = [str(s) for s in dataset["site_no"]]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    geom = GraphGeometry(dataset, Path(args.reach_attributes))
    proto_dir = Path(args.protocol_dir)

    row_by_basin = {b: basin_rows(sites, nodes, b) for b in PRIMARY}
    protocol = {
        "spec": "experiments/kshot_protocol_v2/support_encoder_spec_v2.md",
        "targets": args.targets,
        "seeds": args.seeds,
        "task_seeds": args.task_seeds,
        "inner_exclusion": args.inner_exclusion,
        "max_epochs": args.max_epochs,
        "encoder_epochs": args.encoder_epochs,
        "smoke": bool(args.smoke),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metric_frames, gate_rows = [], []
    for seed in args.seeds:
        for target in args.targets:
            sources = [b for b in PRIMARY if b != target]
            hide_r = list(row_by_basin[target])
            tag_r = f"{target}_excl_s{seed}"
            base_r = get_base_pred(
                cache_dir, tag_r, dataset, hide_r, seed,
                args.max_epochs, args.patience, args.smoke,
            )

            # ---- training episodes from sources (spec §5) ----
            source_cells: list[int] = []
            episodes: list[dict] = []
            # provisional std from source-basin observed cells (no target stats)
            src_rows = sorted({i for s in sources for i in row_by_basin[s]})
            obs_cells = [
                int(f) for f in np.flatnonzero(y_mask.ravel()) if (f // y.shape[1]) in set(src_rows)
            ]
            std = fit_standardizers(y, base_r, regime, obs_cells, geom)

            for src in sources:
                hide_rs = list(hide_r) + list(row_by_basin[src])
                if args.inner_exclusion == "clean":
                    base_src = get_base_pred(
                        cache_dir,
                        f"{target}_{src}_excl_s{seed}",
                        dataset, hide_rs, seed,
                        args.max_epochs, args.patience, args.smoke,
                    )
                else:
                    base_src = base_r
                src_tasks = label_tasks(
                    make_support_query_tasks(
                        y_mask, row_by_basin[src], months,
                        k_list=(1, 3, 5), min_query=2, seed=42,
                    ),
                    src,
                )
                eps = episodes_from_tasks(
                    src_tasks, y, base_src, regime, geom, months, std
                )
                episodes.extend(eps)
                for t in src_tasks:
                    source_cells.extend(t.query)
                    for sup in t.support_by_k.values():
                        source_cells.extend(sup)
            if args.smoke:
                episodes = episodes[:8]

            print(
                f"[{target} s{seed}] episodes={len(episodes)} sources={sources} "
                f"inner={args.inner_exclusion}",
                flush=True,
            )
            if not episodes:
                print(f"[{target} s{seed}] no training episodes — skipping", flush=True)
                continue

            enc = train_encoder(
                episodes, seed=seed, max_epochs=args.encoder_epochs
            )

            # ---- evaluate on frozen v2 tasks ----
            for task_seed in args.task_seeds:
                f_tasks = load_frozen_tasks(proto_dir, target, task_seed)
                if args.smoke:
                    f_tasks = f_tasks[:3]
                methods = (
                    "k0_base", "local_mean", "mean_bias", "idw", "nearest",
                    "support_encoder_v2", "se_value_shuffle", "se_site_shuffle",
                )
                true_by_k: dict[int, list] = {k: [] for k in (0, 1, 3, 5)}
                pred_by_k: dict[int, dict[str, list]] = {
                    k: {m: [] for m in methods} for k in (0, 1, 3, 5)
                }
                for task in f_tasks:
                    query = list(task["query_cells"])
                    if not query:
                        continue
                    y_true = np.clip(y.ravel()[query], 0, None)
                    base_q = np.clip(base_r.ravel()[query], 0, None)
                    for k in (0, 1, 3, 5):
                        true_by_k[k].append(y_true)
                        pred_by_k[k]["k0_base"].append(base_q)
                        if k == 0:
                            for m in methods[1:]:
                                pred_by_k[k][m].append(base_q)
                            continue
                        support = list(task["support_cells_by_k"].get(str(k), []))
                        if not support:
                            for m in methods[1:]:
                                pred_by_k[k][m].append(base_q)
                            continue
                        simple = simple_baselines(support, query, y, base_r, geom)
                        for n, v in simple.items():
                            pred_by_k[k][n].append(np.clip(v, 0, None))
                        ep = build_episode_tensors(
                            support, query, y, base_r, regime, geom, months, std
                        )
                        delta = predict_delta(enc, ep)
                        pred = np.expm1(np.log1p(base_q) + delta)
                        pred_by_k[k]["support_encoder_v2"].append(np.clip(pred, 0, None))
                        gate_rows.append(
                            {
                                "target": target,
                                "seed": seed,
                                "task_seed": task_seed,
                                "month": task["month"],
                                "k": k,
                                **gate_direction_report(
                                    delta, np.log1p(base_q), np.log1p(y_true)
                                ),
                            }
                        )
                        # value-shuffle: keep sites, permute values within the set
                        rng = np.random.default_rng(seed + k)
                        vals = [y.ravel()[c] for c in support]
                        perm = rng.permutation(len(support))
                        y_v = y.copy()
                        for i, sc in enumerate(support):
                            y_v.ravel()[sc] = vals[int(perm[i])]
                        ep_v = build_episode_tensors(
                            support, query, y_v, base_r, regime, geom, months, std
                        )
                        dv = predict_delta(enc, ep_v)
                        pred_by_k[k]["se_value_shuffle"].append(
                            np.clip(np.expm1(np.log1p(base_q) + dv), 0, None)
                        )
                        # site-shuffle: same values, other non-query sites same month
                        month_col = support[0] % y.shape[1]
                        alt = [
                            r_ * y.shape[1] + month_col
                            for r_ in row_by_basin[target]
                            if y_mask[r_, month_col]
                            and (r_ * y.shape[1] + month_col) not in query
                            and (r_ * y.shape[1] + month_col) not in support
                        ]
                        if len(alt) >= len(support):
                            pick = [
                                alt[i]
                                for i in rng.permutation(len(alt))[: len(support)]
                            ]
                            y_keep = y.copy()
                            for src_c, dst_c in zip(support, pick):
                                y_keep.ravel()[dst_c] = y.ravel()[src_c]
                            ep_s = build_episode_tensors(
                                pick, query, y_keep, base_r, regime, geom, months, std
                            )
                            ds = predict_delta(enc, ep_s)
                            pred_by_k[k]["se_site_shuffle"].append(
                                np.clip(np.expm1(np.log1p(base_q) + ds), 0, None)
                            )
                        else:
                            pred_by_k[k]["se_site_shuffle"].append(base_q)

                for k in (0, 1, 3, 5):
                    if not true_by_k[k]:
                        continue
                    yt = np.concatenate(true_by_k[k])
                    preds = {
                        n: np.concatenate(v) if v else np.zeros(0)
                        for n, v in pred_by_k[k].items()
                    }
                    df = score_methods(yt, preds)
                    df["target"] = target
                    df["seed"] = seed
                    df["task_seed"] = task_seed
                    df["k"] = k
                    metric_frames.append(df)
                print(
                    f"[{target} s{seed} task{task_seed}] scored K=0/1/3/5",
                    flush=True,
                )

    if metric_frames:
        all_m = pd.concat(metric_frames, ignore_index=True)
        all_m.to_csv(out / "metrics_by_task.csv", index=False)
        summary = (
            all_m.groupby(["method", "k"])[["mae", "rmse", "r2", "log_mae"]]
            .mean()
            .sort_values(["k", "mae"])
        )
        summary.to_csv(out / "metrics_summary.csv")
        print("\n=== mean over tasks ===")
        print(summary.to_string())
    if gate_rows:
        g = pd.DataFrame(gate_rows)
        g.to_csv(out / "gate_direction.csv", index=False)
        print("\n=== gate direction (Δ vs true residual) ===")
        print(
            g.groupby("k")[
                ["corr_delta_true_resid", "sign_agree", "log_mae_before", "log_mae_after"]
            ]
            .mean()
            .to_string()
        )


if __name__ == "__main__":
    main()
