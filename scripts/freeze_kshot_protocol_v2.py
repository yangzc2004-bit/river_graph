#!/usr/bin/env python3
"""Step 1 (rework): freeze the K-shot protocol — NO training.

Primary protocol (pre-registered):
- Regions: five HUC6 basins {103001, 510020, 102701, 101302, 101900}
- Hide ALL DOC labels of the HUC6 during training
- Evaluate only on the largest connected task component inside that HUC6
- Nested same-month support / fixed query, K in (0, 1, 3, 5)
- Base model for later runs: H2X (architecture=transport_enc); H2 is ablation
- Success gate is judged at K=5 (see protocol.json)

Writes under experiments/kshot_protocol_v2/:
  regions.json, connectivity_audit.csv, tasks/<region>/<task_seed>.json,
  candidate_rejections.md, protocol.json, manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from river_graph.experiments.kshot import MIN_QUERY

PRIMARY_HUC6 = ("103001", "510020", "102701", "101302", "101900")
BACKUP_HUC6 = ("101000", "101301")
SECONDARY_HUC8 = ("10300101", "10130201")
K_LIST = (0, 1, 3, 5)  # K=10 dropped from main protocol
TASK_SEEDS = (42, 43, 44)
MODEL_SEEDS = (42, 43, 44)
MIN_QUERY_DEFAULT = 2
MIN_COMPONENT_STATIONS = 7
MIN_K5_MONTHS = 3

# Frozen H2X base config (GCNDocModel / TransportGCNImputer defaults + run_gnn H2X).
BASE_MODEL_CONFIG = {
    "name": "H2X",
    "architecture": "transport_enc",
    "variant": "river",
    "edge_set": "river",
    "edge_direction": "both",
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "lr": 0.001,
    "weight_decay": 0.0,
    "edge_dropout": 0.0,
    "share_weights": False,
    "max_epochs": 200,
    "patience": 20,
    "env_groups": None,
    "env_encoder": True,
    "env_emb": 32,
    "loss": "log1p_mse",
    "train_split_seed": 42,
    "model_seeds": list(MODEL_SEEDS),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def huc_prefix(sites: list[str], nodes: pd.DataFrame, level: int = 6) -> pd.Series:
    meta = pd.DataFrame({"station": sites})
    meta = meta.merge(
        nodes[["site_no", "huc_cd"]].rename(columns={"site_no": "station"}),
        on="station",
        how="left",
        validate="one_to_one",
    )
    return meta["huc_cd"].astype(str).str[:level]


def largest_component_rows(edge_index: torch.Tensor, rows: list[int]) -> list[int]:
    """Largest weakly-connected component of the induced undirected subgraph."""
    row_set = set(rows)
    g = nx.Graph()
    g.add_nodes_from(rows)
    src, dst = edge_index.tolist()
    for a, b in zip(src, dst):
        a, b = int(a), int(b)
        if a in row_set and b in row_set:
            g.add_edge(a, b)
    if not rows:
        return []
    comps = list(nx.connected_components(g))
    if not comps:
        return []
    best = max(comps, key=len)
    return sorted(best)


def month_feasibility(y_mask: np.ndarray, rows: list[int], k_list: tuple[int, ...], min_query: int) -> dict:
    """Count months that can host each K with fixed query size >= min_query."""
    out = {k: 0 for k in k_list}
    n_months_ge5 = 0
    for j in range(y_mask.shape[1]):
        n = sum(1 for i in rows if y_mask[i, j])
        if n >= 5 + min_query:
            n_months_ge5 += 1
        for k in k_list:
            if k + min_query <= n:
                out[k] += 1
    out["n_months_ge5_support"] = n_months_ge5
    return out


def build_tasks(
    y_mask: np.ndarray,
    rows: list[int],
    months: list[str],
    k_list: tuple[int, ...],
    min_query: int,
    task_seed: int,
) -> list[dict]:
    """Nested support / fixed query per feasible month.

    Only months that can host the FULL K ladder are kept, so query cells are
    identical across every K in ``k_list`` (pre-registered fixed-query rule).
    """
    tasks = []
    t = y_mask.shape[1]
    max_k = max(k_list)
    for j in range(t):
        obs = [i for i in rows if y_mask[i, j]]
        if len(obs) < max_k + min_query:
            continue
        rng = np.random.default_rng(task_seed * 100003 + j)
        order = [int(obs[i]) for i in rng.permutation(len(obs))]
        support_pool = order[:max_k]
        query_rows = order[max_k:]
        if len(query_rows) < min_query:
            continue
        support_by_k = {k: support_pool[:k] for k in k_list}
        tasks.append(
            {
                "month_index": j,
                "month": months[j],
                "n_available": len(obs),
                "query_cells": [i * t + j for i in query_rows],
                "support_cells_by_k": {
                    str(k): [i * t + j for i in support_by_k[k]] for k in k_list
                },
                "query_rows": query_rows,
                "support_rows_by_k": {str(k): support_by_k[k] for k in k_list},
            }
        )
    return tasks


def validate_tasks(tasks: list[dict], y_mask: np.ndarray, k_list, min_query: int) -> list[str]:
    problems = []
    t = y_mask.shape[1]
    flat_obs = set(int(i) for i in np.flatnonzero(y_mask.ravel()))
    for task in tasks:
        q = set(task["query_cells"])
        if len(q) < min_query:
            problems.append(f"{task['month']}: query size {len(q)} < {min_query}")
        for flat in q:
            if flat not in flat_obs:
                problems.append(f"{task['month']}: query {flat} unobserved")
        prev = []
        for k in k_list:
            key = str(k)
            if key not in task["support_cells_by_k"]:
                problems.append(f"{task['month']}: missing K={k}")
                continue
            s = task["support_cells_by_k"][key]
            if len(s) != k:
                problems.append(f"{task['month']}: K={k} support size {len(s)}")
            if set(s) & q:
                problems.append(f"{task['month']}: K={k} support/query overlap")
            if prev and not set(prev) <= set(s):
                problems.append(f"{task['month']}: K={k} not nested")
            if not set(s) <= flat_obs:
                problems.append(f"{task['month']}: K={k} unobserved support")
            if k == 0 and s:
                problems.append(f"{task['month']}: K=0 must be empty")
            prev = s
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--nodes", default="data/processed/graph_nodes_graphfix_st357.csv")
    ap.add_argument("--out-dir", default="experiments/kshot_protocol_v2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "tasks").mkdir(parents=True, exist_ok=True)

    ds = torch.load(args.dataset, weights_only=False)
    y_mask = np.asarray(ds["y_mask"])
    sites = [str(s) for s in ds["site_no"]]
    months = [str(m) for m in ds["months"]]
    edge_index = ds["edge_index"]
    nodes = pd.read_csv(args.nodes, dtype={"site_no": str})
    huc6 = huc_prefix(sites, nodes, level=6)
    huc8 = huc_prefix(sites, nodes, level=8)

    # ---------- connectivity + feasibility audit ----------
    audit_rows = []
    region_defs = {}
    rejections = []
    for label, level_series, level in (
        ("huc6", huc6, 6),
        ("huc8", huc8, 8),
    ):
        for code in sorted(level_series.dropna().unique()):
            rows = [i for i, v in enumerate(level_series) if v == code]
            comps = []
            row_set = set(rows)
            g = nx.Graph()
            g.add_nodes_from(rows)
            for a, b in edge_index.T.tolist():
                a, b = int(a), int(b)
                if a in row_set and b in row_set:
                    g.add_edge(a, b)
            for comp in nx.connected_components(g):
                comp_rows = sorted(comp)
                feas = month_feasibility(y_mask, comp_rows, K_LIST, MIN_QUERY_DEFAULT)
                comps.append((len(comp_rows), comp_rows, feas))
            comps.sort(key=lambda x: x[0], reverse=True)
            for rank, (n_st, comp_rows, feas) in enumerate(comps):
                # structural path check inside the component (undirected)
                comp_g = nx.Graph()
                comp_g.add_nodes_from(comp_rows)
                for a, b in edge_index.T.tolist():
                    a, b = int(a), int(b)
                    if a in set(comp_rows) and b in set(comp_rows):
                        comp_g.add_edge(a, b)
                if n_st <= 1:
                    pairs_connected = True
                    mean_comp_hop = 0.0
                else:
                    try:
                        lengths = dict(nx.all_pairs_shortest_path_length(comp_g))
                        hops = [
                            lengths[a][b]
                            for a in comp_rows
                            for b in comp_rows
                            if a < b and b in lengths.get(a, {})
                        ]
                        pairs_connected = len(hops) == n_st * (n_st - 1) // 2
                        mean_comp_hop = float(np.mean(hops)) if hops else float("nan")
                    except Exception:
                        pairs_connected = False
                        mean_comp_hop = float("nan")
                audit_rows.append(
                    {
                        "scale": label,
                        "code": code,
                        "component_rank": rank,
                        "n_stations_huc": len(rows),
                        "n_stations_component": n_st,
                        "n_components": len(comps),
                        "months_k0": feas[0],
                        "months_k1": feas[1],
                        "months_k3": feas[3],
                        "months_k5": feas[5],
                        "months_ge5_support": feas["n_months_ge5_support"],
                        "support_query_paths_possible": bool(pairs_connected),
                        "mean_comp_hop": mean_comp_hop,
                    }
                )
            # primary selection: HUC6 largest component
            if label == "huc6" and code in PRIMARY_HUC6:
                if not comps:
                    rejections.append(f"HUC6 {code}: no component")
                    continue
                n_st, comp_rows, feas = comps[0]
                ok = n_st >= MIN_COMPONENT_STATIONS and feas["n_months_ge5_support"] >= MIN_K5_MONTHS
                region_defs[code] = {
                    "scale": "huc6",
                    "code": code,
                    "hide_labels": "all stations with this HUC6 prefix",
                    "hide_rows": rows,
                    "task_component_rows": comp_rows,
                    "n_stations_huc": len(rows),
                    "n_stations_component": n_st,
                    "n_components": len(comps),
                    "months_by_k": {str(k): feas[k] for k in K_LIST},
                    "months_ge5_support": feas["n_months_ge5_support"],
                    "eligible": ok,
                }
                if not ok:
                    rejections.append(
                        f"HUC6 {code}: component n={n_st} months_ge5={feas['n_months_ge5_support']} "
                        f"(need >={MIN_COMPONENT_STATIONS} st and >={MIN_K5_MONTHS} months)"
                    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "connectivity_audit.csv", index=False)

    # ---------- tasks for eligible primary regions ----------
    regions_json = {"primary": [], "backup": list(BACKUP_HUC6), "secondary_huc8": list(SECONDARY_HUC8)}
    all_task_hash = {}
    for code, rdef in sorted(region_defs.items()):
        rdir = out / "tasks" / code
        rdir.mkdir(parents=True, exist_ok=True)
        rows = rdef["task_component_rows"]
        if not rdef.get("eligible"):
            rejections.append(f"HUC6 {code}: skipped for main protocol (ineligible)")
            rdef["task_files"] = []
            regions_json["primary"].append(rdef)
            continue
        files = []
        for task_seed in TASK_SEEDS:
            tasks = build_tasks(y_mask, rows, months, K_LIST, MIN_QUERY_DEFAULT, task_seed)
            problems = validate_tasks(tasks, y_mask, K_LIST, MIN_QUERY_DEFAULT)
            if problems:
                raise SystemExit(f"task validation failed {code} seed={task_seed}: {problems[:5]}")
            payload = {
                "region": code,
                "task_seed": task_seed,
                "k_list": list(K_LIST),
                "min_query": MIN_QUERY_DEFAULT,
                "task_component_rows": rows,
                "n_tasks": len(tasks),
                "tasks": tasks,
            }
            path = rdir / f"{task_seed}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            files.append(str(path))
            all_task_hash[f"{code}/{task_seed}"] = sha256_obj(
                [{k: t[k] for k in ("month_index", "query_cells", "support_cells_by_k")} for t in tasks]
            )
        rdef["task_files"] = files
        rdef["task_hashes"] = {f.split("/")[-1]: all_task_hash[f"{code}/{Path(f).stem}"] for f in files}
        regions_json["primary"].append(rdef)
        print(
            f"[{code}] hide={len(rdef['hide_rows'])} component={len(rows)} "
            f"months_ge5={rdef['months_ge5_support']} tasks/task_seed≈{len(tasks)}",
            flush=True,
        )

    (out / "regions.json").write_text(
        json.dumps(regions_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ---------- candidate rejections ----------
    rej_md = ["# Candidate rejections (main protocol)", ""]
    rej_md.append("## Design decisions")
    rej_md.append("- **Region unit = HUC6 + largest connected task component** (not raw HUC8).")
    rej_md.append("- **K = (0, 1, 3, 5)** only; K=10 dropped (no candidate component supports it).")
    rej_md.append("- **Base model = H2X** (`transport_enc`); H2 is ablation only.")
    rej_md.append("")
    rej_md.append("## Component-choice note (HUC6 103001)")
    rej_md.append(
        "- `103001` largest component = 10 stations / 17 K=5 months; a secondary 7-station "
        "component offers 35 K=5 months. Main protocol keeps the **largest** component "
        "(more query sites per month, fewer task-months). Documented, not a blocker."
    )
    rej_md.append("")
    rej_md.append("## All HUC6 components ranked 0 — eligibility table")
    rej_md.append("")
    rej_md.append("| HUC6 | n_comp | n_st | K=5 months | eligible | note |")
    rej_md.append("|------|-------:|-----:|-----------:|----------|------|")
    huc6_top = audit[(audit.scale == "huc6") & (audit.component_rank == 0)].sort_values("code")
    for row in huc6_top.itertuples(index=False):
        ok = row.n_stations_component >= MIN_COMPONENT_STATIONS and row.months_ge5_support >= MIN_K5_MONTHS
        code = row.code
        note = []
        if code in PRIMARY_HUC6:
            note.append("PRIMARY")
        elif code in BACKUP_HUC6:
            note.append("backup")
        if not ok:
            note.append("too few stations/months")
        if code == "103001":
            note.append("largest-comp preferred over 7st/35mo comp")
        rej_md.append(
            f"| `{code}` | {int(row.n_components)} | {int(row.n_stations_component)} | "
            f"{int(row.months_ge5_support)} | {'yes' if ok else 'no'} | {', '.join(note) or '—'} |"
        )
    rej_md.append("")
    rej_md.append("## HUC8 not used as primary")
    for code in SECONDARY_HUC8:
        sub = audit[(audit.scale == "huc8") & (audit.code == code)]
        ncomp = int(sub.n_components.iloc[0]) if len(sub) else 0
        rej_md.append(
            f"- HUC8 `{code}`: {ncomp} components; kept as **secondary case study** only "
            f"(induced subgraph not a single tributary)."
        )
    rej_md.append("")
    rej_md.append("## HUC6 backup / excluded")
    for code in BACKUP_HUC6:
        r = region_defs.get(code)
        sub = audit[(audit.scale == "huc6") & (audit.code == code) & (audit.component_rank == 0)]
        if len(sub):
            rej_md.append(
                f"- HUC6 `{code}`: backup (component n={int(sub.n_stations_component.iloc[0])}, "
                f"months_ge5={int(sub.months_ge5_support.iloc[0])})."
            )
    for line in rejections:
        rej_md.append(f"- {line}")
    rej_md.append("")
    rej_md.append("## K=10")
    rej_md.append(
        "- Removed from main protocol: only `10300101` ever offered K=10 (1 month, 2 query cells). "
        "HUC6 components are 7–10 stations, so K=10 + min_query is infeasible without coarsening to HUC4."
    )
    (out / "candidate_rejections.md").write_text("\n".join(rej_md) + "\n", encoding="utf-8")

    # ---------- protocol.json ----------
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        commit, dirty = "unknown", True

    protocol = {
        "version": "kshot_protocol_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "dataset_sha256": sha256_file(Path(args.dataset)),
        "nodes": args.nodes,
        "nodes_sha256": sha256_file(Path(args.nodes)),
        "regions_primary": PRIMARY_HUC6,
        "region_rule": "hide all HUC6 labels; evaluate largest connected component inside HUC6",
        "k_list": list(K_LIST),
        "min_query": MIN_QUERY_DEFAULT,
        "task_seeds": list(TASK_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "base_model": {
            "primary": "H2X",
            "config": BASE_MODEL_CONFIG,
            "ablation": {
                "name": "H2",
                "architecture": "transport",
                "hidden": 64,
                "layers": 2,
                "dropout": 0.1,
                "lr": 0.001,
                "max_epochs": 200,
                "patience": 20,
                "env_encoder": False,
            },
        },
        "success_gate": {
            "primary_endpoint": "K=5",
            "metric": "paired ΔMAE vs each baseline on fixed query cells",
            "bootstrap": "cluster by task-month, 95% CI upper bound of ΔMAE < 0",
            "effect_size": "MAE reduction vs best simple baseline >= 10%",
            "baselines_to_beat": ["K=0_base", "local_mean", "mean_bias", "IDW", "nearest"],
            "shuffle": "gnn_true must beat gnn_shuf_values and gnn_shuf_sites significantly",
            "regions": "at least 4 of 5 primary basins pass; no basin may worsen >5% and still claim overall success",
            "curve": "MAE(K=5) < MAE(K=0) and MAE(K=5) <= MAE(K=1)",
            "fallback_story": "if only 1-2 basins pass, report river-type transferability instead of universal claim",
        },
        "judge_requires_support_encoder": {
            "must_exceed_simple_baselines": True,
            "cross_fitted_base": True,
            "analytic_blend_role": "baseline only, not model claim",
        },
        "huc8_role": "secondary case study only",
        "k10_role": "appendix / coarse-scale exploratory only",
        "code_commit": commit,
        "workspace_dirty": dirty,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ---------- manifest ----------
    script_path = Path(__file__).resolve()
    manifest = {
        "created_at": protocol["created_at"],
        "code_commit": commit,
        "workspace_dirty": dirty,
        "generator_script": str(script_path),
        "generator_script_sha256": sha256_file(script_path),
        "dataset_sha256": protocol["dataset_sha256"],
        "nodes_sha256": protocol["nodes_sha256"],
        "task_hashes": all_task_hash,
        "artifact_hashes": {
            str(p.relative_to(out)): sha256_file(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {out}")
    print("primary eligible:", [r["code"] for r in regions_json["primary"] if r.get("eligible")])
    print("primary total listed:", [r["code"] for r in regions_json["primary"]])


if __name__ == "__main__":
    main()
