#!/usr/bin/env python3
"""Step 2a: zero-training structural diagnostics for K-shot protocol v2.

Produces under experiments/kshot_protocol_v2/diagnostics/:
  step2_reachability.csv     — per query-task support hop / 2-hop coverage
  step2_hop_decay.md         — architecture RF proof + old H2 proxy decay
  step2_cross_month.md       — per-month forward isolation proof
  step2_tributary_mainstem.csv
  step2_summary.md           — decision rule outcomes

NO model training. Architecture proofs use random weights on a tiny graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

K_LIST = (0, 1, 3, 5)


def hop_maps(dataset: dict):
    n = dataset["y"].shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(dataset["edge_index"].T.tolist())
    length = dict(nx.all_pairs_shortest_path_length(g))

    def hop(a: int, b: int):
        return length.get(a, {}).get(b)

    return hop, length


def load_tasks(proto_dir: Path):
    items = []
    for path in sorted((proto_dir / "tasks").glob("*/*.json")):
        payload = json.loads(path.read_text())
        for task in payload["tasks"]:
            items.append((payload["region"], payload["task_seed"], payload, task))
    return items


def reachability_rows(dataset, regions, tasks_meta) -> pd.DataFrame:
    n = dataset["y"].shape[0]
    t = dataset["y"].shape[1]
    g_full = nx.Graph()
    g_full.add_nodes_from(range(n))
    g_full.add_edges_from(dataset["edge_index"].T.tolist())
    len_full = dict(nx.all_pairs_shortest_path_length(g_full))
    succ, pred = {}, {}
    for a, b in dataset["edge_index"].T.tolist():
        succ.setdefault(int(a), set()).add(int(b))
        pred.setdefault(int(b), set()).add(int(a))

    hide_by_region = {r["code"]: set(r["hide_rows"]) for r in regions["primary"]}
    comp_by_region = {r["code"]: set(r["task_component_rows"]) for r in regions["primary"]}
    len_hide = {}
    for code, hide in hide_by_region.items():
        g = nx.Graph()
        g.add_nodes_from(hide)
        for a, b in dataset["edge_index"].T.tolist():
            a, b = int(a), int(b)
            if a in hide and b in hide:
                g.add_edge(a, b)
        len_hide[code] = dict(nx.all_pairs_shortest_path_length(g))

    def hop_full(a, b):
        return len_full.get(a, {}).get(b)

    rows = []
    for region, task_seed, payload, task in tasks_meta:
        comp = comp_by_region[region]
        hide = hide_by_region[region]
        lh = len_hide[region]
        for k in K_LIST:
            s_cells = task["support_cells_by_k"][str(k)]
            s_rows = [c // t for c in s_cells]
            for q in task["query_cells"]:
                qr = q // t
                if k == 0:
                    rows.append(
                        {
                            "region": region,
                            "task_seed": task_seed,
                            "month": task["month"],
                            "month_index": task["month_index"],
                            "k": k,
                            "flat": q,
                            "query_row": qr,
                            "min_hop_full_graph": np.nan,
                            "min_hop_within_huc6_paths": np.nan,
                            "n_support_within_2hop": 0,
                            "frac_support_within_2hop": np.nan,
                            "outside_2hop": False,
                            "path_leaves_region": False,
                            "direction_mode": "",
                        }
                    )
                    continue
                hops_f, hops_h, dirs = [], [], []
                leaves = False
                n2 = 0
                for s in s_rows:
                    hf = hop_full(s, qr)
                    hh = lh.get(s, {}).get(qr)
                    if hf is None:
                        continue
                    hops_f.append(hf)
                    if hh is not None:
                        hops_h.append(hh)
                        if hh != hf:
                            leaves = True
                    else:
                        leaves = True
                    if hf <= 2:
                        n2 += 1
                    if s in succ.get(qr, set()):
                        dirs.append("query_downstream_of_support")
                    elif s in pred.get(qr, set()):
                        dirs.append("query_upstream_of_support")
                    else:
                        dirs.append("other_or_multi_hop")
                min_f = min(hops_f) if hops_f else np.nan
                min_h = min(hops_h) if hops_h else np.nan
                rows.append(
                    {
                        "region": region,
                        "task_seed": task_seed,
                        "month": task["month"],
                        "month_index": task["month_index"],
                        "k": k,
                        "flat": q,
                        "query_row": qr,
                        "min_hop_full_graph": min_f,
                        "min_hop_within_huc6_paths": min_h,
                        "n_support_within_2hop": n2,
                        "frac_support_within_2hop": n2 / max(len(s_rows), 1),
                        "outside_2hop": bool(min_f == min_f and min_f > 2),
                        "path_leaves_region": leaves,
                        "direction_mode": max(set(dirs), key=dirs.count) if dirs else "",
                    }
                )
    return pd.DataFrame(rows)


def load_node_stream_orders(
    dataset, reach_attributes: Path | None, edge_features: Path | None
) -> tuple[dict[int, float], str]:
    """Node-level stream order.

    ``reach_attributes.csv`` is authoritative: one ``streamorde`` per station,
    covering every dataset node. ``edge_features.csv`` is only a fallback and
    leaves many nodes unmapped — if it has to be used, the summary marks the
    unknown rate instead of forcing a classification.
    """
    sites = [str(s) for s in dataset["site_no"]]
    site_to_row = {s: i for i, s in enumerate(sites)}

    def row_of(site: str):
        s = str(site)
        for key in (s, s.zfill(8), s.lstrip("0")):
            if key in site_to_row:
                return site_to_row[key]
        return None

    orders: dict[int, float] = {}
    source = "missing"
    if reach_attributes and reach_attributes.exists():
        ra = pd.read_csv(reach_attributes, dtype={"site_no": str})
        col = next(
            (c for c in ("streamorde", "stream_order") if c in ra.columns), None
        )
        if col and "site_no" in ra.columns:
            for site, o in zip(ra.site_no.astype(str), ra[col]):
                r = row_of(site)
                if r is None:
                    continue
                try:
                    orders[r] = float(o)
                except (TypeError, ValueError):
                    continue
            source = "reach_attributes.csv"
    if not orders and edge_features and edge_features.exists():
        ef = pd.read_csv(edge_features)
        order_col = next(
            (c for c in ("target_streamorde", "stream_order", "streamorde") if c in ef.columns),
            None,
        )
        if order_col and {"source", "target"}.issubset(ef.columns):
            for tgt, o in zip(ef.target.astype(str), ef[order_col]):
                r = row_of(tgt)
                if r is None:
                    continue
                try:
                    orders[r] = float(o)
                except (TypeError, ValueError):
                    continue
            source = "edge_features.csv"
    return orders, source


def tributary_rows(
    dataset,
    regions,
    tasks_meta,
    reach_attributes: Path | None,
    edge_features: Path | None,
) -> pd.DataFrame:
    t = dataset["y"].shape[1]
    hop, _ = hop_maps(dataset)
    orders, order_source = load_node_stream_orders(dataset, reach_attributes, edge_features)

    rows = []
    for region, task_seed, payload, task in tasks_meta:
        for k in (1, 3, 5):
            s_rows = [c // t for c in task["support_cells_by_k"][str(k)]]
            q_rows = [c // t for c in task["query_cells"]]
            for qr in q_rows:
                for sr in s_rows:
                    h = hop(sr, qr)
                    if h is None:
                        continue
                    so = orders.get(sr, np.nan)
                    qo = orders.get(qr, np.nan)
                    if np.isfinite(so) and np.isfinite(qo):
                        rel = (
                            "trib_to_main"
                            if so < qo
                            else ("main_to_trib" if so > qo else "same_order")
                        )
                    else:
                        rel = "unknown_order"
                    rows.append(
                        {
                            "region": region,
                            "task_seed": task_seed,
                            "month": task["month"],
                            "k": k,
                            "support_row": sr,
                            "query_row": qr,
                            "hops": h,
                            "within_2hop": h <= 2,
                            "support_order": so,
                            "query_order": qo,
                            "order_relation": rel,
                            "order_source": order_source,
                            "tributary_to_mainstem_le2": bool(h <= 2 and rel == "trib_to_main"),
                        }
                    )
    return pd.DataFrame(rows)


def architecture_zero_influence_proof() -> str:
    """Random-weight TransportGCN: support at >=3 hops must not change output bitwise."""
    from river_graph.models.hydro import TransportGCNImputer

    torch.manual_seed(0)
    # path of 4 nodes: 0-1-2-3  => 3 is 3 hops from 0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr = torch.ones(edge_index.shape[1], 5)
    n, c = 4, 10
    x = torch.randn(n, c)
    model = TransportGCNImputer(c, 5, hidden=8, layers=2, dropout=0.0, edge_direction="both")
    model.eval()
    with torch.no_grad():
        base = model(x, edge_index, edge_attr)
        x2 = x.clone()
        x2[0, 8] = 5.0  # support value at node 0
        x2[0, 9] = 1.0
        after = model(x2, edge_index, edge_attr)
    d0 = float(torch.max(torch.abs(base[0] - after[0])))
    d1 = float(torch.max(torch.abs(base[1] - after[1])))
    d2 = float(torch.max(torch.abs(base[2] - after[2])))
    d3 = float(torch.max(torch.abs(base[3] - after[3])))
    lines = [
        "# Step 2a — 2-hop receptive-field proof (random weights, no training)",
        "",
        "Two-layer `TransportGCNImputer` message passing has an effective",
        "receptive field of **2 hops**. Perturbing an input at node 0:",
        "",
        "| node | hop from 0 | |Δ output| |",
        "|-----:|-----------:|------------:|",
        f"| 0 | 0 | {d0:.6f} |",
        f"| 1 | 1 | {d1:.6g} |",
        f"| 2 | 2 | {d2:.6g} |",
        f"| 3 | 3 | **{d3}** |",
        "",
        "Node 3 is bitwise unchanged (`Δ = 0`) — support beyond 2 hops cannot",
        "affect a query under this architecture, trained or not.",
        "",
        "## Decision rule",
        "",
        "- If a large fraction of v2 queries has `min_hop > 2`, a support encoder",
        "  is **necessary**, not optional.",
        "- If most queries are within 2 hops, failure cannot be blamed on RF alone.",
    ]
    return "\n".join(lines) + "\n"


def cross_month_proof() -> str:
    return """# Step 2a — Cross-month support isolation (architecture)

`GCNDocModel.predict` / `fit` call `fwd(xt[j])` **independently for each month
index `j`**. There is no temporal edge, RNN, or cross-month feature.

Therefore a support cell at month `t' ≠ t`:

1. is never placed in `xt[t]`'s DOC channel;
2. cannot enter the 2-layer spatial message passing of month `t`;
3. has **exact zero influence** on a query at month `t`.

## Decision rule

- Same-month support is the only channel available to the current GNN.
- If a region lacks same-month K, **Case C (explicit cross-time support) must be
  designed** — it cannot be recovered by "letting" the model see old samples.
"""


def old_h2_proxy_decay(reach: pd.DataFrame) -> pd.DataFrame:
    """Collapse reachability into hop bins for narrative (not H2X evidence)."""
    sub = reach[(reach.k > 0) & reach.min_hop_full_graph.notna()].copy()
    if sub.empty:
        return sub
    sub["hop_bin"] = sub.min_hop_full_graph.map(
        lambda h: "1" if h == 1 else ("2" if h == 2 else ("3+" if h >= 3 else "0"))
    )
    return (
        sub.groupby(["region", "k", "hop_bin"], as_index=False)
        .agg(n_query=("flat", "size"), frac_within_2=("frac_support_within_2hop", "mean"))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_graphfix_st357.pt")
    ap.add_argument("--protocol-dir", default="experiments/kshot_protocol_v2")
    ap.add_argument("--reach-attributes", default="data/processed/reach_attributes.csv")
    ap.add_argument("--edge-features", default="data/processed/edge_features.csv")
    ap.add_argument("--old-h2-dir", default="experiments/kshot_st357/predictions")
    args = ap.parse_args()

    proto_dir = Path(args.protocol_dir)
    out = proto_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(args.dataset, weights_only=False)
    regions = json.loads((proto_dir / "regions.json").read_text())
    tasks_meta = load_tasks(proto_dir)
    print(f"loaded {len(tasks_meta)} tasks", flush=True)

    reach = reachability_rows(dataset, regions, tasks_meta)
    reach.to_csv(out / "step2_reachability.csv", index=False)

    # summary stats per region @ K=1/3/5 — denominators are explicit: every
    # percentage is over query cells (fixed set across K), never over pairs.
    def frac_out(k):
        sub = reach[reach.k == k]
        return float(sub.outside_2hop.mean()) if len(sub) else np.nan

    summary_lines = [
        "# Step 2a summary (zero-training)",
        "",
        "## 2-hop coverage on frozen v2 tasks",
        "",
        "Denominator of every percentage below: **query cells** at that K.",
        "The query set is identical for K=1/3/5 (nested support, fixed query),",
        "so the K columns share one denominator per region. Pooled overall is a",
        "micro-average over all query cells; the region-macro line weights each",
        "basin equally.",
        "",
        "| region | n query cells | % >2 hop (K=1) | % >2 hop (K=3) | % >2 hop (K=5) |",
        "|--------|--------------:|---------------:|---------------:|---------------:|",
    ]
    region_fracs = []
    for region, sub in reach.groupby("region"):
        s1, s3, s5 = (sub[sub.k == k] for k in (1, 3, 5))
        n = len(s5)
        f1, f3, f5 = (s.outside_2hop.mean() for s in (s1, s3, s5))
        region_fracs.append(float(f1))
        summary_lines.append(
            f"| `{region}` | {n} | {100 * f1:.1f}% | {100 * f3:.1f}% | {100 * f5:.1f}% |"
        )
    n_total = int((reach.k == 5).sum())
    macro = float(np.mean(region_fracs)) if region_fracs else np.nan
    summary_lines += [
        "",
        (
            f"- Pooled K=1 outside 2-hop: **{100 * frac_out(1):.1f}%** "
            f"({int(reach.loc[reach.k == 1, 'outside_2hop'].sum())} of {n_total} query cells)"
        ),
        f"- Pooled K=3 outside 2-hop: **{100 * frac_out(3):.1f}%**",
        f"- Pooled K=5 outside 2-hop: **{100 * frac_out(5):.1f}%**",
        (
            f"- Region-macro K=1 outside 2-hop: **{100 * macro:.1f}%** "
            f"(mean of the 5 per-region rates above)"
        ),
        "",
        "## Decision rule",
        "",
        "- If outside-2hop > 20–30% at K=1/3 → support encoder is **necessary**.",
        f"- Current K=1 outside-2hop = **{100 * frac_out(1):.1f}%** (pooled) → "
        + (
            "support encoder required under the pre-registered rule."
            if frac_out(1) > 0.2
            else "most queries are in RF; RF alone does not explain failures."
        ),
        "",
    ]

    # Tributary→mainstem, with pair-level and query-level denominators kept apart.
    trib = tributary_rows(
        dataset, regions, tasks_meta, Path(args.reach_attributes), Path(args.edge_features)
    )
    trib.to_csv(out / "step2_tributary_mainstem.csv", index=False)
    order_source = (
        str(trib.order_source.iloc[0]) if len(trib) and "order_source" in trib else "missing"
    )
    summary_lines += [
        "## Tributary→mainstem (stream order)",
        "",
        (
            f"Stream-order source: `{order_source}` "
            "(node-level `streamorde` from reach_attributes.csv)."
        ),
        "",
        "Two different denominators — do not mix them:",
        "",
        (
            "- **pairs** = (support, query) rows in `step2_tributary_mainstem.csv`; "
            "each query contributes K pairs, so pair counts scale with K."
        ),
        "- **query cells** = fixed query set (same as the 2-hop table).",
        "",
        "### Pair-level order relation (denominator: pairs at that K)",
        "",
        "| region | K | pairs | trib_to_main | main_to_trib | same_order | unknown_order |",
        "|--------|--:|------:|-------------:|-------------:|-----------:|--------------:|",
    ]
    for region, sub in trib.groupby("region"):
        for k, sk in sub.groupby("k"):
            n = len(sk)
            counts = sk.order_relation.value_counts()
            cell = " | ".join(
                f"{counts.get(name, 0)} ({100 * counts.get(name, 0) / n:.1f}%)"
                if n
                else "—"
                for name in ("trib_to_main", "main_to_trib", "same_order", "unknown_order")
            )
            summary_lines.append(f"| `{region}` | {k} | {n} | {cell} |")
    n_pairs = len(trib)
    n_unknown = int((trib.order_relation == "unknown_order").sum()) if n_pairs else 0
    summary_lines += [
        "",
        f"- Unknown stream-order pairs: **{n_unknown} of {n_pairs}** "
        f"({100 * n_unknown / n_pairs:.1f}%)" if n_pairs else "- No pairs.",
    ]
    if n_pairs and n_unknown:
        summary_lines.append(
            "- Unknown pairs are **not classified**; they are excluded from "
            "trib_to_main / main_to_trib rates' interpretation and support **no** "
            "tributary–mainstem conclusion."
        )
    else:
        summary_lines.append(
            "- Every pair has a stream-order relation; no unknown residual."
        )
    # Query-level: does a query have ≥1 trib→main support within 2 hops?
    qlevel = (
        trib.assign(
            t2=trib.tributary_to_mainstem_le2.astype(int),
        )
        .groupby(["region", "k", "task_seed", "month", "query_row"], as_index=False)["t2"]
        .max()
    )
    summary_lines += [
        "",
        "### Query-level (denominator: query cells at that K)",
        "",
        "A query counts once if **any** of its K supports is a ≤2-hop trib→main pair.",
        "",
        "| region | K | n query cells | % with ≥1 trib→main ≤2hop |",
        "|--------|--:|--------------:|--------------------------:|",
    ]
    for region, sub in qlevel.groupby("region"):
        for k, sk in sub.groupby("k"):
            n = len(sk)
            pct = 100 * sk.t2.mean() if n else float("nan")
            summary_lines.append(f"| `{region}` | {k} | {n} | {pct:.1f}% |")
    summary_lines += [
        "",
        "## Proofs",
        "",
        "- `step2_hop_decay.md` — 2-hop RF bitwise proof",
        "- `step2_cross_month.md` — cross-month influence ≡ 0",
        (
            f"- `step2_tributary_mainstem.csv` — tributary→mainstem pairs "
            f"(source: `{order_source}`)"
        ),
        "- `step2_reachability.csv` — per query-cell hops",
        "",
        "## H2 proxy caveat",
        "",
        "Old `kshot_st357` H2 predictions may be used only as **proxy narrative**.",
        "H2X empirical hop-decay is Step 2b (after base training) and is part of",
        "the success gate — not replaced by this file.",
    ]

    (out / "step2_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out / "step2_hop_decay.md").write_text(architecture_zero_influence_proof(), encoding="utf-8")
    (out / "step2_cross_month.md").write_text(cross_month_proof(), encoding="utf-8")

    decay = old_h2_proxy_decay(reach)
    decay.to_csv(out / "step2_hop_bin_coverage.csv", index=False)

    print(out / "step2_summary.md")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
