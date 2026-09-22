#!/usr/bin/env python3
"""Why did support_encoder_v2 fail the gate-direction check?

Three diagnostics on a re-trained encoder (H2X bases come from the seed42
cache, so this is cheap):

D1  attention entropy — is the support attention uniform (geometry ignored)
    or collapsed onto one support (set structure ignored)?
D2  feature permutation sensitivity — does Δ move when the support *values*
    (log1p_y / resid) are ablated or shuffled? Compare against ablating the
    query-side log1p_base as a reference scale.
D3  gradient share — where does learning signal go: content path (mlp_sup /
    ctx), query path (hq), or the geometry bias?

Run on one failing basin (510020) and one healthy basin (102701).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "kenc", ROOT / "scripts/run_kshot_encoder_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attention_weights(model, ep) -> np.ndarray:
    """(K, Q) softmax weights — mirrors SupportSetEncoder.forward."""
    sup = ep["support_tokens"]
    qry = ep["query_tokens"]
    if sup.shape[0] == 0:
        return np.zeros((0, qry.shape[0]))
    with torch.no_grad():
        h = model.mlp_sup(sup)
        hq = model.mlp_qry(qry)
        hop_e = model.hop_emb(ep["support_hop"].clamp(min=0, max=model.hop_cap + 1))
        dir_e = model.dir_emb(ep["support_dir"].clamp(min=0, max=2))
        d_order = (ep["support_order"].unsqueeze(1) - ep["query_order"].unsqueeze(0)).unsqueeze(-1)
        d_area = (ep["support_area"].unsqueeze(1) - ep["query_area"].unsqueeze(0)).unsqueeze(-1)
        d_regime = (ep["support_regime"].unsqueeze(1) - ep["query_regime"].unsqueeze(0)).abs()
        bias_in = torch.cat([hop_e, dir_e, d_order, d_area, d_regime], dim=-1)
        bias = model.bias_lin(bias_in).squeeze(-1)
        logits = torch.einsum("kd,qd->kq", h, hq) / (h.shape[-1] ** 0.5) + bias
        return torch.softmax(logits, dim=0).numpy()


def delta_of(model, ep) -> np.ndarray:
    with torch.no_grad():
        return model(
            ep["support_tokens"], ep["support_dir"], ep["support_hop"],
            ep["support_order"], ep["support_area"], ep["support_regime"],
            ep["query_tokens"], ep["query_order"], ep["query_area"],
            ep["query_regime"],
        ).numpy()


def with_support(model, ep) -> np.ndarray:
    return delta_of(model, ep)


def d1_attention(episodes, model) -> dict:
    ents, maxes, ks = [], [], []
    for ep in episodes:
        a = attention_weights(model, ep)
        k = a.shape[0]
        if k == 0:
            continue
        p = np.clip(a, 1e-12, 1.0)
        ent = -(p * np.log(p)).sum(axis=0) / np.log(k)  # (Q,) normalized
        ents.append(ent.mean())
        maxes.append(a.max(axis=0).mean())
        ks.append(k)
    return {
        "norm_entropy_mean": float(np.mean(ents)) if ents else np.nan,
        "max_alpha_mean": float(np.mean(maxes)) if maxes else np.nan,
        "n_episodes": len(ents),
        "k_mean": float(np.mean(ks)) if ks else np.nan,
    }


def d2_permutation(episodes, model, seed: int = 0) -> dict:
    """|Δ change| under targeted ablations, vs query-side reference."""
    rng = np.random.default_rng(seed)
    keys_zero_resid = []
    keys_shuf_resid = []
    keys_shuf_values = []
    keys_zero_base = []
    keys_zero_support_all = []
    base_scale = []
    for ep in episodes:
        d0 = with_support(model, ep)
        base_scale.append(float(np.std(d0)))

        # (a) zero the resid column (index 1) of every support token
        ep_a = dict(ep)
        tok = ep["support_tokens"].clone()
        tok[:, 1] = 0.0
        ep_a["support_tokens"] = tok
        keys_zero_resid.append(float(np.mean(np.abs(d0 - with_support(model, ep_a)))))

        # (b) permute resid across supports (values stay on their sites in y,
        #     but the encoder's residual feature is scrambled)
        ep_b = dict(ep)
        tok = ep["support_tokens"].clone()
        k = tok.shape[0]
        if k > 1:
            tok[:, 1] = tok[torch.tensor(rng.permutation(k)), 1]
        ep_b["support_tokens"] = tok
        keys_shuf_resid.append(float(np.mean(np.abs(d0 - with_support(model, ep_b)))))

        # (c) shuffle BOTH log1p_y and resid across supports (value shuffle
        #     at the feature level)
        ep_c = dict(ep)
        tok = ep["support_tokens"].clone()
        if k > 1:
            perm = torch.tensor(rng.permutation(k))
            tok[:, 0] = tok[perm, 0]
            tok[:, 1] = tok[perm, 1]
        ep_c["support_tokens"] = tok
        keys_shuf_values.append(float(np.mean(np.abs(d0 - with_support(model, ep_c)))))

        # (d) reference: zero the query log1p_base (column 0 of query token)
        ep_d = dict(ep)
        qt = ep["query_tokens"].clone()
        qt[:, 0] = 0.0
        ep_d["query_tokens"] = qt
        keys_zero_base.append(float(np.mean(np.abs(d0 - with_support(model, ep_d)))))

        # (e) zero all support content tokens (keep geometry)
        ep_e = dict(ep)
        tok = ep["support_tokens"].clone()
        tok[:] = 0.0
        ep_e["support_tokens"] = tok
        keys_zero_support_all.append(float(np.mean(np.abs(d0 - with_support(model, ep_e)))))

    return {
        "mean_abs_d_delta_zero_resid": float(np.mean(keys_zero_resid)),
        "mean_abs_d_delta_shuf_resid": float(np.mean(keys_shuf_resid)),
        "mean_abs_d_delta_shuf_values": float(np.mean(keys_shuf_values)),
        "mean_abs_d_delta_zero_support_all": float(np.mean(keys_zero_support_all)),
        "mean_abs_d_delta_zero_query_base_REF": float(np.mean(keys_zero_base)),
        "mean_abs_delta_scale": float(np.mean(base_scale)),
    }


def d3_grad_share(episodes, model) -> dict:
    model.zero_grad()
    n = 0
    for ep in episodes[:32]:
        if ep["support_tokens"].shape[0] == 0:
            continue
        delta = model(
            ep["support_tokens"], ep["support_dir"], ep["support_hop"],
            ep["support_order"], ep["support_area"], ep["support_regime"],
            ep["query_tokens"], ep["query_order"], ep["query_area"],
            ep["query_regime"],
        )
        delta.sum().backward()
        n += 1
    if n == 0:
        return {}
    groups = {"mlp_sup": 0.0, "mlp_qry": 0.0, "bias_lin": 0.0, "hop_emb": 0.0,
              "dir_emb": 0.0, "mlp_out": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.norm())
        for key in groups:
            if name.startswith(key):
                groups[key] += g ** 2
    norms = {k: np.sqrt(v) for k, v in groups.items()}
    total = sum(norms.values()) + 1e-12
    return {f"grad_share_{k}": v / total for k, v in norms.items()} | {
        f"grad_norm_{k}": v for k, v in norms.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["510020", "102701"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-epochs", type=int, default=200)
    ap.add_argument("--out", default="experiments/kshot_encoder_v2/diagnose_gate_direction.csv")
    args = ap.parse_args()

    mod = load_runner()
    dataset = mod.load_dataset(str(ROOT / "data/processed/mississippi_graph_graphfix_st357.pt"))
    y = np.asarray(dataset["y"])
    y_mask = np.asarray(dataset["y_mask"])
    regime = np.asarray(dataset["regime"], dtype=np.float64)
    months = [str(m) for m in dataset["months"]]
    sites = [str(s) for s in dataset["site_no"]]
    nodes = pd.read_csv(
        ROOT / "data/processed/graph_nodes_graphfix_st357.csv", dtype={"site_no": str}
    )
    geom = mod.GraphGeometry(dataset, ROOT / "data/processed/reach_attributes.csv")
    row_by_basin = {b: mod.basin_rows(sites, nodes, b) for b in mod.PRIMARY}
    cache_dir = ROOT / "experiments/kshot_encoder_v2/base_preds"

    rows = []
    for target in args.targets:
        sources = [b for b in mod.PRIMARY if b != target]
        hide_r = list(row_by_basin[target])
        base_r = mod.get_base_pred(
            cache_dir, f"{target}_excl_s{args.seed}", dataset, hide_r,
            args.seed, 200, 20, False,
        )
        src_rows = sorted({i for s in sources for i in row_by_basin[s]})
        obs_cells = [
            int(f) for f in np.flatnonzero(y_mask.ravel())
            if (f // y.shape[1]) in set(src_rows)
        ]
        std = mod.fit_standardizers(y, base_r, regime, obs_cells, geom)
        episodes = []
        for src in sources:
            base_src = mod.get_base_pred(
                cache_dir, f"{target}_{src}_excl_s{args.seed}", dataset,
                hide_r + list(row_by_basin[src]), args.seed, 200, 20, False,
            )
            src_tasks = mod.label_tasks(
                mod.make_support_query_tasks(
                    y_mask, row_by_basin[src], months,
                    k_list=(1, 3, 5), min_query=2, seed=42,
                ),
                src,
            )
            episodes.extend(
                mod.episodes_from_tasks(src_tasks, y, base_src, regime, geom, months, std)
            )
        print(f"[{target}] training encoder on {len(episodes)} episodes", flush=True)
        enc = mod.train_encoder(episodes, seed=args.seed, max_epochs=args.encoder_epochs)

        # sanity: training-set residual of the encoder (did it fit sources at all?)
        train_resid = []
        for ep in episodes[:200]:
            d = delta_of(enc, ep)
            train_resid.append(float(np.mean(np.abs(ep["base_log"].numpy() + d - ep["y_log"].numpy()))))
        row = {"target": target, "n_episodes": len(episodes),
               "train_log_mae_with_delta": float(np.mean(train_resid))}
        row.update(d1_attention(episodes, enc))
        row.update(d2_permutation(episodes, enc))
        row.update(d3_grad_share(episodes, enc))
        rows.append(row)
        print(pd.Series(row).to_string(), flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
