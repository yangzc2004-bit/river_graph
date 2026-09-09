"""Build the training dataset (Milestone 2).

Stages: select graph nodes -> fetch WQP observations (resumable) ->
fetch daily discharge -> monthly aggregation -> dataset .pt + report.

    python scripts/build_dataset.py --smoke 50   # small-sample pipeline test
    python scripts/build_dataset.py              # full run (all graph nodes)
"""

import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.data.aggregate import to_monthly
from river_graph.data.nwis import fetch_daily_discharge, load_daily_discharge
from river_graph.data.wqp import (
    extract_covariate_obs,
    extract_doc_obs,
    fetch_station_results,
    load_station_results,
)
from river_graph.dataset import build_dataset, save_dataset

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def select_sites(nodes: pd.DataFrame, edges: pd.DataFrame, smoke: int | None) -> list[str]:
    """Smoke mode: best-sampled sites inside the largest connected component."""
    if smoke is None:
        return sorted(nodes["site_no"])
    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    largest = max(nx.weakly_connected_components(g), key=len)
    inv = pd.read_csv(PROCESSED / "doc_site_inventory.csv", dtype={"site_no": str})
    top = (
        inv[inv["site_no"].isin(largest)]
        .nlargest(smoke, "count_nu")["site_no"]
        .tolist()
    )
    print(f"smoke mode: top {len(top)} DOC-sampled sites in largest component "
          f"({len(largest)} nodes)")
    return top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="limit to N sites")
    args = ap.parse_args()

    with open(PROCESSED / "mississippi_graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    nodes, edges = graph["nodes"], graph["edges"]
    sites = select_sites(nodes, edges, args.smoke)
    nodes = nodes[nodes["site_no"].isin(sites)].reset_index(drop=True)
    print(f"sites: {len(sites)}")

    # --- fetch observations (resumable, per-station cache) ---
    cache = RAW / "wqp_results"
    results = []
    failed = []
    for i, site_no in enumerate(sites):
        path = fetch_station_results(site_no, cache)
        if path is None:
            failed.append(site_no)
            continue
        results.append(load_station_results(path))
        if (i + 1) % 25 == 0:
            print(f"  fetched {i + 1}/{len(sites)}")
    if failed:
        print(f"WQP failed for {len(failed)} sites (rerun to retry): {failed[:10]}")

    # --- validation 1: DOC field consistency ---
    allres = pd.concat(results, ignore_index=True)
    oc = allres[allres["variable"] == "organic_carbon"]
    print("\n[validation 1] organic carbon rows by fraction/unit:")
    print(oc.groupby(["fraction", "unit"], dropna=False).size().nlargest(8))
    doc_obs = extract_doc_obs(allres)
    print(f"DOC obs (dissolved, mg/L, uncensored): {len(doc_obs):,} "
          f"across {doc_obs['site_no'].nunique()} sites")

    # --- monthly aggregation ---
    monthly_doc = to_monthly(doc_obs, "doc")
    temp_obs = extract_covariate_obs(allres, "temperature")
    monthly_temp = to_monthly(temp_obs, "temperature")
    print("\n[validation 2] monthly DOC labels per site:")
    print(monthly_doc.groupby("site_no").size().describe().round(1))

    # --- discharge ---
    fetch_daily_discharge(sites, RAW / "nwis_dv")
    dv = load_daily_discharge(RAW / "nwis_dv")
    dv = dv[dv["site_no"].isin(sites)]
    monthly_flow = to_monthly(dv, "discharge_cfs").rename(
        columns={"discharge_cfs": "discharge"}
    )

    # --- validation 3: missingness funnel ---
    n_feat = monthly_flow["site_no"].nunique()
    print("\n[validation 3] funnel:")
    print(f"  selected sites:            {len(sites)}")
    print(f"  with >=1 DOC obs:          {doc_obs['site_no'].nunique()}")
    print(f"  with >=12 monthly labels:  "
          f"{(monthly_doc.groupby('site_no').size() >= 12).sum()}")
    print(f"  with discharge data:       {n_feat}")
    print(f"  with temperature data:     {monthly_temp['site_no'].nunique()}")

    # --- assemble dataset ---
    if monthly_doc.empty:
        raise RuntimeError("no DOC labels extracted; check validation 1 output above")
    months = pd.date_range(
        monthly_doc["month"].min(), monthly_doc["month"].max(), freq="MS"
    )

    # H2 additions: node regime + edge transport attributes (v03)
    reach = pd.read_csv(PROCESSED / "reach_attributes.csv", dtype={"site_no": str})
    reach = reach.set_index("site_no")
    in_deg = edges.groupby("target").size()
    regime = []
    for s in nodes["site_no"]:
        r = reach.loc[s]
        regime.append([
            r["streamorde"], np.log1p(r["totdasqkm"]), r["slope"],
            float(in_deg.get(s, 0) == 0),  # is_headwater
        ])
    ef = pd.read_csv(PROCESSED / "edge_features.csv", dtype={"source": str, "target": str})
    ef = edges.merge(ef, on=["source", "target"], how="left")
    edge_attr = np.nan_to_num(ef[[
        "hop_dist", "geo_dist_deg", "target_lengthkm",
        "target_totdasqkm", "target_slope", "target_streamorde",
    ]].values.astype(float))
    edge_attr[:, 2] = np.log1p(edge_attr[:, 2])  # lengthkm
    edge_attr[:, 3] = np.log1p(edge_attr[:, 3])  # drainage area

    dataset = build_dataset(
        nodes, edges, monthly_doc,
        {"temperature": monthly_temp, "discharge": monthly_flow},
        months,
        edge_attr=edge_attr,
        regime=np.asarray(regime, dtype=np.float32),
    )
    tag = f"_smoke{args.smoke}" if args.smoke else ""
    out = PROCESSED / f"mississippi_graph_v03{tag}.pt"
    save_dataset(dataset, out)
    print(f"\nsaved {out}: N={len(dataset['site_no'])}, T={len(dataset['months'])}, "
          f"E={dataset['edge_index'].shape[1]}, "
          f"label coverage={dataset['y_mask'].float().mean():.1%}")


if __name__ == "__main__":
    main()
