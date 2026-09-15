"""Build the h3a_v1 internal validation masks (T03).

    python scripts/build_h3_masks.py

Reads the frozen masks in experiments/masks/ and writes
experiments/h3a_v1/masks/*.npz plus experiments/h3a_v1/split_manifest.json.
The outer test arrays are copied unchanged; nothing in experiments/masks is
touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from river_graph.experiments.h3_masks import (
    KEY_SCENARIOS,
    build_masks,
    component_report,
    write_masks,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v05.pt")
    ap.add_argument("--source-masks", default="experiments/masks")
    ap.add_argument("--out", default="experiments/h3a_v1r2/masks")
    ap.add_argument("--edges", default="data/processed/graph_edges.csv")
    args = ap.parse_args()

    dataset = torch.load(args.dataset, weights_only=False)
    y_mask = dataset["y_mask"].numpy()
    sites = list(dataset["site_no"])
    edges = pd.read_csv(args.edges, dtype=str)

    masks, manifest = build_masks(Path(args.source_masks), y_mask, edges, sites)
    manifest["dataset_path"] = args.dataset
    manifest["edge_file"] = args.edges
    manifest["graph_cv_report"] = component_report(edges, sites)
    hashes = write_masks(masks, manifest, Path(args.out))

    print(f"wrote {len(masks)} masks to {args.out}")
    for name in list(KEY_SCENARIOS) + list(manifest["diagnostic_scenarios"]):
        entry = manifest["masks"][name]
        roles = " ".join(f"{k}={v}" for k, v in entry["roles"].items())
        visible = ",".join(entry["visible_roles"])
        print(f"  {name:24s} {roles}  visible=[{visible}] sha={hashes[name][:12]}")
    e3 = next(k for k in manifest["masks"] if k.startswith("e3_internal"))
    cv = manifest["masks"][e3]["cv_report"]
    print(
        f"graph: {cv['n_components']} components, largest={cv['largest_component_size']}, "
        f"isolated={cv['n_isolated_stations']}"
    )
    print(
        f"{e3}: {cv['n_selected_stations']} val stations in "
        f"{cv['selected_components_nonempty']} components "
        f"({cv['selected_in_largest_component']} in the largest, "
        f"{cv['selected_isolated']} isolated)"
    )
    print(f"manifest: {Path(args.out).parent / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
