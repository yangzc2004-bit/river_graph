"""Build the h3a_v1 internal validation masks (T03).

    python scripts/build_h3_masks.py --register-source-grid \
        --grid-from experiments/h3a_v1/split_manifest.json
    python scripts/build_h3_masks.py --dataset data/processed/mississippi_graph_v06.pt

Reads the frozen masks in experiments/masks/ and writes
experiments/h3a_v1r2/masks/*.npz plus experiments/h3a_v1r2/split_manifest.json.
Nothing in experiments/masks is touched.

The frozen masks are encoded as flat cell indices, which only mean something
together with the grid they were built on (station_index * n_months +
month_index). --register-source-grid records that grid once, next to the frozen
masks. Every later build checks it: an identical grid is reused as is, a
different grid is migrated by decoding to station/month and re-encoding, and an
incompatible station order is refused outright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from river_graph.experiments.h3_masks import (
    KEY_SCENARIOS,
    build_masks,
    component_report,
    write_masks,
    write_source_grid,
)


def register(args) -> None:
    dataset = torch.load(args.dataset, weights_only=False)
    sites = list(dataset["site_no"])
    grid = None
    if args.grid_from:
        # the dataset the frozen masks were built on may be gone; its grid is
        # still recorded in the manifest that describes those masks
        grid = json.loads(Path(args.grid_from).read_text(encoding="utf-8"))["dataset"]
        print(f"declaring the grid from {args.grid_from}: "
              + json.dumps({k: grid[k] for k in
                            ("n_stations", "n_months", "observed_cells")}))
    record = write_source_grid(
        Path(args.source_masks), sites, grid=grid, y_mask=dataset["y_mask"].numpy()
    )
    print("registered source grid:", json.dumps(record, ensure_ascii=False))
    print("path:", Path(args.source_masks) / "source_grid.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v07.pt")
    ap.add_argument("--source-masks", default="experiments/masks")
    ap.add_argument("--out", default="experiments/h3a_v1r3/masks")
    ap.add_argument("--edges", default="data/processed/graph_edges.csv")
    ap.add_argument("--register-source-grid", action="store_true")
    ap.add_argument(
        "--grid-from",
        default=None,
        help="a split manifest whose dataset block must agree with the dataset",
    )
    args = ap.parse_args()
    if args.register_source_grid:
        register(args)
        return

    dataset = torch.load(args.dataset, weights_only=False)
    y_mask = dataset["y_mask"].numpy()
    sites = list(dataset["site_no"])
    edges = pd.read_csv(args.edges, dtype=str)

    masks, manifest = build_masks(Path(args.source_masks), y_mask, edges, sites)
    manifest["dataset_path"] = args.dataset
    manifest["edge_file"] = args.edges
    manifest["graph_cv_report"] = component_report(edges, sites)
    hashes = write_masks(masks, manifest, Path(args.out))

    print(f"\nwrote {len(masks)} masks to {args.out}")
    for name in list(KEY_SCENARIOS) + list(manifest["diagnostic_scenarios"]):
        entry = manifest["masks"][name]
        roles = " ".join(f"{k}={v}" for k, v in entry["roles"].items())
        visible = ",".join(entry["visible_roles"])
        dropped = entry.get("outer_test_cells_dropped_unobserved", 0)
        print(f"  {name:24s} {roles}  visible=[{visible}] "
              f"test_dropped={dropped} sha={hashes[name][:12]}")
    if manifest["migration_audit"]:
        print("\nmigration audit (source grid -> this grid):")
        for source, roles in manifest["migration_audit"].items():
            for role, counts in roles.items():
                if counts["dropped_unobserved"]:
                    print(f"  {source}.{role}: {counts['before']} -> "
                          f"{counts['after']} ({counts['dropped_unobserved']} "
                          f"no longer observed)")
    e3 = next(k for k in manifest["masks"] if k.startswith("e3_internal"))
    cv = manifest["masks"][e3]["cv_report"]
    print(
        f"\ngraph: {cv['n_components']} components, largest={cv['largest_component_size']}, "
        f"isolated={cv['n_isolated_stations']}"
    )
    print(f"manifest: {Path(args.out).parent / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
