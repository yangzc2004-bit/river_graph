"""Build matched masks/datasets for the non-ST context intervention.

The intervention keeps the ST target station-month cells fixed and creates:

* ``st_context``: the 370-node graph with non-ST DOC labels masked out;
* ``full_labels``: the same graph with non-ST observed labels available in
  train/validation/context cells.

The existing ST357 dataset/masks provide the common target grid.  This script
only writes new derived artifacts and never modifies the frozen datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def remap_mask(
    mask: dict[str, np.ndarray],
    source_sites: list[str],
    target_sites: list[str],
    t: int,
) -> dict[str, np.ndarray]:
    target_index = {str(site): i for i, site in enumerate(target_sites)}
    source_to_target = np.asarray([target_index[str(site)] for site in source_sites])
    out: dict[str, np.ndarray] = {}
    for key, values in mask.items():
        values = np.asarray(values)
        if key == "held_out_sites":
            continue
        rows, cols = values // t, values % t
        out[key] = source_to_target[rows] * t + cols
    return out


def non_st_cells(values: np.ndarray, st_index: set[int], t: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return values[~np.isin(values // t, list(st_index))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-dataset",
        default="data/processed/mississippi_graph_graphfix_cached370.pt",
    )
    ap.add_argument(
        "--st-dataset", default="data/processed/mississippi_graph_graphfix_st357.pt"
    )
    ap.add_argument("--st-masks", default="experiments/masks_graphfix_st357")
    ap.add_argument("--full-masks", default="experiments/masks_graphfix_cached370")
    ap.add_argument(
        "--context-dataset",
        default="data/processed/mississippi_graph_graphfix_context_only370.pt",
    )
    ap.add_argument("--context-masks", default="experiments/masks_direction_context370")
    ap.add_argument("--label-masks", default="experiments/masks_direction_full370")
    args = ap.parse_args()

    full = torch.load(args.full_dataset, weights_only=False)
    st = torch.load(args.st_dataset, weights_only=False)
    full_sites = [str(s) for s in full["site_no"]]
    st_sites = [str(s) for s in st["site_no"]]
    if not set(st_sites).issubset(full_sites):
        raise ValueError("ST dataset contains sites absent from the full dataset")
    t = int(full["y"].shape[1])
    st_index = {full_sites.index(site) for site in st_sites}
    names = ["e2a_strict", "e2b_partial"]

    context_masks = Path(args.context_masks)
    label_masks = Path(args.label_masks)
    context_masks.mkdir(parents=True, exist_ok=True)
    label_masks.mkdir(parents=True, exist_ok=True)

    for name in names:
        with np.load(Path(args.st_masks) / f"{name}.npz", allow_pickle=False) as z:
            st_mask = {key: z[key] for key in z.files}
        with np.load(Path(args.full_masks) / f"{name}.npz", allow_pickle=False) as z:
            full_mask = {key: z[key] for key in z.files}
        common = remap_mask(st_mask, st_sites, full_sites, t)
        np.savez(context_masks / f"{name}.npz", **common)

        # Preserve the original full-graph train/validation/context arrays and
        # their order exactly. The model's per-epoch RNG permutes the stored
        # train-cell array, so reordering an otherwise identical set would
        # create an avoidable training difference. Only the scored test role
        # is replaced by the common ST-only test cells.
        hybrid = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in full_mask.items()
            if key != "held_out_sites"
        }
        hybrid["test"] = common["test"]
        np.savez(label_masks / f"{name}.npz", **hybrid)

        print(
            f"{name}: context train={len(common['train'])} val={len(common['val'])} "
            f"test={len(common['test'])}; full-label train={len(hybrid['train'])} "
            f"val={len(hybrid['val'])} test={len(hybrid['test'])} "
            f"context={len(hybrid.get('context', []))}"
        )

    context = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in full.items()
    }
    context["y_mask"] = full["y_mask"].clone()
    non_st_rows = [i for i in range(len(full_sites)) if i not in st_index]
    context["y_mask"][non_st_rows, :] = False
    torch.save(context, args.context_dataset)

    metadata = {
        "full_dataset": args.full_dataset,
        "st_dataset": args.st_dataset,
        "context_dataset": args.context_dataset,
        "context_masks": args.context_masks,
        "label_masks": args.label_masks,
        "n_full_nodes": len(full_sites),
        "n_st_nodes": len(st_sites),
        "n_non_st_nodes": len(non_st_rows),
        "non_st_labels_masked": True,
        "target_definition": "ST station-month test cells remapped from the ST357 masks",
        "full_label_definition": "same ST train/val/test plus observed non-ST train/val/context cells from the 370-node masks",
    }
    Path(args.context_dataset).with_suffix(".provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.context_dataset}")


if __name__ == "__main__":
    main()
