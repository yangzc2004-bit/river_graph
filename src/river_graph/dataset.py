"""Assemble the training graph dataset (torch tensors, temporal layout).

The dataset keeps monthly resolution: y and every feature channel are
(n_nodes, n_months) matrices with NaN for unobserved cells, plus an
observation mask for y. Edge indices follow the PyG convention
(2, n_edges), source -> target = upstream -> downstream.

Conversion to per-snapshot PyG Data objects happens at training time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_CHANNELS = ["temperature", "discharge"]  # (N, T) each -> x (N, T, F)


def build_dataset(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    monthly_doc: pd.DataFrame,
    monthly_features: dict[str, pd.DataFrame],
    months: pd.DatetimeIndex,
) -> dict:
    """Build the dataset dict. torch is only needed to save as .pt."""
    import torch  # deferred: only needed here, not by the data pipeline

    from river_graph.data.aggregate import to_matrix

    sites = list(nodes["site_no"])
    site_idx = {s: i for i, s in enumerate(sites)}
    e = edges[
        edges["source"].isin(site_idx) & edges["target"].isin(site_idx)
    ]
    edge_index = torch.tensor(
        [[site_idx[s] for s in e["source"]], [site_idx[t] for t in e["target"]]],
        dtype=torch.long,
    )

    y = to_matrix(monthly_doc, "doc", sites, months)
    mask = ~np.isnan(y)
    channels = [
        to_matrix(monthly_features[ch], ch, sites, months) for ch in FEATURE_CHANNELS
    ]
    x = np.stack(channels, axis=-1)  # (N, T, F)

    return {
        "site_no": sites,
        "months": [str(m.date()) for m in months],
        "edge_index": edge_index,
        "y": torch.tensor(np.nan_to_num(y), dtype=torch.float32),
        "y_mask": torch.tensor(mask, dtype=torch.bool),
        "x": torch.tensor(np.nan_to_num(x), dtype=torch.float32),
        "x_mask": torch.tensor(~np.isnan(x), dtype=torch.bool),
        "feature_channels": FEATURE_CHANNELS,
        "static": torch.tensor(
            nodes[["dec_lat_va", "dec_long_va"]].astype(float).values,
            dtype=torch.float32,
        ),
    }


def save_dataset(dataset: dict, out_path: str | Path) -> None:
    import torch

    torch.save(dataset, out_path)
