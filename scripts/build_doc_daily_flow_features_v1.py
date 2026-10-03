"""Freeze within-month daily-discharge descriptors without reading DOC labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import river_graph.models.daily_flow_features as daily_module
from river_graph.models.daily_flow_features import (
    POLICY,
    build_daily_flow_features,
    file_sha256,
    load_first_daily_discharge,
)

DEFAULT_ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
DEFAULT_DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
DEFAULT_CACHE = Path("data/raw/nwis_dv")


def freeze_daily_features(dataset_path: Path, raw_cache: Path, output_root: Path) -> dict:
    """Create one immutable data product, or verify/reuse its identical inputs."""
    raw_paths = sorted(raw_cache.glob("dv_*.rdb"))
    if not raw_paths:
        raise FileNotFoundError(f"no daily cache files under {raw_cache}")
    identity = {
        "dataset_hash": file_sha256(dataset_path),
        "raw_cache_hashes": {str(path): file_sha256(path) for path in raw_paths},
        "module_hash": file_sha256(Path(daily_module.__file__)),
        "builder_hash": file_sha256(Path(__file__)),
        "policy": POLICY,
    }
    archive_path = output_root / "daily_features.npz"
    metadata_path = output_root / "daily_features.meta.json"
    if archive_path.exists() or metadata_path.exists():
        if not archive_path.exists() or not metadata_path.exists():
            raise RuntimeError("incomplete frozen daily feature product; use a new versioned directory")
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise RuntimeError("daily feature inputs/code changed; existing product cannot be overwritten")
        if metadata["feature_product_hash"] != file_sha256(archive_path):
            raise RuntimeError("frozen daily feature archive does not match its recorded content hash")
        return metadata

    # Deserialize the archive, then immediately retain only label-free fields.
    archive = torch.load(dataset_path, weights_only=False, map_location="cpu")
    dataset = {key: archive[key] for key in ("site_no", "months", "x_mask")}
    del archive
    daily, inventory = load_first_daily_discharge(raw_cache)
    features = build_daily_flow_features(dataset, daily)
    del daily
    full = features.pop("full")
    metadata = {"version": 1, "dataset_path": str(dataset_path),
                "dataset_hash": identity["dataset_hash"], "identity": identity,
                "feature_shape": list(full.shape), "dtype": str(full.dtype),
                "station_order": [str(x) for x in dataset["site_no"]],
                "month_order": [str(x) for x in dataset["months"]],
                "feature_product": str(archive_path), **features, **inventory}
    output_root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is deliberate: versioning, never replacement.
    with archive_path.open("xb") as stream:
        np.savez_compressed(stream, full=full,
                            site_no=np.asarray(metadata["station_order"]),
                            months=np.asarray(metadata["month_order"]))
    metadata["feature_product_hash"] = file_sha256(archive_path)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    metadata = freeze_daily_features(args.dataset, args.raw_cache, args.output_root)
    print(json.dumps({"product": metadata["feature_product"],
                      "feature_shape": metadata["feature_shape"],
                      "quality_summary": metadata["quality_summary"]}, indent=2))


if __name__ == "__main__":
    main()
