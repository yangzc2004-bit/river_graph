"""Verify fixed release identity, old-score preservation and named-site inference."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from portable_current_source_ensemble_v2 import PortableCurrentSourceEnsemble
from run_doc_current_source_external_v2 import (
    EXTERNAL_PROTOCOL,
    NEW_MODELS,
    PREVIOUS,
    ROOT,
    SOURCE,
)
from run_unified_doc_spatial import digest, write_json

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root
    points = json.loads((root/"point_generation_complete.json").read_text())
    if points["external_DOC_read"] or points["config_hash"] != digest(points["config"]):
        raise ValueError("point-generation identity or information condition changed")
    for name, expected in points["files"].items():
        if sha256_file(root/name) != expected:
            raise ValueError("external point content changed")
    meta = json.loads((root/"predictions.meta.json").read_text())
    if meta["config_hash"] != digest(meta["config"]) or sha256_file(root/"predictions.parquet") != meta["prediction_sha256"]:
        raise ValueError("external scored product identity changed")
    old = pd.read_parquet(PREVIOUS/"predictions.parquet")
    panel = pd.read_parquet(root/"predictions.parquet")
    retained = panel[~panel.model_name.isin(NEW_MODELS)][old.columns]
    keys = ["population", "model_name", "k", "cell"]
    pd.testing.assert_frame_equal(retained.sort_values(keys).reset_index(drop=True),
        old.sort_values(keys).reset_index(drop=True), check_exact=True)
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    with np.load(external["label_free_inputs"], allow_pickle=False) as saved:
        inputs = {key: saved[key][:3].copy() for key in ("site_no", "x", "x_mask", "static", "regime")}
        inputs.update(months=saved["months"].copy(), daily_features=saved["daily"][:3].copy())
    release = PortableCurrentSourceEnsemble.load(SOURCE/"release.json")
    actual = release.predict(inputs)
    with np.load(root/"seed_components.npz", allow_pickle=False) as saved:
        expected = saved[f"{NEW_MODELS[1]}__final_pred"].mean(0)[:3]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    # Target chemistry fields never enter point inference. Future water input
    # perturbations cannot change earlier reconstructed months with fixed bank.
    changed = {**inputs, "y": np.full(actual.shape, 12345.), "ph": np.full(actual.shape, 14.),
        "spec_conductance": np.full(actual.shape, 1e8), "x": inputs["x"].copy(),
        "daily_features": inputs["daily_features"].copy()}
    boundary = len(inputs["months"])//2
    changed["x"][:, boundary:] += 100.
    changed["daily_features"][:, boundary:] = 0.
    checked = release.predict(changed)
    np.testing.assert_allclose(checked[:, :boundary], actual[:, :boundary], atol=1e-12, rtol=1e-12)
    write_json(root/"verification.json", {"point_files": "verified", "scored_identity": "verified",
        "previous_prediction_rows_preserved_exactly": len(old), "new_named_stations": 3,
        "max_ensemble_replay_error_mg_l": float(np.abs(actual-expected).max()),
        "hidden_chemistry_and_future_hydro_contract": "passed", "release_manifest": str(SOURCE/"release.json"),
        "release_manifest_sha256": sha256_file(SOURCE/"release.json"), "verifier_sha256": sha256_file(__file__),
        "scope": points["config"]["external_exposure"]})
    print("Fixed release replay, hidden/future inputs and all old external rows verified", flush=True)


if __name__ == "__main__":
    main()
