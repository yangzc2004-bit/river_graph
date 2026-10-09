"""Named-station inference for the fixed five-member current-source DOC release."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from portable_current_source_doc_v2 import PortableCurrentSourceDOC
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_doc_external_replication_v1 import empirical_interval

from river_graph.experiments.provenance import sha256_file


class PortableCurrentSourceEnsemble:
    """Average all fixed members in native units, then adapt explicit support.

    Member export paths are relative to the release manifest; the bundle can
    move with its source-fit directory. No ST357 receiving-node index is used.
    """

    def __init__(self, *, member_paths, calibration, metadata):
        self.member_paths = [Path(path) for path in member_paths]
        if len(self.member_paths) != 5:
            raise ValueError("the fixed release requires all five source members")
        self.calibration = calibration
        self.metadata = {**metadata, "support_adapters": calibration["curve"]}

    def predict_components(self, inputs):
        result, retrieval = {}, []
        for path in self.member_paths:
            model = PortableCurrentSourceDOC.load(path)
            components = model.predict_components(inputs)
            for key, value in components.items():
                if key == "retrieval_sources":
                    retrieval.extend({"seed": model.metadata["source_seed"], **row} for row in value)
                    continue
                value = np.asarray(value)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite member prediction or diagnostic")
                if key in result and result[key].shape != value.shape:
                    raise ValueError("member station/month component identities disagree")
                result[key] = result.get(key, np.zeros_like(value, dtype=float))+value/5
        result["retrieval_sources"] = retrieval
        return result

    def predict(self, inputs):
        return self.predict_components(inputs)["final_pred"]

    def source_candidates(self, inputs):
        """Name each member's ecological candidates and observed-month support.

        These are candidate identities, not learned attention weights. Static
        memory retrieval is recorded separately and may be empty when gamma=0.
        """
        records = []
        names = np.asarray(inputs["site_no"], str)
        for path in self.member_paths:
            member = PortableCurrentSourceDOC.load(path)
            _, _, neural = member.prepare_inputs(inputs)
            for row, target in enumerate(names):
                positions = np.flatnonzero(neural["donor_owner"][row] >= 0)
                owners = neural["donor_owner"][row, positions]
                records.append({"seed": member.metadata["source_seed"], "target_station": str(target),
                    "source_stations": member.relative_library.source_names_[owners].tolist(),
                    "observed_month_count": neural["donor_valid"][row][:, positions].sum(0).tolist(),
                    "definition": "ecological candidate pool; learned weights are not inferred from candidate ranks"})
        return records

    def predict_with_support(self, inputs, *, k, support_cells, support_values):
        return PortableDOCReconstructor.predict_with_support(self, inputs, k=k,
            support_cells=support_cells, support_values=support_values)

    def predict_interval(self, inputs, *, k=0, support_cells=None, support_values=None):
        if k == 0:
            if support_cells is not None or support_values is not None:
                raise ValueError("K0 interval accepts no receiving support")
            center = self.predict(inputs)
            width = self.calibration["primary_k0_log_half_width"]
        else:
            center = self.predict_with_support(inputs, k=k, support_cells=support_cells,
                support_values=support_values)
            width = self.calibration["curve"][str(k)]["log_half_width"]
        lower, upper = empirical_interval(center, width)
        return {"y_pred": center, "pi_lower": lower, "pi_upper": upper,
                "definition": "source-validation empirical log1p interval; no coverage guarantee"}

    def save(self, path):
        """Save a small manifest pointing to already serialized fixed fits."""
        import os
        path = Path(path)
        if path.exists():
            raise ValueError("preserve release manifest; choose a new file")
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 2, "metadata": self.metadata, "calibration": self.calibration,
            "members": [{"path": os.path.relpath(directory.resolve(), path.parent.resolve()),
                "manifest_sha256": sha256_file(directory/"manifest.json")} for directory in self.member_paths]}
        path.write_text(json.dumps(manifest, indent=2)+"\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads(path.read_text())
        if manifest["schema_version"] != 2:
            raise ValueError("unsupported ensemble release schema")
        members = [(path.parent/row["path"]).resolve() for row in manifest["members"]]
        for directory, row in zip(members, manifest["members"], strict=True):
            if sha256_file(directory/"manifest.json") != row["manifest_sha256"]:
                raise ValueError("saved release member manifest changed")
        return cls(member_paths=members, calibration=manifest["calibration"], metadata=manifest["metadata"])


def main():
    """Export complete station-month predictions from label-free named inputs."""
    import argparse

    import pandas as pd
    import torch
    from run_unified_doc_spatial import digest, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = args.output.with_suffix(".sources.json")
    if args.output.exists() or args.output.with_suffix(".meta.json").exists() or diagnostics.exists():
        raise ValueError("preserve existing prediction; choose a new output")
    torch.set_num_threads(2)
    with np.load(args.inputs, allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("site_no", "months", "x", "x_mask", "static", "regime")}
        inputs["daily_features"] = saved["daily_features" if "daily_features" in saved.files else "daily"].copy()
        if "river_edges" in saved.files:
            inputs["river_edges"] = saved["river_edges"].copy()
    model = PortableCurrentSourceEnsemble.load(args.release)
    components = model.predict_components(inputs)
    lower, upper = empirical_interval(components["final_pred"], model.calibration["primary_k0_log_half_width"])
    names, months = np.asarray(inputs["site_no"], str), np.asarray(inputs["months"], str)
    numeric = {key: value.ravel() for key, value in components.items()
               if isinstance(value, np.ndarray) and value.ndim == 2}
    panel = pd.DataFrame({"station": np.repeat(names, len(months)), "month": np.tile(months, len(names)),
        "analyte": "doc", "model_name": "environmental_memory_source", "seed": -1,
        "visibility_role": "unmonitored_new_station", "k": 0, "support_count": 0,
        **numeric, "y_pred": components["final_pred"].ravel(), "pi_lower": lower.ravel(), "pi_upper": upper.ravel(),
        "interval_width": (upper-lower).ravel(),
        "attention_prior_mass_mean": components["attention_prior_mass"].mean(-1).ravel(),
        "attention_entropy_mean": components["attention_entropy"].mean(-1).ravel()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    write_json(diagnostics, {"static_memory_sources": components["retrieval_sources"],
        "attention_candidates": model.source_candidates(inputs)})
    config = {"release_sha256": sha256_file(args.release), "inputs_sha256": sha256_file(args.inputs),
        "input_path": str(args.inputs), "release_path": str(args.release), "k": 0,
        "target_information": "receiving DOC/pH/conductance fields ignored", "script_sha256": sha256_file(__file__)}
    write_json(args.output.with_suffix(".meta.json"), {"config": config, "config_hash": digest(config),
        "prediction_sha256": sha256_file(args.output), "source_diagnostics_sha256": sha256_file(diagnostics),
        "n_stations": len(names), "n_months": len(months), "rows": len(panel),
        "uncertainty": "source-validation empirical log1p intervals; no coverage guarantee"})
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
