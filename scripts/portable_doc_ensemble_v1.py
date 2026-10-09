"""Portable five-seed DOC prediction with source-selected support and intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_doc_external_replication_v1 import COMPONENTS, empirical_interval
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.station_adapted_hybrid import station_residual_correction


class PortableDOCEnsemble:
    """One prediction interface around the retained fixed source members.

    Averaging is in mg/L before any support correction. Intervals and K-specific
    shrinkage use the frozen source-validation *ensemble* policy. Per-member
    variation is a descriptive output, not a calibrated uncertainty rank.
    """

    def __init__(self, members, policy):
        self.members, self.policy = dict(members), policy
        if set(self.members) != {42, 43, 44, 45, 46}:
            raise ValueError("the fixed deployment requires all five source seeds")

    @classmethod
    def from_source_fits(cls, source_root, calibration_path):
        root = Path(source_root)
        source = json.loads(Path(calibration_path).read_text())
        if source["external_DOC_used"] or source["external_support_used"]:
            raise ValueError("ensemble policy must use source validation only")
        return cls({seed: PortableDOCReconstructor.load(root/"exports"/f"seed{seed}")
            for seed in (42, 43, 44, 45, 46)}, source["procedures"]["unmonitored_integrated"])

    def predict_components(self, inputs):
        parts = [member.predict_components(inputs) for _, member in sorted(self.members.items())]
        result = {key: np.mean([part[key] for part in parts], axis=0) for key in COMPONENTS}
        result["member_prediction_sd"] = np.std([part["final_pred"] for part in parts], axis=0, ddof=1)
        result["pi_lower"], result["pi_upper"] = empirical_interval(
            result["final_pred"], self.policy["primary_k0_log_half_width"])
        result["hydro_availability"] = parts[0]["hydro_availability"]
        result["retrieval_sources"] = [{"seed": seed, **record} for (seed, _), part in
            zip(sorted(self.members.items()), parts, strict=True) for record in part["retrieval_sources"]]
        return result

    def predict(self, inputs):
        return self.predict_components(inputs)["final_pred"]

    def predict_with_support(self, inputs, *, k, support_cells, support_values):
        if k not in (0, 1, 3, 5):
            raise ValueError("supported K values are0,1,3,5")
        base = self.predict(inputs)
        cells, values = np.asarray(support_cells), np.asarray(support_values, float)
        if (cells.ndim != 1 or cells.dtype.kind not in "iu" or cells.shape != values.shape
                or (cells < 0).any() or (cells >= base.size).any() or len(np.unique(cells)) != len(cells)
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError("designated support cells and values must be finite, unique and aligned")
        if np.any(np.bincount(cells//base.shape[1], minlength=base.shape[0]) > k):
            raise ValueError("support count exceeds K")
        result = base.copy().ravel()
        if k:
            query = np.setdiff1d(np.arange(base.size), cells)
            result[query] = station_residual_correction(result[query], query, result[cells], cells,
                values, n_months=base.shape[1], alpha=self.policy["curve"][str(k)]["selected"]["alpha"])
        lower, upper = empirical_interval(result.reshape(base.shape), self.policy["curve"][str(k)]["log_half_width"])
        return {"final_pred": result.reshape(base.shape), "pi_lower": lower, "pi_upper": upper}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        files = {}
        for seed, member in sorted(self.members.items()):
            member.save(directory/"members"/f"seed{seed}")
            name = f"members/seed{seed}/manifest.json"
            files[name] = sha256_file(directory/name)
        write_json(directory/"ensemble.json", {"schema_version": 1, "seeds": sorted(self.members),
            "mean_scale": "native mg/L", "interval_type": "source-validation empirical log1p calibration",
            "policy": self.policy, "member_manifests": files})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        saved = json.loads((directory/"ensemble.json").read_text())
        if saved["schema_version"] != 1 or saved["mean_scale"] != "native mg/L":
            raise ValueError("unsupported ensemble schema")
        for name, expected in saved["member_manifests"].items():
            if sha256_file(directory/name) != expected:
                raise ValueError("portable ensemble member definition changed")
        return cls({seed: PortableDOCReconstructor.load(directory/"members"/f"seed{seed}")
                    for seed in saved["seeds"]}, saved["policy"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("experiments/phase4_transfer/doc_portable_source_fit_v1"))
    parser.add_argument("--calibration", type=Path, default=Path("experiments/phase4_transfer/doc_external_replication_v1/source_calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/phase4_transfer/doc_portable_source_fit_v1/ensemble"))
    args = parser.parse_args()
    PortableDOCEnsemble.from_source_fits(args.source_root, args.calibration).save(args.output)
    print(args.output, flush=True)
