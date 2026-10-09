"""Deploy the fixed current-source attention model at arbitrary new DOC sites."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from portable_doc_reconstructor_v1 import PortableDOCReconstructor
from run_unified_doc_spatial import verify_files

from river_graph.experiments.provenance import sha256_file
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
    current_available_candidates,
)
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)
from river_graph.models.source_innovation_training import innovation_readout_features
from river_graph.models.source_level_attention import source_level_input_view


class PortableCurrentSourceDOC:
    """Reuse saved preprocessing and source libraries; accept no receiver labels.

    The local helper supplies the existing forest features, ecological memory
    and support adapter. This class adds only the already evaluated 41-feature
    readout and current-source attention, with calendar-aligned source hydro.
    """

    def __init__(self, *, local_helper, native, aggregate_library, relative_library,
                 source_hydro, memory_state, metadata):
        if not isinstance(native, AvailableSourceAttentionResidual) or native.extra_dim != 41:
            raise ValueError("fixed 41-feature current-availability readout required")
        self.local_helper, self.native = local_helper, native
        self.aggregate_library = copy.deepcopy(aggregate_library)
        self.relative_library = copy.deepcopy(relative_library)
        self.source_hydro = np.asarray(source_hydro).copy()
        self.metadata = copy.deepcopy(metadata)
        self.local_helper.memory_state = copy.deepcopy(memory_state)
        self.local_helper._profile_cache = {}
        a, b = self.aggregate_library, self.relative_library
        if (a.candidate_count != 20 or b.candidate_count != 20
                or not np.array_equal(a.source_names_, b.source_names_)
                or not np.array_equal(a.months_, b.months_)
                or self.source_hydro.shape != (len(b.source_names_), len(b.months_), 8)
                or not np.isfinite(self.source_hydro).all()
                or not np.isin(b.source_names_, local_helper.source_bank["station"]).all()):
            raise ValueError("aligned source-only 20-candidate libraries and hydrology required")

    @classmethod
    def from_geographical_run(cls, run, *, prefix="available_real"):
        """Export an evaluated fixed fit without choosing among geographical K results."""
        run = Path(run)
        config = json.loads((run/"config.json").read_text())
        verify_files(run, "complete.json", config)
        parent = Path(config["parent_run"])
        if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
            raise ValueError("fitted geographical reference changed")
        helper = PortableDOCReconstructor.from_fitted_run(parent)
        dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(config["mask_path"], allow_pickle=False) as masks:
            train = masks["train"].copy()
        with np.load(parent/"oof.npz", allow_pickle=False) as saved:
            ids, residual = source_residual_grid(dataset["y"], saved["new"], train)
        relative = SourceDOCInnovationLibrary.load(run/"relative_source_library.npz")
        np.testing.assert_array_equal(relative.source_names_, np.asarray(dataset["site_no"], str)[ids])
        aggregate = SourceDOCInnovationLibrary().fit(relative.source_names_, relative.months_,
            relative.ecology_, residual)
        with np.load(run/"attention_candidates.npz", allow_pickle=False) as saved:
            hydro = saved["full_donor_hydro_bank"].copy()
        metadata = {"source_run": str(run), "source_completion_sha256": sha256_file(run/"complete.json"),
            "source_seed": config["seed"], "recipe": f"{prefix}_integrated",
            "support_adapters": json.loads((run/"support_adapters.json").read_text())[f"{prefix}_integrated"],
            "target_information": "no receiving DOC/pH/conductance",
            "scope": "portable replay of evaluated ST357 geographical fit; not deployment refit"}
        return cls(local_helper=helper,
            native=AvailableSourceAttentionResidual.from_payload(torch.load(run/f"{prefix}.pt",
                weights_only=False, map_location="cpu")), aggregate_library=aggregate,
            relative_library=relative, source_hydro=hydro,
            memory_state=json.loads((run/f"{prefix}_memory.json").read_text()), metadata=metadata)

    def _aligned_source_hydro(self, months):
        source = np.asarray(self.relative_library.months_, dtype="datetime64[M]").astype(int)
        lookup = {int(month): i for i, month in enumerate(source)}
        dates = np.asarray(months, dtype="datetime64[M]").astype(int)
        result = np.zeros((len(self.relative_library.source_names_), len(dates), 8),
                          dtype=self.source_hydro.dtype)
        for column, month in enumerate(dates):
            if int(month) in lookup:
                result[:, column] = self.source_hydro[:, lookup[int(month)]]
        return result

    def prepare_inputs(self, inputs):
        features, environment, neural = self.local_helper.prepare_inputs(inputs)
        names, months = np.asarray(inputs["site_no"], str), np.asarray(inputs["months"], str)
        parts = self.aggregate_library.predict_components(names, neural["env"], months)
        extra = innovation_readout_features(parts, "real", 1.)
        candidates = current_available_candidates(self.relative_library, names, neural["env"], months,
            self.relative_library.source_names_, self._aligned_source_hydro(months))
        view = source_level_input_view({**neural, "attention_reference": environment,
            "extra": np.concatenate([neural["extra"], extra], -1)}, candidates, "real")
        return features, environment, view

    def predict_components(self, inputs):
        _, environment, neural = self.prepare_inputs(inputs)
        # This counterfactual readout decomposition uses the same fitted weights.
        # Remove dynamic source descriptors/values, retaining all local inputs.
        local = {**neural, "extra": neural["extra"].copy(),
                 "donor_values": np.zeros_like(neural["donor_values"])}
        local["extra"][..., 38:] = 0.
        local_pred = self.native.predict(local, environment)
        native_pred = self.native.predict(neural, environment)
        gamma = self.local_helper.memory_state["selected"]["gamma"]
        retrieval = []
        if gamma == 0:
            final = native_pred.copy()
        else:
            coefficients, retrieval = self.local_helper._memory_profiles(
                np.asarray(inputs["site_no"], str), inputs["regime"])
            norm = self.local_helper.memory_state["normalization"]
            u = (np.log1p(environment)-norm["log_context_mean"])/norm["log_context_sd"]
            memory = norm["residual_scale"]*(coefficients[:, :1]+coefficients[:, 1:]*u)
            final = np.maximum(0., environment+(1-gamma)*(native_pred-environment)+gamma*memory)
        n, t = environment.shape
        diagnostics = self.native.diagnostics(neural)
        return {"environment_pred": environment, "local_temporal_correction": local_pred-environment,
            "source_observation_correction": native_pred-local_pred,
            "source_static_correction": final-native_pred,
            "source_transfer_correction": final-local_pred, "river_correction": np.zeros_like(final),
            "native_pred": native_pred, "final_pred": final,
            "hydro_availability": np.asarray(inputs["x_mask"]).sum(-1),
            "source_support_count": neural["donor_valid"][..., :-1].sum(-1),
            "attention_prior_mass": diagnostics["prior_mass"].reshape(n, t, -1),
            "attention_entropy": diagnostics["entropy"].reshape(n, t, -1),
            "retrieval_sources": retrieval}

    def predict(self, inputs):
        return self.predict_components(inputs)["final_pred"]

    def predict_with_support(self, inputs, *, k, support_cells, support_values):
        # Reuse the exact explicit-support adapter with this class's prediction
        # and source-selected metadata; it does not inspect receiver labels.
        return PortableDOCReconstructor.predict_with_support(self, inputs, k=k,
            support_cells=support_cells, support_values=support_values)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.local_helper.save(directory/"local_helper")
        payload = {"schema_version": 2, "native": self.native.to_payload(),
            "aggregate_library": self.aggregate_library, "relative_library": self.relative_library,
            "source_hydro": self.source_hydro, "memory": self.local_helper.memory_state,
            "metadata": self.metadata}
        joblib.dump(payload, directory/"source_attention.joblib", compress=3)
        files = [directory/"source_attention.joblib", directory/"local_helper/manifest.json",
                 directory/"local_helper/model.joblib"]
        (directory/"manifest.json").write_text(json.dumps({"schema_version": 2, **self.metadata,
            "files": {str(p.relative_to(directory)): sha256_file(p) for p in files}}, indent=2)+"\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest["schema_version"] != 2 or any(sha256_file(directory/name) != expected
                for name, expected in manifest["files"].items()):
            raise ValueError("saved current-source model content or schema changed")
        payload = joblib.load(directory/"source_attention.joblib")
        if payload["schema_version"] != 2:
            raise ValueError("unsupported current-source deployment schema")
        return cls(local_helper=PortableDOCReconstructor.load(directory/"local_helper"),
            native=AvailableSourceAttentionResidual.from_payload(payload["native"]),
            aggregate_library=payload["aggregate_library"], relative_library=payload["relative_library"],
            source_hydro=payload["source_hydro"], memory_state=payload["memory"], metadata=payload["metadata"])
