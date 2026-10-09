"""Portable inference for the fitted unmonitored-station DOC recipe.

This inference adapter reuses fitted forests, native ecology/GRU weights and
source-OOF memory. It neither fits on new-station labels nor replaces training.
It lives outside the executing training modules while geography is running.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from run_unified_doc_spatial import verify_files

from river_graph.experiments.provenance import sha256_file
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import _solve_profile
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.station_adapted_hybrid import station_residual_correction

PROCEDURES = ("current_model", "matched_daily_trees", "station_hidden_trees",
              "unmonitored_residual", "unmonitored_integrated")


def _tensor(value):
    array = np.asarray(value)
    if any(stride < 0 for stride in array.strides):
        array = array.copy()
    return torch.as_tensor(array, dtype=torch.float32)


def _stats(value):
    return {"mean": value.mean(0).numpy(),
            "sd": value.std(0, unbiased=False).clamp_min(1e-8).numpy()}


def fitted_preprocessing(dataset, split):
    """Save exactly the existing backbone's fitted input statistics.

    Hydro/target statistics use source train cells. Coordinate/regime scales
    preserve the historical backbone's known-cohort population normalization;
    deployment never estimates them on a new cohort.
    """
    shape = np.asarray(dataset["x"]).shape[:2]
    train = np.asarray(split["train"], dtype=np.int64)
    target = _tensor(np.asarray(dataset["y"]).ravel()[train])
    if not len(train) or not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError("source training labels must be finite nonnegative DOC")
    target = target.log1p()
    x, coordinates = _tensor(dataset["x"]), _tensor(dataset["static"])
    row, column = train//shape[1], train % shape[1]
    hydro_refs = [x[row, column, channel] for channel in range(2)]
    coordinate_refs = [coordinates[:, channel:channel+1].reshape(-1) for channel in range(2)]

    def stacked_stats(refs):
        return {"mean": torch.stack([ref.mean() for ref in refs]).numpy(),
                "sd": torch.stack([ref.std(unbiased=False).clamp_min(1e-8) for ref in refs]).numpy()}

    return {"hydro": stacked_stats(hydro_refs),
            "coordinates": stacked_stats(coordinate_refs),
            "regime": _stats(_tensor(dataset["regime"])),
            "target": _stats(target), "reference_cohort_size": shape[0],
            "calendar_origin": str(np.asarray(dataset["months"], dtype="datetime64[M]")[0]),
            "normalization_scope": "saved historical backbone; no new-cohort fit"}


def fitted_source_bank(dataset, split, oof):
    """Retain source support and OOF errors, excluding validation/test labels."""
    n, months = np.asarray(dataset["x"]).shape[:2]
    train = np.sort(np.asarray(split["train"], dtype=np.int64))
    visible = np.zeros(n*months, dtype=bool)
    for role in ("train", "context"):
        visible[np.asarray(split.get(role, []), dtype=np.int64)] = True
    visible = visible.reshape(n, months)
    ids = np.unique(np.flatnonzero(visible.any(1)))
    values = np.zeros((n, months), dtype=np.float32)
    selected = np.flatnonzero(visible.ravel())
    values.ravel()[selected] = _tensor(np.asarray(dataset["y"]).ravel()[selected]).log1p().numpy()
    edges = np.asarray(dataset["edge_index"], dtype=np.int64).reshape(2, -1)
    names = np.asarray(dataset["site_no"], str)
    if len(np.unique(names)) != n:
        raise ValueError("source site identifiers must be unique")
    source_ids = np.unique(train//months)
    return {"station": names[ids], "original_ids": ids,
            "months": np.asarray(dataset["months"], dtype="datetime64[M]"),
            "visible": visible[ids], "log_values": values[ids],
            "river_edges": names[edges.T],
            "profile_station": names[source_ids], "profile_original_ids": source_ids,
            "profile_regime": np.asarray(dataset["regime"], float)[source_ids],
            "profile_owner": np.searchsorted(source_ids, train//months),
            "profile_context": np.maximum(0, np.expm1(np.asarray(oof).ravel()[train])),
            "profile_truth": np.asarray(dataset["y"], float).ravel()[train].copy()}


class PortableDOCReconstructor:
    """Use saved source preprocessing for arbitrary new sites/month grids.

    New-site inputs contain site_no, months, x, x_mask, static, regime and
    daily_features[N,T,8]. Labels and chemical inputs are never read. Calendar
    months must be consecutive. Known source-site identifiers are rejected.
    Optional river_edges names source/target sites; no positional ST node IDs
    are needed. Source context is aligned by calendar month, never extrapolated.
    """

    def __init__(self, *, forest, native, memory_state, preprocessing, source_bank,
                 readout_normalization, metadata=None):
        self.forest, self.native = forest, native
        self.memory_state = copy.deepcopy(memory_state)
        self.preprocessing = copy.deepcopy(preprocessing)
        self.source_bank = copy.deepcopy(source_bank)
        self.readout_normalization = copy.deepcopy(readout_normalization)
        self.metadata = copy.deepcopy(metadata or {})
        if forest.n_features_in_ not in (39, 47) or native.extra_dim != 38:
            raise ValueError("portable v1 requires a fitted 39/47-feature forest and 38-feature readout")
        if native.spatial_architecture["in_channels"] != 23:
            raise ValueError("portable v1 requires the local M1 backbone")
        self._profile_cache = {}

    @classmethod
    def fit(cls, *, dataset_path, mask_path, daily_path, workdir, seed=42,
            procedure="unmonitored_integrated", runtime_snapshot_hash,
            progress_path=None):
        """Fit the existing recipe using explicit source-only training roles.

        The training facade consumes a source dataset and a station-disjoint
        train/val split, with no test/context role. External labels are never
        accepted. Saved stages allow the same fixed fit to resume.
        """
        from fit_portable_doc_sources_v1 import fit_source_recipe

        if procedure not in PROCEDURES:
            raise ValueError("unsupported fitted DOC procedure")
        fit_source_recipe(dataset_path=dataset_path, mask_path=mask_path,
            daily_path=daily_path, run=workdir, seed=seed, runtime=runtime_snapshot_hash,
            progress_path=progress_path)
        return cls.from_fitted_run(workdir, procedure=procedure)

    @classmethod
    def from_geographical_run(cls, run, *, procedure="unmonitored_integrated"):
        """Compatibility entry point for the geographical inference fixtures."""
        return cls.from_fitted_run(run, procedure=procedure)

    @classmethod
    def from_fitted_run(cls, run, *, procedure="unmonitored_integrated"):
        """Export a completed fixed model; do not fit or reselect components."""
        run = Path(run)
        if procedure not in PROCEDURES:
            raise ValueError("unsupported fitted DOC procedure")
        config = json.loads((run/"config.json").read_text())
        for kind in ("dataset", "mask"):
            if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
                raise ValueError(f"fitted source {kind} content changed")
        for name in ("complete.json", "backbone_complete.json", "trees_complete.json", "current_complete.json"):
            verify_files(run, name, config)
        dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(config["mask_path"], allow_pickle=False) as saved:
            split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
        with np.load(run/"oof.npz", allow_pickle=False) as saved:
            oof = saved["old" if procedure == "current_model" else "new"].copy()
        prefix = "current_" if procedure == "current_model" else ""
        forest_file = {"current_model": run/"backbone/context.joblib",
                       "matched_daily_trees": run/"matched_daily_trees.joblib"}.get(
                           procedure, run/"station_hidden_trees.joblib")
        return cls(forest=joblib.load(forest_file),
            native=EncoderNativeResidual.from_payload(torch.load(run/f"{prefix}native.pt", weights_only=False)),
            memory_state=json.loads((run/f"{prefix}memory.json").read_text()),
            preprocessing=fitted_preprocessing(dataset, split),
            source_bank=fitted_source_bank(dataset, split, oof),
            readout_normalization=json.loads((run/f"{prefix}readout_normalization.json").read_text()),
            metadata={"source_run": str(run), "source_completion_sha256": sha256_file(run/"complete.json"),
                      "source_seed": config["seed"], "recipe": procedure,
                      "support_adapters": json.loads((run/"support_adapters.json").read_text())[procedure],
                      "target_information": "no target DOC/pH/conductance", "fit_role": "source_training"})

    def _validated_inputs(self, inputs):
        names = np.asarray(inputs["site_no"], str)
        dates = np.asarray(inputs["months"], dtype="datetime64[M]")
        if (names.ndim != 1 or not len(names) or len(np.unique(names)) != len(names)
                or np.intersect1d(names, self.source_bank["station"]).size):
            raise ValueError("new-site identifiers must be unique and exclude the source library")
        if dates.ndim != 1 or not len(dates) or np.isnat(dates).any() or np.any(np.diff(dates.astype(int)) != 1):
            raise ValueError("months must form a nonempty consecutive calendar grid")
        shape = (len(names), len(dates))
        arrays = {key: np.asarray(inputs[key]) for key in ("x", "x_mask", "static", "regime", "daily_features")}
        expected = {"x": (*shape, 2), "x_mask": (*shape, 2), "static": (shape[0], 2),
                    "regime": (shape[0], 13), "daily_features": (*shape, 8)}
        if any(arrays[k].shape != s or not np.isfinite(arrays[k]).all() for k, s in expected.items()):
            raise ValueError("new-site features must be finite, aligned and have the documented dimensions")
        if not np.isin(arrays["x_mask"], [0, 1]).all():
            raise ValueError("hydro visibility must be binary")
        return names, dates, arrays

    def _source_context(self, names, dates, river_edges):
        bank = self.source_bank
        month_lookup = {int(value): i for i, value in enumerate(bank["months"].astype(int))}
        visible = np.zeros((len(bank["station"]), len(dates)), dtype=bool)
        values = np.zeros(visible.shape, dtype=np.float32)
        for column, date in enumerate(dates.astype(int)):
            if int(date) in month_lookup:
                source_month = month_lookup[int(date)]
                visible[:, column] = bank["visible"][:, source_month]
                values[:, column] = bank["log_values"][:, source_month]
        count = visible.sum(0)
        # Preserve the legacy float32 global reduction, whereas directional
        # summaries use the float64 adjacency-matrix arithmetic of Temporal RF.
        observed = np.where(visible, values, np.float32(0.))
        context = np.zeros((len(names), len(dates), 6), dtype=np.float32)
        context[..., 0] = observed.sum(0)/np.maximum(count, 1)
        context[..., 1] = count/max(self.preprocessing["reference_cohort_size"]-1, 1)
        edges = np.asarray(river_edges, str).reshape(-1, 2)
        source_lookup = {name: i for i, name in enumerate(bank["station"])}
        # A neighbor with no source DOC contributes to degree but not support.
        for row, name in enumerate(names):
            for slot, neighbors in ((2, np.unique(edges[edges[:, 1] == name, 0])),
                                    (4, np.unique(edges[edges[:, 0] == name, 1]))):
                donors = [source_lookup[s] for s in neighbors if s in source_lookup]
                if donors:
                    support = visible[donors].sum(0)
                    context[row, :, slot] = observed[donors].astype(float).sum(0)/np.maximum(support, 1)
                    context[row, :, slot+1] = support/max(len(neighbors), 1)
        return context

    def prepare_inputs(self, inputs):
        """Label-free forest and causal neural inputs with frozen source scales."""
        names, dates, arrays = self._validated_inputs(inputs)
        n, months = len(names), len(dates)
        phase = 2*np.pi*(dates.astype(int) % 12)/12
        season = np.stack([np.sin(phase), np.cos(phase)], -1)
        context = self._source_context(names, dates, inputs.get("river_edges", self.source_bank["river_edges"]))
        lags = np.zeros((n, months, 12), dtype=np.float32)
        lags[..., 2::3] = [1, 3, 6, 12]
        forest_features = np.concatenate([
            _tensor(arrays["x"]).numpy(), _tensor(arrays["x_mask"]).numpy(),
            np.broadcast_to(season, (n, months, 2)),
            np.broadcast_to(_tensor(arrays["static"]).numpy()[:, None], (n, months, 2)),
            np.broadcast_to(_tensor(arrays["regime"]).numpy()[:, None], (n, months, 13)),
            context, lags, arrays["daily_features"]], -1).astype(np.float32)
        selected_features = forest_features[..., :self.forest.n_features_in_]
        forest_prediction = np.maximum(0, np.expm1(self.forest.predict(
            selected_features.reshape(-1, self.forest.n_features_in_)))).reshape(n, months)

        def standardized(value, key):
            norm = self.preprocessing[key]
            return (_tensor(value)-_tensor(norm["mean"]))/ _tensor(norm["sd"])

        raw = torch.zeros((n, months, 23), dtype=torch.float32)
        hydro, coordinates, regime = (standardized(arrays[k], key) for k, key in
            (("x", "hydro"), ("static", "coordinates"), ("regime", "regime")))
        raw[..., 0], raw[..., 1] = hydro[..., 0], _tensor(arrays["x_mask"])[..., 0]
        raw[..., 2], raw[..., 3] = hydro[..., 1], _tensor(arrays["x_mask"])[..., 1]
        calendar_months = torch.as_tensor(dates.astype(int) % 12, dtype=torch.float32)
        angle = 2.*torch.pi*calendar_months/12.
        raw[..., 4], raw[..., 5] = torch.sin(angle), torch.cos(angle)
        raw[..., 6:8] = coordinates[:, None]
        raw[..., 10:14] = regime[:, None, :4]
        origin = np.datetime64(self.preprocessing["calendar_origin"], "M").astype(int)
        age = torch.log1p(torch.as_tensor(np.maximum(dates.astype(int)-origin+1, 1), dtype=torch.float32))/float(np.log1p(12))
        raw[..., 15] = age
        raw[..., 21] = _tensor(context[..., 3])
        support = raw[..., [9, 21, 22]].numpy()
        flow = build_causal_flow_features({"x": arrays["x"], "x_mask": arrays["x_mask"]})["full"]
        norm = self.readout_normalization
        ecology = np.asarray(arrays["regime"], float)[:, 4:13]
        valid = np.isfinite(ecology) & (ecology != -1)
        median, scale = np.asarray(norm["ecology"]["median"]), np.asarray(norm["ecology"]["iqr"])
        difference = np.where(valid, ecology, median)-median
        eco = difference/(scale+np.abs(difference))
        eco[:, ~np.asarray(norm["ecology"]["active"], bool)] = 0
        extra = np.empty((n, months, 38), dtype=np.float32)
        extra[..., :10] = flow
        extra[..., 10:19], extra[..., 19:28] = eco[:, None], valid[:, None]
        difference = np.log1p(forest_prediction)-norm["context"]["log_mean"]
        extra[..., 28] = difference/(norm["context"]["log_sd"]+np.abs(difference))
        extra[..., 29], extra[..., 30:] = 1., arrays["daily_features"]
        neural = {"raw": raw.numpy(), "env": regime[:, 4:13].numpy(),
                  "age": raw[..., 15].numpy(), "support": support, "extra": extra}
        return forest_features, forest_prediction, neural

    def _memory_profiles(self, names, regime):
        state, bank = self.memory_state, self.source_bank
        norm, settings = state["normalization"], state["selected"]
        scale = state["ecology_scaler"]
        raw = np.asarray(regime, float)[:, 4:13]
        donors_raw = bank["profile_regime"][:, 4:13]
        valid, donors_valid = np.isfinite(raw) & (raw != -1), np.isfinite(donors_raw) & (donors_raw != -1)
        center, width = np.asarray(scale["median"]), np.asarray(scale["iqr"])
        receiving = (np.where(valid, raw, center)-center)/width
        donors = (np.where(donors_valid, donors_raw, center)-center)/width
        receiving[:, ~np.asarray(scale["active"], bool)] = 0
        donors[:, ~np.asarray(scale["active"], bool)] = 0
        u = (np.log1p(bank["profile_context"])-norm["log_context_mean"])/norm["log_context_sd"]
        design = np.column_stack([np.ones(len(u)), u]) if state["mode"].endswith("affine") else np.ones((len(u), 1))
        target = (bank["profile_truth"]-bank["profile_context"])/norm["residual_scale"]
        coefficients, diagnostics = [], []
        for row, name in enumerate(names):
            pool = np.flatnonzero(bank["profile_station"] != name)
            if state["mode"].startswith("ecological") and valid[row].sum() >= 5:
                eligible = pool[donors_valid[pool].sum(1) >= 5]
                if len(eligible):
                    distance = np.mean((donors[eligible]-receiving[row])**2, axis=1)
                    order = np.lexsort((bank["profile_original_ids"][eligible], distance))
                    pool = eligible[order[:min(settings["k"], len(order))]]
            key = tuple(sorted(int(p) for p in pool))
            if not key:
                raise ValueError("ecological memory has no eligible source donors")
            if key not in self._profile_cache:
                selected, weights = [], []
                for owner in key:
                    positions = np.flatnonzero(bank["profile_owner"] == owner)
                    selected.extend(positions)
                    weights.extend(np.full(len(positions), 1/(len(key)*len(positions))))
                x, r, weights = design[selected], target[selected], np.asarray(weights)

                def objective(theta, x=x, r=r, weights=weights):
                    error = r-x@theta
                    smooth = np.sqrt(error**2+.05**2)
                    return (float(weights@smooth+settings["ridge"]*(theta@theta)),
                            -(x.T@(weights*error/smooth))+2*settings["ridge"]*theta)

                result, _ = _solve_profile(objective, x.shape[1])
                theta = np.zeros(2)
                theta[:x.shape[1]] = result.x
                self._profile_cache[key] = theta
            coefficients.append(self._profile_cache[key])
            diagnostics.append({"station": str(name), "source_stations": bank["profile_station"][list(key)].tolist(),
                                "donor_count": len(key)})
        return np.asarray(coefficients), diagnostics

    def predict_components(self, inputs):
        _, environment, neural = self.prepare_inputs(inputs)
        names = np.asarray(inputs["site_no"], str)
        recipe = self.metadata.get("recipe", "unmonitored_integrated")
        native = (environment.copy() if recipe.endswith("trees") else
                  np.maximum(0, environment+self.native.selected_scale_*self.native.predict_delta(neural)))
        gamma = (0. if recipe.endswith("trees") or recipe == "unmonitored_residual"
                 else self.memory_state["selected"]["gamma"])
        diagnostics = []
        if gamma == 0:
            final = native.copy()
        else:
            coefficients, diagnostics = self._memory_profiles(names, inputs["regime"])
            norm = self.memory_state["normalization"]
            u = (np.log1p(environment)-norm["log_context_mean"])/norm["log_context_sd"]
            memory_delta = norm["residual_scale"]*(coefficients[:, :1]+coefficients[:, 1:]*u)
            final = np.maximum(0, environment+(1-gamma)*(native-environment)+gamma*memory_delta)
        return {"environment_pred": environment, "local_temporal_correction": native-environment,
                "source_transfer_correction": final-native, "river_correction": np.zeros_like(final),
                "native_pred": native, "final_pred": final,
                "hydro_availability": np.asarray(inputs["x_mask"]).sum(-1),
                "retrieval_sources": diagnostics}

    def predict(self, inputs):
        return self.predict_components(inputs)["final_pred"]

    def predict_with_support(self, inputs, *, k, support_cells, support_values):
        """Apply the saved source-selected retrospective K-support calibration.

        No target/query values are accepted. Supports calibrate predictions;
        they do not enter the encoder, forest or source library. Scoring must
        exclude the same five candidate supports at every K.
        """
        if k not in (0, 1, 3, 5):
            raise ValueError("supported K values are 0,1,3,5")
        base = self.predict(inputs)
        cells, values = np.asarray(support_cells), np.asarray(support_values, float)
        if (cells.ndim != 1 or cells.dtype.kind not in "iu" or cells.shape != values.shape
                or (cells < 0).any() or (cells >= base.size).any() or len(np.unique(cells)) != len(cells)
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError("support must have unique valid cell identities and finite nonnegative DOC")
        counts = np.bincount(cells//base.shape[1], minlength=base.shape[0])
        if np.any(counts > k):
            raise ValueError("support count exceeds the specified K")
        if k == 0:
            return base
        alpha = self.metadata["support_adapters"][str(k)]["selected"]["alpha"]
        query = np.setdiff1d(np.arange(base.size), cells)
        result = base.copy().ravel()
        result[query] = station_residual_correction(base.ravel()[query], query,
            base.ravel()[cells], cells, values, n_months=base.shape[1], alpha=alpha)
        # Designated supports retain their base prediction and are never scored
        # as query cells. They are not passed to the disjoint-query adapter.
        return result.reshape(base.shape)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = {"schema_version": 1, "forest": self.forest, "native": self.native.to_payload(),
                   "memory": self.memory_state, "preprocessing": self.preprocessing,
                   "source_bank": self.source_bank, "readout_normalization": self.readout_normalization,
                   "metadata": self.metadata}
        joblib.dump(payload, directory/"model.joblib", compress=3)
        (directory/"manifest.json").write_text(json.dumps({"schema_version": 1,
            "model_sha256": sha256_file(directory/"model.joblib"), **self.metadata}, indent=2)+"\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        if sha256_file(directory/"model.joblib") != manifest["model_sha256"]:
            raise ValueError("saved portable model content changed")
        payload = joblib.load(directory/"model.joblib")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported portable DOC model schema")
        return cls(forest=payload["forest"], native=EncoderNativeResidual.from_payload(payload["native"]),
            memory_state=payload["memory"], preprocessing=payload["preprocessing"],
            source_bank=payload["source_bank"], readout_normalization=payload["readout_normalization"],
            metadata=payload["metadata"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--procedure", choices=PROCEDURES, default="unmonitored_integrated")
    args = parser.parse_args()
    torch.set_num_threads(2)
    PortableDOCReconstructor.from_geographical_run(args.source_run, procedure=args.procedure).save(args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
