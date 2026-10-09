"""Environmental and temporal DOC reconstruction with station calibration.

All three components form one fitted, serializable predictor. Held-out target
observations enter only the explicit calibration call, never either expert.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.spatial_fewshot import K_VALUES
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    LocalTransportKGML,
    build_rf_features,
    fit_rf_artifacts,
    target_values,
)
from river_graph.models.station_adapted_hybrid import (
    CalibrationEpisode,
    StationAdaptedHybrid,
)


class UnifiedDOCReconstructor:
    """Fixed environmental trees + OOF temporal residuals + local adaptation."""

    def __init__(self, *, seed=42, n_estimators=300, n_jobs=4, max_epochs=20,
                 patience=5, hidden=64, chunk_months=64, epoch_callback=None):
        self.settings = {"seed": seed, "n_estimators": n_estimators, "n_jobs": n_jobs,
                         "max_epochs": max_epochs, "patience": patience,
                         "hidden": hidden, "chunk_months": chunk_months}
        self.epoch_callback = epoch_callback

    def _new_residual(self):
        return LocalTransportKGML(
            analyte="doc", edge_direction="upstream", edge_set="empty",
            base_variant="local", spatial_variant="baseline", temporal_operator="gru",
            lookback=12, dropout=0.1, lr=0.001, forest_backend="extra_trees",
            forest_min_samples_leaf=4, forest_max_features=1.0,
            epoch_callback=self.epoch_callback, **self.settings,
        )

    def fit(self, dataset, split, *, rf=None, forest_callback=None):
        """Fit experts and select fusion/calibration on source validation only."""
        self.dataset, self.split = dataset, split
        self.n_months = dataset["y"].shape[1]
        if rf is None:
            rf = fit_rf_artifacts(
                dataset, split, target_transform="log1p", seed=self.settings["seed"],
                n_estimators=self.settings["n_estimators"], n_jobs=self.settings["n_jobs"],
                forest_backend="extra_trees", min_samples_leaf=4, max_features=1.0,
            )
        self.rf = rf
        if forest_callback is not None:
            forest_callback(rf)
        features = build_rf_features(dataset, split, FIT_ROLES,
                                    target_transform="log1p", include_network=True)
        z = target_values(dataset, "log1p").ravel()
        y = np.asarray(dataset["y"], dtype=np.float64).ravel()
        _, validation_query = support_query_cells(
            split, target_role="val", k=0, n_months=self.n_months,
        )
        candidates = [("et_leaf4", 4, 1.0), ("et_leaf2", 2, 1.0),
                      ("et_leaf1", 1, 1.0), ("et_sqrt", 1, "sqrt")]
        self.context_selection = []
        best = np.inf
        for name, leaf, max_features in candidates:
            forest = rf.context if name == "et_leaf4" else ExtraTreesRegressor(
                n_estimators=self.settings["n_estimators"], random_state=self.settings["seed"],
                n_jobs=self.settings["n_jobs"], min_samples_leaf=leaf, max_features=max_features,
            ).fit(features[split["train"]], z[split["train"]])
            prediction = np.maximum(0, np.expm1(forest.predict(features[validation_query])))
            score = float(np.abs(prediction - y[validation_query]).mean())
            self.context_selection.append({"candidate": name, "validation_mae": score})
            if score < best:
                best = score
                self.context_forest, self.context_name = forest, name
        self.residual = self._new_residual()
        self.residual.fit(dataset, split, rf=rf)
        temporal_components = self.residual.predict_components(FIT_ROLES)
        context = np.maximum(0, np.expm1(self.context_forest.predict(features)))
        temporal = temporal_components["final_pred"].ravel()
        self.adapter = StationAdaptedHybrid(n_months=self.n_months)
        self.adapter.fit_fusion(context[validation_query], temporal[validation_query],
                                y[validation_query], selection_role="source_validation")
        episodes = []
        for k in K_VALUES:
            support, query = support_query_cells(split, target_role="val", k=k,
                                                 n_months=self.n_months)
            episodes.append(CalibrationEpisode(
                k=k, query_cells=query, query_values=y[query],
                context_query_prediction=context[query], temporal_query_prediction=temporal[query],
                support_cells=support, support_values=y[support],
                context_support_prediction=context[support], temporal_support_prediction=temporal[support],
            ))
        self.adapter.fit_calibration(episodes, selection_role="source_validation")
        self.components_ = self._assemble(context, temporal, temporal_components)
        return self

    def _assemble(self, context, temporal, temporal_components):
        assembled = self.adapter.predict_components(context, temporal)
        shape = self.dataset["y"].shape
        return {
            "context_pred": assembled["context"].reshape(shape),
            "temporal_pred": assembled["temporal"].reshape(shape),
            "hybrid_pred": assembled["hybrid"].reshape(shape),
            "local_pred": temporal_components["local_pred"],
            "temporal_delta": temporal_components["graph_delta"],
        }

    def predict_components(self, *, recompute=False):
        if recompute or not hasattr(self, "components_"):
            x = build_rf_features(self.dataset, self.split, FIT_ROLES,
                                  target_transform="log1p", include_network=True)
            context = np.maximum(0, np.expm1(self.context_forest.predict(x)))
            temporal = self.residual.predict_components(FIT_ROLES)
            self.components_ = self._assemble(context, temporal["final_pred"].ravel(), temporal)
        return self.components_

    def predict(self, query_cells, *, support_cells=None, support_values=None,
                k=0, arm="hybrid", calibrated=True):
        """Reconstruct any query cells using exactly K support labels/station."""
        if arm not in ("context", "hybrid"):
            raise ValueError("arm must be context or hybrid")
        base = self.predict_components()[f"{arm}_pred"].ravel()
        def indices(values):
            raw = np.asarray(values)
            if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
                raise ValueError("cell identities must be a one-dimensional integer array")
            selected = raw.astype(np.int64)
            if (selected < 0).any() or (selected >= base.size).any():
                raise ValueError("cell identities are outside the station-month grid")
            return selected
        query = indices(query_cells)
        if not calibrated:
            return base[query].copy()
        support = indices([] if support_cells is None else support_cells)
        values = np.asarray([] if support_values is None else support_values, dtype=np.float64)
        return self.adapter.adapt(base[query], query, base[support], support, values, k=k, arm=arm)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf, directory / "forests.joblib", compress=3)
        joblib.dump(self.context_forest, directory / "context.joblib", compress=3)
        torch.save(self.residual.model.state_dict(), directory / "residual.pt")
        payload = {
            "settings": self.settings, "context_name": self.context_name,
            "context_selection": self.context_selection,
            "adapter": self.adapter.to_dict(), "trace": self.residual.trace,
            "best_epoch": self.residual.best_epoch,
            "best_val_loss": self.residual.best_val_loss,
        }
        (directory / "model.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        (directory / "adapter.json").write_text(json.dumps(payload["adapter"], indent=2) + "\n")

    @classmethod
    def load(cls, directory, dataset, split):
        directory = Path(directory)
        payload = json.loads((directory / "model.json").read_text())
        obj = cls(**payload["settings"])
        obj.dataset, obj.split = dataset, split
        obj.n_months = dataset["y"].shape[1]
        obj.rf = joblib.load(directory / "forests.joblib")
        obj.context_forest = joblib.load(directory / "context.joblib")
        obj.context_name, obj.context_selection = payload["context_name"], payload["context_selection"]
        obj.adapter = StationAdaptedHybrid.from_dict(payload["adapter"])
        obj.residual = obj._new_residual()
        obj.residual.initialize(dataset, split, obj.rf)
        obj.residual.model.load_state_dict(torch.load(directory / "residual.pt", weights_only=True))
        obj.residual.model.eval()
        obj.residual.trace = payload["trace"]
        obj.residual.epochs_run = len(payload["trace"])
        obj.residual.best_epoch = payload["best_epoch"]
        obj.residual.best_val_loss = payload["best_val_loss"]
        return obj
