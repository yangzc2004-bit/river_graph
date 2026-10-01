"""Fixed local forests plus a causal, directed river residual.

RF-context reproduces the existing Temporal RF feature set. RF-local adds
strictly past local observation statistics, with no other-station features.
Residual training uses five fixed station-held-out views and OOF base errors.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from torch import nn

from river_graph.experiments.graph_upgrade_v2 import GraphUpgradeModel
from river_graph.experiments.temporal_h2x import (
    TARGET_TRANSFORMS,
    _as_tensor,
    build_temporal_inputs,
    inverse_target,
    transform_target,
)

FIT_ROLES = ("train", "context")
TEST_ROLES = ("train", "val", "context")
LAGS = (1, 3, 6, 12)
FOREST_BACKENDS = ("random_forest", "extra_trees")


def role_visible(split: dict, roles: Iterable[str], shape: tuple) -> np.ndarray:
    roles = tuple(roles)
    if not set(roles) <= set(TEST_ROLES):
        raise ValueError("test labels cannot be visible inputs")
    visible = np.zeros(np.prod(shape), dtype=bool)
    for role in roles:
        visible[np.asarray(split.get(role, []), dtype=np.int64)] = True
    return visible.reshape(shape)


def target_values(dataset: dict, transform: str) -> np.ndarray:
    return transform_target(_as_tensor(dataset["y"]), transform).numpy()


def local_history(values: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Last value, age, valid, and counts, all strictly before the query month."""
    n, t = values.shape
    out = np.zeros((n, t, 7), dtype=np.float32)
    last = np.zeros(n, dtype=np.float32)
    last_month = np.full(n, -1, dtype=int)
    csum = np.concatenate([np.zeros((n, 1)), visible.cumsum(1)], axis=1)
    for month in range(t):
        valid = last_month >= 0
        out[:, month, 0] = np.where(valid, last, 0)
        out[:, month, 1] = np.log1p(np.where(valid, month - last_month, month + 1))
        out[:, month, 2] = valid
        for slot, width in enumerate(LAGS):
            out[:, month, slot + 3] = csum[:, month] - csum[:, max(0, month - width)]
        seen = visible[:, month]
        last = np.where(seen, values[:, month], last)
        last_month = np.where(seen, month, last_month)
    return out


def network_context(values: np.ndarray, visible: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Exactly the six current-month context columns of Temporal RF."""
    n, _t = values.shape
    counts = visible.astype(float)
    observed = np.where(visible, values, 0.0)
    total_count = counts.sum(0)[None, :] - counts
    total_sum = observed.sum(0)[None, :] - observed
    up = np.zeros((n, n), dtype=float)
    if edges.size:
        up[edges[1], edges[0]] = 1
    up_count, down_count = up @ counts, up.T @ counts
    return np.stack([
        total_sum / np.maximum(total_count, 1), total_count / max(n - 1, 1),
        (up @ observed) / np.maximum(up_count, 1),
        up_count / np.maximum(up.sum(1)[:, None], 1),
        (up.T @ observed) / np.maximum(down_count, 1),
        down_count / np.maximum(up.sum(0)[:, None], 1),
    ], -1)


def rf_feature_names(include_network: bool) -> list[str]:
    names = ["temperature", "discharge", "temperature_visible", "discharge_visible",
             "month_sin", "month_cos", "latitude", "longitude"]
    names += [f"regime_{i}" for i in range(13)]
    if include_network:
        names += ["global_value", "global_fraction", "upstream_value", "upstream_fraction",
                  "downstream_value", "downstream_fraction"]
    else:
        names += ["last_value", "log_age", "last_valid", "count_1", "count_3", "count_6", "count_12"]
    names += [f"lag_{lag}_{name}" for lag in LAGS for name in ("value", "visible", "months")]
    return names


def build_rf_features(dataset: dict, split: dict, roles: Iterable[str], *,
                      target_transform: str, include_network: bool) -> np.ndarray:
    """RF-local (40 columns) or unchanged Temporal RF-context (39 columns).

    Both exclude the current cell's target. Context has the legacy network
    summaries and lags; local replaces those summaries with local history.
    """
    values = target_values(dataset, target_transform)
    n, t = values.shape
    visible = role_visible(split, roles, (n, t))
    phase = 2 * np.pi * (np.asarray(dataset["months"], dtype="datetime64[M]").astype(int) % 12) / 12
    blocks = [_as_tensor(dataset["x"]).numpy(), _as_tensor(dataset["x_mask"]).numpy(),
              np.broadcast_to(np.stack([np.sin(phase), np.cos(phase)], -1), (n, t, 2))]
    for key in ("static", "regime"):
        a = _as_tensor(dataset[key]).numpy()
        blocks.append(np.broadcast_to(a[:, None, :], (n, t, a.shape[1])))
    if include_network:
        blocks.append(network_context(values, visible, np.asarray(dataset["edge_index"], dtype=int)))
    else:
        blocks.append(local_history(values, visible))
    for lag in LAGS:
        b = np.zeros((n, t, 3))
        b[..., 2] = lag
        if lag < t:
            b[:, lag:, 0] = np.where(visible[:, :-lag], values[:, :-lag], 0)
            b[:, lag:, 1] = visible[:, :-lag]
        blocks.append(b)
    features = np.concatenate(blocks, -1).reshape(n * t, -1).astype(np.float32)
    if features.shape[1] != len(rf_feature_names(include_network)) or not np.isfinite(features).all():
        raise ValueError("invalid RF features")
    return features


def station_folds(train: np.ndarray, t: int, seed: int) -> list[np.ndarray]:
    stations = np.unique(train // t)
    if len(stations) < 2:
        raise ValueError("OOF requires at least two training stations")
    return list(np.array_split(np.random.default_rng(seed).permutation(stations), min(5, len(stations))))


def fold_split(split: dict, held_stations: np.ndarray, t: int) -> dict:
    """Hide every target at a held station, including any context labels."""
    result = {k: np.asarray(v, dtype=np.int64) for k, v in split.items()}
    for role in FIT_ROLES:
        cells = result.get(role, np.array([], dtype=np.int64))
        result[role] = cells[~np.isin(cells // t, held_stations)]
    return result


@dataclass
class RFArtifacts:
    local: RandomForestRegressor
    context: RandomForestRegressor
    local_oof_z: np.ndarray  # unstandardized log1p DOC/EC or raw pH
    folds: list[np.ndarray]
    target_mu: float
    target_sd: float
    target_transform: str
    context_oof_z: np.ndarray | None = None
    forest_backend: str = "random_forest"
    forest_min_samples_leaf: int = 1
    forest_max_features: float | str | None = 1.0

    def predict_std(self, dataset: dict, split: dict, roles: Iterable[str]) -> tuple:
        shape = dataset["y"].shape
        predictions = []
        for network, forest in ((False, self.local), (True, self.context)):
            x = build_rf_features(dataset, split, roles, target_transform=self.target_transform,
                                  include_network=network)
            predictions.append(((forest.predict(x) - self.target_mu) / self.target_sd).reshape(shape))
        return tuple(predictions)

    def inverse_std(self, z: np.ndarray) -> np.ndarray:
        return inverse_target(torch.as_tensor(z * self.target_sd + self.target_mu), self.target_transform).numpy()


def _forest(backend: str, *, n_estimators: int, n_jobs: int, seed: int,
            min_samples_leaf: int, max_features: float | str | None):
    _validate_forest_settings(backend, min_samples_leaf, max_features)
    cls = RandomForestRegressor if backend == "random_forest" else ExtraTreesRegressor
    return cls(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed,
               min_samples_leaf=min_samples_leaf, max_features=max_features)


def _validate_forest_settings(backend: str, min_samples_leaf: int,
                              max_features: float | str | None) -> None:
    """Validate settings shared by fitted and cached forest artifacts.

    Keeping this check before sklearn construction gives a deterministic error
    for malformed experiment configurations instead of a backend-specific
    failure several minutes into OOF fitting.
    """
    if backend not in FOREST_BACKENDS:
        raise ValueError(f"unknown forest backend: {backend}")
    if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be a positive integer")
    if max_features is None:
        return
    if isinstance(max_features, str):
        if max_features not in ("sqrt", "log2"):
            raise ValueError("max_features must be a fraction, sqrt, log2, or None")
        return
    if isinstance(max_features, Real) and not isinstance(max_features, bool):
        # sklearn accepts a positive integer feature count or a fraction in
        # (0, 1].  The public CLI currently emits fractions; retain integer
        # support for direct Python callers.
        if isinstance(max_features, (int, np.integer)):
            if max_features < 1:
                raise ValueError("integer max_features must be positive")
        elif not (0 < float(max_features) <= 1):
            raise ValueError("fraction max_features must be in (0, 1]")
        return
    raise ValueError("invalid max_features")


def fit_oof_fold(dataset: dict, split: dict, held_stations: np.ndarray, *,
                 target_transform: str, seed: int, n_estimators: int, n_jobs: int,
                 forest_backend: str = "random_forest", min_samples_leaf: int = 1,
                 max_features: float | str | None = 1.0) -> tuple:
    """Fit a fold without using its targets, including in target statistics."""
    t = dataset["y"].shape[1]
    train = np.asarray(split["train"], dtype=np.int64)
    held = train[np.isin(train // t, held_stations)]
    view = fold_split(split, held_stations, t)
    x_local = build_rf_features(dataset, view, FIT_ROLES, target_transform=target_transform,
                                include_network=False)
    x_context = build_rf_features(dataset, view, FIT_ROLES, target_transform=target_transform,
                                  include_network=True)
    y = target_values(dataset, target_transform).ravel()
    # Trees fit directly in log1p/raw units; thus no held-label normalizer.
    local = _forest(forest_backend, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed,
                    min_samples_leaf=min_samples_leaf, max_features=max_features)
    context = _forest(forest_backend, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed,
                      min_samples_leaf=min_samples_leaf, max_features=max_features)
    local.fit(x_local[view["train"]], y[view["train"]])
    context.fit(x_context[view["train"]], y[view["train"]])
    return held, local.predict(x_local[held]), context.predict(x_context[held])


def fit_rf_artifacts(dataset: dict, split: dict, *, target_transform: str, seed: int,
                     n_estimators: int = 200, n_jobs: int = 4,
                     forest_backend: str = "random_forest", min_samples_leaf: int = 1,
                     max_features: float | str | None = 1.0) -> RFArtifacts:
    y = target_values(dataset, target_transform)
    train = np.asarray(split["train"], dtype=np.int64)
    ref = torch.as_tensor(y.ravel()[train])
    mu, sd = float(ref.mean()), float(ref.std(unbiased=False).clamp_min(1e-8))
    forests = []
    for network in (False, True):
        x = build_rf_features(dataset, split, FIT_ROLES, target_transform=target_transform, include_network=network)
        rf = _forest(forest_backend, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed,
                     min_samples_leaf=min_samples_leaf, max_features=max_features)
        rf.fit(x[train], y.ravel()[train])
        forests.append(rf)
    folds = station_folds(train, y.shape[1], seed)
    local_oof = np.full(y.shape, np.nan)
    context_oof = np.full(y.shape, np.nan)
    for stations in folds:
        cells, local_pred, context_pred = fit_oof_fold(
            dataset, split, stations, target_transform=target_transform,
            seed=seed, n_estimators=n_estimators, n_jobs=n_jobs,
            forest_backend=forest_backend, min_samples_leaf=min_samples_leaf,
            max_features=max_features)
        local_oof.ravel()[cells] = local_pred
        context_oof.ravel()[cells] = context_pred
    if not np.isfinite(local_oof.ravel()[train]).all() or not np.isfinite(context_oof.ravel()[train]).all():
        raise ValueError("OOF coverage incomplete")
    return RFArtifacts(*forests, local_oof, folds, mu, sd, target_transform, context_oof,
                       forest_backend, min_samples_leaf, max_features)


def residual_inputs(builder, inputs, visible, model):
    x, age = builder._make_input(inputs, visible, model)
    # All residual arms share upstream support features. Remove downstream
    # support from ALL arms so upstream-only has no reverse information path.
    x[..., -1] = 0
    return x, age


class LocalTransportKGML:
    """A fixed RF base plus a signed causal residual in target space.

    ``base_variant='local'`` is the original K1 model.  ``base_variant='context'``
    uses the explicit current-month network summaries of RF-context and lets
    the temporal graph branch fit only the residual left by that stronger
    spatial base.
    """

    def __init__(self, *, analyte="doc", seed=42, edge_direction="upstream", edge_set="river",
                 max_epochs=30, patience=5, n_estimators=200, n_jobs=4, hidden=64,
                 dropout=0.1, lr=1e-3, chunk_months=64, epoch_callback=None,
                 spatial_variant="baseline", message_feature_mode="all",
                 base_variant="local", temporal_operator="gru",
                 attention_heads=2, attention_dropout=0.1, lookback=12,
                 forest_backend="random_forest", forest_min_samples_leaf=1,
                 forest_max_features: float | str | None = 1.0):
        if analyte not in TARGET_TRANSFORMS or edge_direction not in ("upstream", "both"):
            raise ValueError("invalid analyte/direction")
        if edge_set not in ("river", "empty") or min(max_epochs, patience, n_estimators, hidden, chunk_months) < 1:
            raise ValueError("invalid model settings")
        self.analyte, self.seed = analyte, seed
        self.edge_direction, self.edge_set = edge_direction, edge_set
        self.max_epochs, self.patience = max_epochs, patience
        self.n_estimators, self.n_jobs = n_estimators, n_jobs
        _validate_forest_settings(forest_backend, forest_min_samples_leaf, forest_max_features)
        self.forest_backend = forest_backend
        self.forest_min_samples_leaf = int(forest_min_samples_leaf)
        self.forest_max_features = forest_max_features
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.chunk_months, self.epoch_callback = chunk_months, epoch_callback
        if temporal_operator not in ("gru", "gru_attention", "gru_attention_river"):
            raise ValueError("invalid temporal_operator")
        if attention_heads < 1 or attention_dropout < 0 or attention_dropout >= 1:
            raise ValueError("invalid attention settings")
        self.temporal_operator = temporal_operator
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        if lookback < 1 or (temporal_operator == "gru_attention_river" and lookback < 13):
            raise ValueError("invalid temporal lookback")
        self.lookback = int(lookback)
        if spatial_variant not in ("baseline", "msgonly", "msgdual"):
            raise ValueError("invalid spatial_variant")
        self.spatial_variant = spatial_variant
        if message_feature_mode not in ("all", "target_only", "hydro_ecology"):
            raise ValueError("invalid message feature mode")
        if spatial_variant not in ("msgonly", "msgdual") and message_feature_mode != "all":
            raise ValueError("message feature mode only applies to msgonly")
        self.message_feature_mode = message_feature_mode
        if base_variant not in ("local", "context"):
            raise ValueError("invalid base variant")
        self.base_variant = base_variant
        self.context_mode = "all" if base_variant == "context" else "none"

    def initialize(self, dataset: dict, split: dict, rf: RFArtifacts):
        """Build the zero-residual model; also supports a real initialization test."""
        artifact_backend = getattr(rf, "forest_backend", "random_forest")
        artifact_leaf = getattr(rf, "forest_min_samples_leaf", 1)
        artifact_features = getattr(rf, "forest_max_features", 1.0)
        if (artifact_backend != self.forest_backend
                or int(artifact_leaf) != self.forest_min_samples_leaf
                or artifact_features != self.forest_max_features):
            raise ValueError("RF artifact settings do not match model settings")
        torch.manual_seed(self.seed)
        self.dataset, self.split, self.rf = dataset, split, rf
        self.inputs = build_temporal_inputs(dataset, split, target_transform=TARGET_TRANSFORMS[self.analyte],
                                            env_encoder=True, context_mode=self.context_mode)
        self.builder = GraphUpgradeModel(mechanism="m1", seed=self.seed, lookback=self.lookback,
            hidden=self.hidden, temporal_hidden=self.hidden, layers=2, dropout=self.dropout,
            target_transform=TARGET_TRANSFORMS[self.analyte], env_encoder=True,
            edge_direction=self.edge_direction, edge_set=self.edge_set, feature_edge_set="river",
            chunk_months=self.chunk_months, spatial_variant=self.spatial_variant,
            message_feature_mode=self.message_feature_mode,
            temporal_operator=self.temporal_operator,
            attention_heads=self.attention_heads,
            attention_dropout=self.attention_dropout)
        self.model = self.builder._build_model(self.inputs, dataset)
        nn.init.zeros_(self.model.spatial.head.weight)
        nn.init.zeros_(self.model.spatial.head.bias)

    def input_view(self, split: dict, roles=FIT_ROLES):
        visible = torch.as_tensor(role_visible(split, roles, self.inputs.y_model.shape))
        return residual_inputs(self.builder, self.inputs, visible, self.model)

    def delta_tensor(self, x, age):
        return self.builder._forward(self.model, x, self.inputs, self.dataset, age)

    def fit(self, dataset: dict, split: dict, rf: RFArtifacts | None = None) -> None:
        if rf is None:
            rf = fit_rf_artifacts(dataset, split, target_transform=TARGET_TRANSFORMS[self.analyte],
                                  seed=self.seed, n_estimators=self.n_estimators, n_jobs=self.n_jobs,
                                  forest_backend=self.forest_backend,
                                  min_samples_leaf=self.forest_min_samples_leaf,
                                  max_features=self.forest_max_features)
        self.initialize(dataset, split, rf)
        _n, t = self.inputs.y_model.shape
        train, val = (np.asarray(split.get(r, []), dtype=np.int64) for r in ("train", "val"))
        if not len(val):
            raise ValueError("validation cells required for checkpoint selection")
        if self.base_variant == "context":
            base_oof = getattr(rf, "context_oof_z", None)
            if base_oof is None:
                raise ValueError("context OOF predictions are required for context-base residuals")
        else:
            base_oof = rf.local_oof_z
        residual = (target_values(dataset, rf.target_transform) - base_oof) / rf.target_sd
        residual = torch.as_tensor(residual, dtype=torch.float32)
        views = []
        for stations in rf.folds:
            cells = train[np.isin(train // t, stations)]
            x, age = self.input_view(fold_split(split, stations, t))
            views.append((cells, x, age))
        val_x, val_age = self.input_view(split)
        local_val, context_val = rf.predict_std(dataset, split, FIT_ROLES)
        base_val = context_val if self.base_variant == "context" else local_val
        base_val = torch.as_tensor(base_val.ravel()[val], dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, state, bad = float("inf"), None, 0
        self.trace = []
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            # One optimizer update per epoch, like M1; folds only supply safe
            # views. Backprop each fold separately to bound memory usage.
            for cells, x, age in views:
                delta = self.delta_tensor(x, age).reshape(-1)[cells]
                loss = F.mse_loss(delta, residual.reshape(-1)[cells]) * (len(cells) / len(train))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite KGML training loss")
                if loss.requires_grad:
                    loss.backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                delta = self.delta_tensor(val_x, val_age).reshape(-1)[val]
                criterion = float(F.mse_loss(base_val + delta, self.inputs.y_model.reshape(-1)[val]))
            if not np.isfinite(criterion):
                raise FloatingPointError("nonfinite KGML validation loss")
            row = {"epoch": epoch, "loss": total_loss, "val_loss": criterion}
            self.trace.append(row)
            if self.epoch_callback:
                self.epoch_callback(row)
            if criterion < best - 1e-6:
                best, bad = criterion, 0
                self.best_epoch = epoch
                state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if bad >= self.patience:
                break
        self.model.load_state_dict(state)
        self.epochs_run, self.best_val_loss = epoch, best
        self.model.eval()

    def predict_components(self, visible_roles=TEST_ROLES) -> dict:
        local_std, context_std = self.rf.predict_std(self.dataset, self.split, visible_roles)
        x, age = self.input_view(self.split, visible_roles)
        self.model.eval()
        with torch.no_grad():
            delta_std = self.delta_tensor(x, age).numpy().astype(np.float64)
        local = self.rf.inverse_std(local_std)
        base_std = context_std if self.base_variant == "context" else local_std
        base = self.rf.inverse_std(base_std)
        final = self.rf.inverse_std(base_std + delta_std)
        if not np.isfinite(final).all():
            raise FloatingPointError("nonfinite KGML full-grid prediction")
        # graph_delta is in transform space (log1p DOC/EC, standardized pH).
        delta = delta_std if self.analyte == "ph" else delta_std * self.rf.target_sd
        result = {"local_pred": local, "context_pred": self.rf.inverse_std(context_std),
                "base_pred": base,
                "final_pred": final, "graph_delta": delta, "graph_delta_std": delta_std,
                "graph_delta_raw": final - base, "graph_delta_abs": np.abs(delta)}
        diagnostics = getattr(self.model, "last_attention_diagnostics", None)
        if diagnostics is not None:
            weights = diagnostics["weights"].numpy()
            valid = diagnostics["valid"].numpy()
            # Keep per-cell diagnostics compact while preserving the main
            # interpretability quantities for downstream products.
            p = np.clip(weights, 1e-12, 1.0)
            entropy = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)
            lags = np.arange(weights.shape[-1] - 1, -1, -1)
            recent = weights[..., lags <= 1].sum(axis=-1).mean(axis=-1)
            seasonal = weights[..., lags >= 6].sum(axis=-1).mean(axis=-1)
            observed = weights[..., :].copy()
            observed *= (x[..., 9].numpy() > 0)[:, :, None, None]
            observed_mass = observed.sum(axis=-1).mean(axis=-1)
            result.update({"temporal_attention_entropy": entropy.T,
                           "temporal_recent_mass": recent.T,
                           "temporal_seasonal_mass": seasonal.T,
                           "temporal_observed_mass": observed_mass.T,
                           "temporal_history_valid": valid.astype(np.int8).sum(axis=1)[:, None].repeat(weights.shape[1], axis=1).T})
        river_diagnostics = getattr(self.model, "last_river_attention_diagnostics", None)
        if river_diagnostics is not None:
            lag_mass = river_diagnostics["lag_mass"].numpy()
            result["river_attention_entropy"] = river_diagnostics["entropy"].numpy().T
            for lag_id, lag in enumerate(tuple(int(x) for x in river_diagnostics["lags"].tolist())):
                result[f"river_lag_mass_{lag}"] = lag_mass[..., lag_id].mean(axis=2).T
        return result

    def predict(self, visible_roles=TEST_ROLES) -> np.ndarray:
        return self.predict_components(visible_roles)["final_pred"]


class AdditiveLocalTransportKGML:
    """Joint local-residual plus message-only residual model.

    The local branch keeps the K1 self/temporal representation with an empty
    message edge set. The message branch is the zero-preserving K2 module.
    They share the RF-local base and are optimized against the same OOF
    residual, so the message branch must explain error left by the local
    branch rather than duplicating an arbitrary offset.
    """

    def __init__(self, *, analyte="doc", seed=42, max_epochs=30, patience=5,
                 n_estimators=200, n_jobs=4, hidden=64, dropout=0.1, lr=1e-3,
                 chunk_months=64, epoch_callback=None):
        common = {"analyte": analyte, "seed": seed, "edge_direction": "upstream",
                  "max_epochs": max_epochs, "patience": patience,
                  "n_estimators": n_estimators, "n_jobs": n_jobs, "hidden": hidden,
                  "dropout": dropout, "lr": lr, "chunk_months": chunk_months}
        self.local = LocalTransportKGML(edge_set="empty", spatial_variant="baseline", **common)
        self.message = LocalTransportKGML(edge_set="river", spatial_variant="msgonly", **common)
        self.epoch_callback = epoch_callback
        self.seed = seed

    def initialize(self, dataset: dict, split: dict, rf: RFArtifacts):
        self.dataset, self.split, self.rf = dataset, split, rf
        self.local.initialize(dataset, split, rf)
        self.message.initialize(dataset, split, rf)

    def fit(self, dataset: dict, split: dict, rf: RFArtifacts | None = None) -> None:
        if rf is None:
            rf = fit_rf_artifacts(dataset, split, target_transform=TARGET_TRANSFORMS[self.local.analyte],
                                  seed=self.seed, n_estimators=self.local.n_estimators,
                                  n_jobs=self.local.n_jobs)
        self.initialize(dataset, split, rf)
        _n, t = self.local.inputs.y_model.shape
        train = np.asarray(split["train"], dtype=np.int64)
        val = np.asarray(split.get("val", []), dtype=np.int64)
        if not len(val):
            raise ValueError("validation cells required for checkpoint selection")
        residual = (target_values(dataset, rf.target_transform) - rf.local_oof_z) / rf.target_sd
        residual = torch.as_tensor(residual, dtype=torch.float32)
        views = []
        for stations in rf.folds:
            cells = train[np.isin(train // t, stations)]
            view = fold_split(split, stations, t)
            local_x, local_age = self.local.input_view(view)
            message_x, message_age = self.message.input_view(view)
            views.append((cells, local_x, local_age, message_x, message_age))
        local_val_x, local_val_age = self.local.input_view(split)
        message_val_x, message_val_age = self.message.input_view(split)
        base_val, _ = rf.predict_std(dataset, split, FIT_ROLES)
        base_val = torch.as_tensor(base_val.ravel()[val], dtype=torch.float32)
        parameters = list(self.local.model.parameters()) + list(self.message.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.local.lr)
        best, state, bad = float("inf"), None, 0
        self.trace = []
        for epoch in range(1, self.local.max_epochs + 1):
            self.local.model.train()
            self.message.model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for cells, local_x, local_age, message_x, message_age in views:
                local_delta = self.local.delta_tensor(local_x, local_age).reshape(-1)[cells]
                message_delta = self.message.delta_tensor(message_x, message_age).reshape(-1)[cells]
                loss = F.mse_loss(local_delta + message_delta, residual.reshape(-1)[cells]) * (len(cells) / len(train))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite additive KGML training loss")
                loss.backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            self.local.model.eval()
            self.message.model.eval()
            with torch.no_grad():
                local_delta = self.local.delta_tensor(local_val_x, local_val_age).reshape(-1)[val]
                message_delta = self.message.delta_tensor(message_val_x, message_val_age).reshape(-1)[val]
                criterion = float(F.mse_loss(base_val + local_delta + message_delta,
                                             self.local.inputs.y_model.reshape(-1)[val]))
            row = {"epoch": epoch, "loss": total_loss, "val_loss": criterion}
            self.trace.append(row)
            if self.epoch_callback:
                self.epoch_callback(row)
            if criterion < best - 1e-6:
                best, bad = criterion, 0
                self.best_epoch = epoch
                state = {
                    "local": {k: v.detach().clone() for k, v in self.local.model.state_dict().items()},
                    "message": {k: v.detach().clone() for k, v in self.message.model.state_dict().items()},
                }
            else:
                bad += 1
            if bad >= self.local.patience:
                break
        self.local.model.load_state_dict(state["local"])
        self.message.model.load_state_dict(state["message"])
        self.epochs_run, self.best_val_loss = epoch, best
        self.local.epochs_run, self.local.best_epoch, self.local.best_val_loss = epoch, self.best_epoch, best
        self.local.model.eval()
        self.message.model.eval()

    def predict_components(self, visible_roles=TEST_ROLES) -> dict:
        local_std, context_std = self.rf.predict_std(self.dataset, self.split, visible_roles)
        local_x, local_age = self.local.input_view(self.split, visible_roles)
        message_x, message_age = self.message.input_view(self.split, visible_roles)
        with torch.no_grad():
            local_delta_std = self.local.delta_tensor(local_x, local_age).numpy().astype(np.float64)
            message_delta_std = self.message.delta_tensor(message_x, message_age).numpy().astype(np.float64)
        total_delta_std = local_delta_std + message_delta_std
        local = self.rf.inverse_std(local_std)
        final = self.rf.inverse_std(local_std + total_delta_std)
        delta_scale = self.rf.target_sd if self.local.analyte != "ph" else 1.0
        return {"local_pred": local, "context_pred": self.rf.inverse_std(context_std),
                "final_pred": final, "local_delta": local_delta_std * delta_scale,
                "message_delta": message_delta_std * delta_scale,
                "graph_delta": total_delta_std * delta_scale,
                "graph_delta_std": total_delta_std,
                "graph_delta_raw": final - local,
                "graph_delta_abs": np.abs(total_delta_std * delta_scale)}

    def predict(self, visible_roles=TEST_ROLES) -> np.ndarray:
        return self.predict_components(visible_roles)["final_pred"]
