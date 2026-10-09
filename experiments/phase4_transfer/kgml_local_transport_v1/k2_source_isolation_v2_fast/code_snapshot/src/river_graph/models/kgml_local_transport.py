"""Fixed local forests plus a causal, directed river residual.

RF-context reproduces the existing Temporal RF feature set. RF-local adds
strictly past local observation statistics, with no other-station features.
Residual training uses five fixed station-held-out views and OOF base errors.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
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


def fit_oof_fold(dataset: dict, split: dict, held_stations: np.ndarray, *,
                 target_transform: str, seed: int, n_estimators: int, n_jobs: int) -> tuple:
    """Fit a fold without using its targets, including in target statistics."""
    t = dataset["y"].shape[1]
    train = np.asarray(split["train"], dtype=np.int64)
    held = train[np.isin(train // t, held_stations)]
    view = fold_split(split, held_stations, t)
    x = build_rf_features(dataset, view, FIT_ROLES, target_transform=target_transform, include_network=False)
    y = target_values(dataset, target_transform).ravel()
    # Trees fit directly in log1p/raw units; thus no held-label normalizer.
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed)
    rf.fit(x[view["train"]], y[view["train"]])
    return held, rf.predict(x[held])


def fit_rf_artifacts(dataset: dict, split: dict, *, target_transform: str, seed: int,
                     n_estimators: int = 200, n_jobs: int = 4) -> RFArtifacts:
    y = target_values(dataset, target_transform)
    train = np.asarray(split["train"], dtype=np.int64)
    ref = torch.as_tensor(y.ravel()[train])
    mu, sd = float(ref.mean()), float(ref.std(unbiased=False).clamp_min(1e-8))
    forests = []
    for network in (False, True):
        x = build_rf_features(dataset, split, FIT_ROLES, target_transform=target_transform, include_network=network)
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed)
        rf.fit(x[train], y.ravel()[train])
        forests.append(rf)
    folds = station_folds(train, y.shape[1], seed)
    oof = np.full(y.shape, np.nan)
    for stations in folds:
        cells, pred = fit_oof_fold(dataset, split, stations, target_transform=target_transform,
                                   seed=seed, n_estimators=n_estimators, n_jobs=n_jobs)
        oof.ravel()[cells] = pred
    if not np.isfinite(oof.ravel()[train]).all():
        raise ValueError("OOF coverage incomplete")
    return RFArtifacts(*forests, oof, folds, mu, sd, target_transform)


def residual_inputs(builder, inputs, visible, model):
    x, age = builder._make_input(inputs, visible, model)
    # All residual arms share upstream support features. Remove downstream
    # support from ALL arms so upstream-only has no reverse information path.
    x[..., -1] = 0
    return x, age


class LocalTransportKGML:
    """RF-local + signed residual in train-standardized target space."""

    def __init__(self, *, analyte="doc", seed=42, edge_direction="upstream", edge_set="river",
                 max_epochs=30, patience=5, n_estimators=200, n_jobs=4, hidden=64,
                 dropout=0.1, lr=1e-3, chunk_months=64, epoch_callback=None,
                 spatial_variant="baseline"):
        if analyte not in TARGET_TRANSFORMS or edge_direction not in ("upstream", "both"):
            raise ValueError("invalid analyte/direction")
        if edge_set not in ("river", "empty") or min(max_epochs, patience, n_estimators, hidden, chunk_months) < 1:
            raise ValueError("invalid model settings")
        self.analyte, self.seed = analyte, seed
        self.edge_direction, self.edge_set = edge_direction, edge_set
        self.max_epochs, self.patience = max_epochs, patience
        self.n_estimators, self.n_jobs = n_estimators, n_jobs
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.chunk_months, self.epoch_callback = chunk_months, epoch_callback
        if spatial_variant not in ("baseline", "msgonly"):
            raise ValueError("invalid spatial_variant")
        self.spatial_variant = spatial_variant

    def initialize(self, dataset: dict, split: dict, rf: RFArtifacts):
        """Build the zero-residual model; also supports a real initialization test."""
        torch.manual_seed(self.seed)
        self.dataset, self.split, self.rf = dataset, split, rf
        self.inputs = build_temporal_inputs(dataset, split, target_transform=TARGET_TRANSFORMS[self.analyte],
                                            env_encoder=True, context_mode="none")
        self.builder = GraphUpgradeModel(mechanism="m1", seed=self.seed, lookback=12,
            hidden=self.hidden, temporal_hidden=self.hidden, layers=2, dropout=self.dropout,
            target_transform=TARGET_TRANSFORMS[self.analyte], env_encoder=True,
            edge_direction=self.edge_direction, edge_set=self.edge_set, feature_edge_set="river",
            chunk_months=self.chunk_months, spatial_variant=self.spatial_variant)
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
                                  seed=self.seed, n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        self.initialize(dataset, split, rf)
        _n, t = self.inputs.y_model.shape
        train, val = (np.asarray(split.get(r, []), dtype=np.int64) for r in ("train", "val"))
        if not len(val):
            raise ValueError("validation cells required for checkpoint selection")
        residual = (target_values(dataset, rf.target_transform) - rf.local_oof_z) / rf.target_sd
        residual = torch.as_tensor(residual, dtype=torch.float32)
        views = []
        for stations in rf.folds:
            cells = train[np.isin(train // t, stations)]
            x, age = self.input_view(fold_split(split, stations, t))
            views.append((cells, x, age))
        val_x, val_age = self.input_view(split)
        base_val, _ = rf.predict_std(dataset, split, FIT_ROLES)
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
        final = self.rf.inverse_std(local_std + delta_std)
        if not np.isfinite(final).all():
            raise FloatingPointError("nonfinite KGML full-grid prediction")
        # graph_delta is in transform space (log1p DOC/EC, standardized pH).
        delta = delta_std if self.analyte == "ph" else delta_std * self.rf.target_sd
        return {"local_pred": local, "context_pred": self.rf.inverse_std(context_std),
                "final_pred": final, "graph_delta": delta, "graph_delta_std": delta_std,
                "graph_delta_raw": final - local, "graph_delta_abs": np.abs(delta)}

    def predict(self, visible_roles=TEST_ROLES) -> np.ndarray:
        return self.predict_components(visible_roles)["final_pred"]
