"""Training utilities for the H2X-T causal temporal extension.

This module keeps the released H2X spatial model unchanged and adds a rolling
GRU over monthly node representations.  One model is fitted per validated
analyte; the code and visibility protocol are shared across DOC, pH and
specific conductance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.temporal import TemporalTransportGCNImputer

TARGET_TRANSFORMS = {
    "doc": "log1p",
    "ph": "standard",
    "spec_conductance": "log1p",
}


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def transform_target(y: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log1p":
        if torch.any(y < 0):
            raise ValueError("log1p target transform received a negative value")
        return torch.log1p(y)
    if transform == "standard":
        return y
    raise ValueError(f"unknown target transform: {transform}")


def inverse_target(y: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log1p":
        return torch.expm1(y).clamp_min(0.0)
    if transform == "standard":
        return y
    raise ValueError(f"unknown target transform: {transform}")


@dataclass
class TemporalInputs:
    """Prepared tensors and train-derived target statistics."""

    xt_static: torch.Tensor
    y_raw: torch.Tensor
    y_model: torch.Tensor
    base_visible: torch.Tensor
    train_cells: torch.Tensor
    target_mu: torch.Tensor
    target_sd: torch.Tensor
    target_transform: str
    env_raw: torch.Tensor | None


def _split_target_stats(y_model: torch.Tensor,
                        train_cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, t = y_model.shape
    ti, tj = train_cells // t, train_cells % t
    if len(train_cells) == 0:
        raise ValueError("cannot standardize a target without train cells")
    mu = y_model[ti, tj].mean()
    sd = y_model[ti, tj].std(unbiased=False).clamp_min(1e-8)
    return mu, sd


def build_temporal_inputs(
    dataset: dict,
    split: dict[str, np.ndarray],
    *,
    target_transform: str,
    env_groups: list[str] | None = None,
    env_encoder: bool = True,
) -> TemporalInputs:
    """Build H2X-compatible monthly features without exposing hidden labels."""
    y_raw = _as_tensor(dataset["y"])
    y_mask = _as_tensor(dataset["y_mask"]).bool()
    x = _as_tensor(dataset["x"])
    x_mask = _as_tensor(dataset["x_mask"]).float()
    if y_raw.ndim != 2 or y_mask.shape != y_raw.shape:
        raise ValueError("target y/y_mask must be aligned [nodes, months]")
    n, t = y_raw.shape
    if x.shape != (n, t, 2) or x_mask.shape != x.shape:
        raise ValueError("hydro inputs must have shape [nodes, months, 2]")
    role_cells = {}
    for role in ("train", "val", "test", "context"):
        cells = np.asarray(split.get(role, np.array([], dtype=np.int64)), dtype=np.int64)
        if cells.size and (cells.min() < 0 or cells.max() >= n * t):
            raise ValueError(f"{role} cells are outside the target grid")
        if len(np.unique(cells)) != len(cells):
            raise ValueError(f"{role} cells contain duplicates")
        if cells.size and not y_mask.reshape(-1)[cells].all():
            raise ValueError(f"{role} contains an unobserved target cell")
        role_cells[role] = cells
    for i, left in enumerate(("train", "val", "test", "context")):
        for right in ("train", "val", "test", "context")[i + 1:]:
            if np.intersect1d(role_cells[left], role_cells[right]).size:
                raise ValueError(f"{left} and {right} roles overlap")
    if "train" not in split:
        raise ValueError("split must provide train cells")
    train_cells = torch.as_tensor(role_cells["train"])
    ti, tj = train_cells // t, train_cells % t

    y_model_raw = transform_target(y_raw, target_transform)
    target_mu, target_sd = _split_target_stats(y_model_raw, train_cells)
    y_model = (y_model_raw - target_mu) / target_sd

    def standardized(v: torch.Tensor) -> torch.Tensor:
        if v.shape == (n, t):
            ref = v[ti, tj]
        else:
            ref = v.reshape(-1)
        return (v - ref.mean()) / (ref.std(unbiased=False).clamp_min(1e-8))

    months = torch.tensor(
        [int(str(m)[5:7]) for m in dataset["months"]], dtype=torch.float32
    )
    season_angle = 2.0 * torch.pi * (months - 1.0) / 12.0
    season = torch.stack([torch.sin(season_angle), torch.cos(season_angle)], dim=1)

    static = _as_tensor(dataset["static"])
    if static.shape != (n, 2):
        raise ValueError("static features must have shape [nodes, 2]")
    feats = [
        standardized(x[:, :, 0]), x_mask[:, :, 0],
        standardized(x[:, :, 1]), x_mask[:, :, 1],
        season[:, 0].expand(n, t), season[:, 1].expand(n, t),
        standardized(static[:, 0:1]).expand(n, t),
        standardized(static[:, 1:2]).expand(n, t),
        torch.zeros(n, t), torch.zeros(n, t),  # target value + visibility slots
    ]

    env_raw = None
    if "regime" in dataset:
        regime = _as_tensor(dataset["regime"])
        if regime.shape != (n, 13):
            raise ValueError("regime features must have shape [nodes, 13]")
        group_idx = {
            "hydro": [0, 1, 2, 3],
            "landcover": [4, 5, 6, 7],
            "climate": [8, 9],
            "soil": [10],
            "topo": [11, 12],
        }
        keep = sorted(
            i for group in (env_groups or list(group_idx))
            for i in group_idx[group]
        )
        regime_norm = (regime - regime.mean(0)) / regime.std(0, unbiased=False).clamp_min(1e-8)
        regime_norm = regime_norm[:, keep]
        pos = {original: k for k, original in enumerate(keep)}
        hydro_cols = [pos[i] for i in keep if i < 4]
        ecological_cols = [pos[i] for i in keep if i >= 4]
        if env_encoder:
            env_raw = regime_norm[:, ecological_cols] if ecological_cols else None
            regime_for_input = regime_norm[:, hydro_cols] if hydro_cols else None
        else:
            regime_for_input = regime_norm
        if regime_for_input is not None:
            for c in range(regime_for_input.shape[1]):
                feats.append(regime_for_input[:, c:c + 1].expand(n, t))

    xt_static = torch.stack(feats, dim=-1).permute(1, 0, 2).contiguous()
    base_visible = torch.zeros(n * t, dtype=torch.bool)
    for key in ("val", "context"):
        cells = np.asarray(split.get(key, np.array([], dtype=np.int64)), dtype=np.int64)
        if cells.size:
            base_visible[cells] = True
    return TemporalInputs(
        xt_static=xt_static,
        y_raw=y_raw,
        y_model=y_model,
        base_visible=base_visible.reshape(n, t),
        train_cells=train_cells,
        target_mu=target_mu,
        target_sd=target_sd,
        target_transform=target_transform,
        env_raw=env_raw,
    )


def fill_target_channel(xt_static: torch.Tensor, y_model: torch.Tensor,
                        visible: torch.Tensor) -> torch.Tensor:
    """Return an input copy with the visible target channel populated."""
    if visible.shape != y_model.shape:
        raise ValueError("visible and target shapes differ")
    xt = xt_static.clone()
    observed = torch.where(visible, y_model, torch.zeros_like(y_model))
    xt[:, :, 8] = observed.T
    xt[:, :, 9] = visible.float().T
    return xt


class H2XTemporalModel:
    """Train/predict wrapper for one analyte using the H2X-T architecture."""

    def __init__(
        self,
        *,
        seed: int = 42,
        lookback: int = 12,
        temporal_hidden: int = 64,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 50,
        patience: int = 10,
        weight_decay: float = 0.0,
        env_groups: list[str] | None = None,
        env_encoder: bool = True,
        edge_set: str = "river",
        edge_direction: str = "both",
        target_transform: str = "log1p",
        chunk_months: int = 256,
        history_ablation: str = "none",
    ):
        self.seed = int(seed)
        self.lookback = int(lookback)
        self.temporal_hidden = int(temporal_hidden)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.env_groups = env_groups
        self.env_encoder = bool(env_encoder)
        self.edge_set = edge_set
        self.edge_direction = edge_direction
        self.target_transform = target_transform
        self.chunk_months = int(chunk_months)
        self.history_ablation = history_ablation

    def _build_model(self, inputs: TemporalInputs, dataset: dict) -> TemporalTransportGCNImputer:
        edge_attr = _as_tensor(dataset["edge_attr"])
        edge_attr = (edge_attr - edge_attr.mean(0)) / edge_attr.std(0, unbiased=False).clamp_min(1e-8)
        edge_index = _as_tensor(dataset["edge_index"]).long()
        if self.edge_set == "empty":
            edge_index = torch.empty((2, 0), dtype=torch.long)
        spatial = TransportGCNImputer(
            in_channels=inputs.xt_static.shape[-1],
            edge_dim=edge_attr.shape[1],
            hidden=self.hidden,
            layers=self.layers,
            dropout=self.dropout,
            env_dim=inputs.env_raw.shape[1] if inputs.env_raw is not None else 0,
            edge_direction=self.edge_direction,
        )
        model = TemporalTransportGCNImputer(
            spatial,
            lookback=self.lookback,
            temporal_hidden=self.temporal_hidden,
            chunk_months=self.chunk_months,
            history_ablation=self.history_ablation,
        )
        model._edge_index = edge_index
        model._edge_attr = edge_attr
        return model

    @staticmethod
    def _target_cells(cells: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        return cells // t, cells % t

    def _predict_model(self, model: TemporalTransportGCNImputer,
                       xt: torch.Tensor, inputs: TemporalInputs,
                       dataset: dict) -> torch.Tensor:
        return model.forward_sequence(
            xt,
            model._edge_index,
            model._edge_attr,
            inputs.env_raw,
        ).T

    def fit(self, dataset: dict, split: dict[str, np.ndarray]) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        transform = self.target_transform
        inputs = build_temporal_inputs(
            dataset,
            split,
            target_transform=transform,
            env_groups=self.env_groups,
            env_encoder=self.env_encoder,
        )
        model = self._build_model(inputs, dataset)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        _, t = inputs.y_model.shape
        train_cells = inputs.train_cells
        val_cells = torch.as_tensor(
            np.asarray(split.get("val", np.array([], dtype=np.int64)), dtype=np.int64)
        )
        base_train = inputs.base_visible.clone()
        if val_cells.numel():
            base_train.reshape(-1)[val_cells] = False
        best_loss, best_state, bad = float("inf"), None, 0

        for epoch in range(self.max_epochs):
            perm = rng.permutation(train_cells.numpy())
            half = len(perm) // 2
            context = torch.as_tensor(perm[:half], dtype=torch.long)
            targets = torch.as_tensor(perm[half:], dtype=torch.long)
            visible = base_train.clone()
            if context.numel():
                visible.reshape(-1)[context] = True
            xt = fill_target_channel(inputs.xt_static, inputs.y_model, visible)
            model.train()
            pred = self._predict_model(model, xt, inputs, dataset)
            ti, tj = self._target_cells(targets, t)
            loss = F.mse_loss(pred[ti, tj], inputs.y_model[ti, tj])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_visible = base_train.clone()
                if train_cells.numel():
                    eval_visible.reshape(-1)[train_cells] = True
                eval_xt = fill_target_channel(inputs.xt_static, inputs.y_model, eval_visible)
                eval_pred = self._predict_model(model, eval_xt, inputs, dataset)
                if val_cells.numel():
                    vi, vj = self._target_cells(val_cells, t)
                    criterion = F.mse_loss(eval_pred[vi, vj], inputs.y_model[vi, vj]).item()
                else:
                    criterion = float(loss.item())
            if criterion < best_loss - 1e-6:
                best_loss = criterion
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if val_cells.numel() and bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._bundle = {
            "model": model,
            "inputs": inputs,
            "dataset": dataset,
            "split": split,
            "best_val_loss": best_loss,
            "epochs_run": epoch + 1,
        }

    def _visible_input(
        self,
        inputs: TemporalInputs,
        *,
        only_visible: np.ndarray | torch.Tensor | None,
        extra_visible: np.ndarray | torch.Tensor | None,
        extra_values: np.ndarray | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, t = inputs.y_model.shape
        if only_visible is not None:
            visible = torch.zeros(n * t, dtype=torch.bool)
            cells = torch.as_tensor(np.asarray(only_visible, dtype=np.int64)).reshape(-1)
            if cells.numel():
                if int(cells.min()) < 0 or int(cells.max()) >= n * t:
                    raise ValueError("only_visible out of range")
                visible[cells] = True
            visible = visible.reshape(n, t)
        else:
            visible = inputs.base_visible.clone()
            visible.reshape(-1)[inputs.train_cells] = True
        y_feed = inputs.y_model.clone()
        if extra_visible is not None:
            cells = torch.as_tensor(np.asarray(extra_visible, dtype=np.int64)).reshape(-1)
            if cells.numel():
                if int(cells.min()) < 0 or int(cells.max()) >= n * t:
                    raise ValueError("extra_visible out of range")
                visible.reshape(-1)[cells] = True
                if extra_values is not None:
                    values = torch.as_tensor(
                        np.asarray(extra_values, dtype=np.float32)
                    ).reshape(-1)
                    if values.numel() != cells.numel():
                        raise ValueError("extra_values must match extra_visible")
                    raw = transform_target(values, inputs.target_transform)
                    y_feed.reshape(-1)[cells] = (
                        raw - inputs.target_mu
                    ) / inputs.target_sd
        return visible, y_feed

    def predict(
        self,
        *,
        only_visible: np.ndarray | torch.Tensor | None = None,
        extra_visible: np.ndarray | torch.Tensor | None = None,
        extra_values: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "_bundle"):
            raise RuntimeError("call fit() before predict()")
        bundle = self._bundle
        inputs: TemporalInputs = bundle["inputs"]
        model: TemporalTransportGCNImputer = bundle["model"]
        visible, y_feed = self._visible_input(
            inputs,
            only_visible=only_visible,
            extra_visible=extra_visible,
            extra_values=extra_values,
        )
        xt = fill_target_channel(inputs.xt_static, y_feed, visible)
        model.eval()
        with torch.no_grad():
            pred_std = self._predict_model(model, xt, inputs, bundle["dataset"])
        train_i, train_j = self._target_cells(inputs.train_cells, inputs.y_model.shape[1])
        train_values = inputs.y_model[train_i, train_j]
        if inputs.target_transform == "log1p":
            lo, hi = train_values.min(), torch.quantile(train_values, 0.995)
        else:
            lo, hi = train_values.min(), train_values.max()
        # ``train_values`` is already in standardized model space.  Applying
        # the raw-space affine transform a second time here would make pH and
        # other non-log targets clip to the wrong interval.
        pred_std = pred_std.clamp(lo, hi)
        pred_raw_model = pred_std * inputs.target_sd + inputs.target_mu
        pred_raw = inverse_target(pred_raw_model, inputs.target_transform)
        return pred_raw.numpy()

    def fit_predict(self, dataset: dict, split: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        self.fit(dataset, split)
        return self.predict(**kwargs)
