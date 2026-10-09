"""Observation-driven graph upgrade experiments.

This module is intentionally separate from the released H2X-T wrapper.  It
provides a small, research-facing interface for testing three mechanisms:
``m1`` observation-aware memory, ``m2`` lagged river transport, ``m3``
multi-scale temporal features, and ``m13`` their observation-aware
multi-scale combination.  Each mechanism uses the same split and target
transform code as H2X-T, which makes comparisons to the existing model direct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from river_graph.experiments.temporal_h2x import (
    TARGET_TRANSFORMS,
    TemporalInputs,
    _as_tensor,
    build_temporal_inputs,
    fill_target_channel,
    inverse_target,
    transform_target,
)
from river_graph.models.graph_upgrade import (
    DualMessageOnlyTransportGCNImputer,
    LaggedTransportTemporalImputer,
    MessageOnlyTemporalTransportImputer,
    MessageOnlyTransportGCNImputer,
    MessageResidualTransportGCNImputer,
    MultiScaleTemporalTransportImputer,
    ObservationAwareAttentionTemporalTransportGCNImputer,
    ObservationAwareMultiScaleTemporalTransportImputer,
    ObservationAwareTemporalTransportGCNImputer,
    ObservationGatedTransportGCNImputer,
    RiverLagAttentionTemporalTransportGCNImputer,
)
from river_graph.models.hydro import TransportGCNImputer

MECHANISMS = ("m1", "m2", "m3", "m13")
SPATIAL_VARIANTS = ("baseline", "res2", "res3", "res4", "res3_jk", "msgres2", "msggate2", "msgonly", "msgdual")
OBS_FEATURE_NAMES = (
    "last_observed_value",
    "observation_age_log",
    "last_observation_valid",
    "support_count_1",
    "support_count_3",
    "support_count_6",
    "support_count_12",
    "upstream_visible_fraction",
    "downstream_visible_fraction",
)


def _rolling_fraction(values: torch.Tensor, width: int) -> torch.Tensor:
    """Causal rolling fraction for ``[time, nodes]`` visibility values."""
    t = values.shape[0]
    csum = torch.cat([values.new_zeros((1, values.shape[1])), values.cumsum(0)], dim=0)
    result = values.new_zeros(values.shape)
    for j in range(t):
        left = max(0, j - width + 1)
        result[j] = (csum[j + 1] - csum[left]) / float(width)
    return result


def observation_statistics(y_model: torch.Tensor, visible: torch.Tensor,
                           edge_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return causal observation age, last value and network support features.

    All computations use ``visible`` as the information boundary.  Values in
    ``y_model`` at hidden cells are never read.  The result follows the model
    convention ``[time, nodes]``.
    """
    if y_model.shape != visible.shape or y_model.ndim != 2:
        raise ValueError("y_model and visible must be aligned [nodes, months]")
    n, t = y_model.shape
    age = y_model.new_zeros((n, t))
    last = y_model.new_zeros((n, t))
    valid = torch.zeros((n, t), dtype=torch.bool)
    current_age = y_model.new_zeros((n,))
    current_last = y_model.new_zeros((n,))
    current_valid = torch.zeros((n,), dtype=torch.bool)
    for j in range(t):
        current_age = current_age + 1.0
        observed = visible[:, j]
        current_last = torch.where(observed, y_model[:, j], current_last)
        current_age = torch.where(observed, torch.zeros_like(current_age), current_age)
        current_valid = current_valid | observed
        last[:, j] = torch.where(current_valid, current_last, torch.zeros_like(current_last))
        age[:, j] = current_age
        valid[:, j] = current_valid

    visible_t = visible.float().T
    counts = [visible_t]
    counts.extend(_rolling_fraction(visible_t, width) for width in (3, 6, 12))

    upstream = visible_t.new_zeros((t, n))
    downstream = visible_t.new_zeros((t, n))
    if edge_index.numel():
        src, dst = edge_index
        upstream.index_add_(1, dst, visible_t[:, src])
        downstream.index_add_(1, src, visible_t[:, dst])
        up_degree = visible_t.new_zeros((n,)).index_add_(0, dst, visible_t.new_ones(len(dst)))
        down_degree = visible_t.new_zeros((n,)).index_add_(0, src, visible_t.new_ones(len(src)))
        upstream = upstream / up_degree.clamp_min(1.0)[None, :]
        downstream = downstream / down_degree.clamp_min(1.0)[None, :]
    # A fixed scale preserves the prefix when future months are appended.
    # A separate valid flag distinguishes never observed from old readings.
    scale = float(np.log1p(12))
    age_log = torch.log1p(age) / scale
    return (
        last.T,
        age_log.T,
        valid.float().T,
        *counts,
        upstream,
        downstream,
    )


def build_observation_features(inputs: TemporalInputs, visible: torch.Tensor,
                               edge_index: torch.Tensor,
                               y_feed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Populate target inputs and append M1 observation features.

    Returns ``(x, age)`` with x in ``[time, nodes, channels]`` and age in
    ``[time, nodes]``.  ``age`` is kept separately so the GRU-D module can
    attenuate hidden states without relying on a hard-coded channel index.
    """
    values = inputs.y_model if y_feed is None else y_feed
    base = fill_target_channel(
        inputs.xt_static, values, visible, context_mode=inputs.context_mode,
        edge_index=edge_index,
    )
    stats = observation_statistics(values, visible, edge_index)
    extra = torch.stack(stats, dim=-1)
    return torch.cat([base, extra], dim=-1), stats[1]


def sample_training_view(train_cells: torch.Tensor, n: int, t: int,
                         rng: np.random.Generator, mode: str | None = None,
                         block_months: int = 3) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Split train labels into visible context and loss targets.

    The three masking patterns are deliberately simple and interpretable:
    random points, short contiguous temporal blocks, and whole stations.  The
    returned target cells are always a subset of the frozen train role.
    """
    if mode is None:
        mode = str(rng.choice(("point", "temporal_block", "station_block"), p=(.4, .3, .3)))
    cells = train_cells.detach().cpu().numpy().astype(np.int64)
    if len(cells) < 2:
        raise ValueError("at least two train cells are required")
    coords = np.column_stack((cells // t, cells % t))
    if mode == "point":
        perm = rng.permutation(len(cells))
        target = cells[perm[len(perm) // 2:]]
    elif mode == "temporal_block":
        target_mask = np.zeros(len(cells), dtype=bool)
        for station in np.unique(coords[:, 0]):
            idx = np.flatnonzero(coords[:, 0] == station)
            if len(idx) < 2:
                continue
            months = coords[idx, 1]
            start = int(rng.choice(months))
            target_mask[idx] = (months >= start) & (months < start + block_months)
        target = cells[target_mask]
        if len(target) == 0 or len(target) == len(cells):
            return sample_training_view(train_cells, n, t, rng, "point", block_months)
    elif mode == "station_block":
        stations = np.unique(coords[:, 0])
        chosen = stations[rng.random(len(stations)) < .2]
        if len(chosen) == 0 or len(chosen) == len(stations):
            chosen = rng.choice(stations, size=1, replace=False)
        target = cells[np.isin(coords[:, 0], chosen)]
        if len(target) == 0 or len(target) == len(cells):
            return sample_training_view(train_cells, n, t, rng, "point", block_months)
    else:
        raise ValueError("mode must be point, temporal_block, or station_block")
    context = np.setdiff1d(cells, target, assume_unique=False)
    if len(context) == 0 or len(target) == 0:
        return sample_training_view(train_cells, n, t, rng, "point", block_months)
    return torch.as_tensor(context), torch.as_tensor(target), mode


@dataclass
class UpgradeBundle:
    model: torch.nn.Module
    inputs: TemporalInputs
    dataset: dict
    split: dict[str, np.ndarray]
    best_val_loss: float
    epochs_run: int
    masking_history: list[str]


class GraphUpgradeModel:
    """Fit one observation/transport mechanism for one analyte."""

    def __init__(self, *, mechanism: str = "m1", seed: int = 42,
                 lookback: int = 12, temporal_hidden: int = 64,
                 hidden: int = 64, layers: int = 2, dropout: float = .1,
                 lr: float = 1e-3, max_epochs: int = 30, patience: int = 5,
                 target_transform: str = "log1p", env_groups: list[str] | None = None,
                 env_encoder: bool = True, edge_direction: str = "both",
                 context_mode: str = "none", edge_set: str = "river",
                 chunk_months: int = 32, masking: str = "mixed",
                 epoch_callback=None, lag_mode: str = "learned",
                 spatial_variant: str = "baseline",
                 feature_edge_set: str | None = None,
                 message_feature_mode: str = "all",
                 temporal_operator: str = "gru",
                 attention_heads: int = 2,
                 attention_dropout: float = 0.1):
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism: {mechanism}")
        if spatial_variant not in SPATIAL_VARIANTS:
            raise ValueError(f"unknown spatial variant: {spatial_variant}")
        if message_feature_mode not in ("all", "target_only", "hydro_ecology"):
            raise ValueError("invalid message feature mode")
        if temporal_operator not in ("gru", "gru_attention", "gru_attention_river"):
            raise ValueError("invalid temporal_operator")
        if temporal_operator in ("gru_attention", "gru_attention_river") and mechanism != "m1":
            raise ValueError("attention temporal operators currently support mechanism='m1' only")
        if temporal_operator in ("gru_attention", "gru_attention_river") and spatial_variant in ("msgonly", "msgdual"):
            raise ValueError("attention temporal operators are not available for message-only variants")
        if temporal_operator == "gru_attention_river" and edge_direction != "upstream":
            raise ValueError("river attention requires edge_direction='upstream'")
        if temporal_operator == "gru_attention_river" and lookback < 13:
            raise ValueError("river attention requires lookback >= 13 for the 12-month lag")
        if mechanism == "m2" and spatial_variant != "baseline":
            raise ValueError("lagged transport currently supports only the baseline spatial trunk")
        self.mechanism = mechanism
        self.temporal_operator = temporal_operator
        if attention_heads < 1 or attention_dropout < 0 or attention_dropout >= 1:
            raise ValueError("invalid attention settings")
        self.attention_heads = int(attention_heads)
        self.attention_dropout = float(attention_dropout)
        self.spatial_variant = spatial_variant
        self.message_feature_mode = message_feature_mode
        self.seed, self.lookback, self.temporal_hidden = int(seed), int(lookback), int(temporal_hidden)
        self.hidden, self.layers, self.dropout = int(hidden), int(layers), float(dropout)
        self.lr, self.max_epochs, self.patience = float(lr), int(max_epochs), int(patience)
        self.target_transform = target_transform
        self.env_groups, self.env_encoder = env_groups, bool(env_encoder)
        self.edge_direction, self.context_mode, self.edge_set = edge_direction, context_mode, edge_set
        self.feature_edge_set = feature_edge_set or edge_set
        if (edge_set not in ("river", "empty")
                or self.feature_edge_set not in ("river", "empty")
                or masking not in ("mixed", "point")):
            raise ValueError("invalid edge set or masking mode")
        if max_epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must be positive")
        self.chunk_months, self.masking = int(chunk_months), masking
        self.epoch_callback = epoch_callback
        self.lag_mode = lag_mode

    @staticmethod
    def _edges(dataset: dict, edge_set: str) -> tuple[torch.Tensor, torch.Tensor]:
        edge_attr = _as_tensor(dataset["edge_attr"])
        edge_attr = (edge_attr - edge_attr.mean(0)) / edge_attr.std(0, unbiased=False).clamp_min(1e-8)
        edge_index = _as_tensor(dataset["edge_index"]).long()
        if edge_set == "empty":
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = edge_attr[:0]
        return edge_index, edge_attr

    def _build_model(self, inputs: TemporalInputs, dataset: dict) -> torch.nn.Module:
        edge_index, edge_attr = self._edges(dataset, self.edge_set)
        feature_edge_index, _ = self._edges(dataset, self.feature_edge_set)
        variant_layers = {"baseline": self.layers, "res2": 2,
                          "res3": 3, "res4": 4, "res3_jk": 3,
                          "msgres2": 2, "msggate2": 2, "msgonly": 2, "msgdual": 2}
        spatial_layers = variant_layers[self.spatial_variant]
        residual = self.spatial_variant in {"res2", "res3", "res4", "res3_jk"}
        jumping_knowledge = self.spatial_variant == "res3_jk"
        spatial_cls = {
            "msgres2": MessageResidualTransportGCNImputer,
            "msggate2": ObservationGatedTransportGCNImputer,
            "msgonly": MessageOnlyTransportGCNImputer,
            "msgdual": DualMessageOnlyTransportGCNImputer,
        }.get(self.spatial_variant, TransportGCNImputer)
        spatial = spatial_cls(
            in_channels=inputs.xt_static.shape[-1] + (
                len(OBS_FEATURE_NAMES) if self.mechanism in ("m1", "m2", "m3", "m13") else 0
            ),
            edge_dim=edge_attr.shape[1], hidden=self.hidden, layers=spatial_layers,
            dropout=self.dropout, env_dim=inputs.env_raw.shape[1] if inputs.env_raw is not None else 0,
            edge_direction=self.edge_direction,
            residual=residual, jumping_knowledge=jumping_knowledge,
            **({"target_base_end": 10 + {"none": 0, "global": 2, "directional": 6, "all": 8}[inputs.context_mode]}
               if self.spatial_variant == "msgdual" else {}),
        )
        cls = {
            "m1": ObservationAwareTemporalTransportGCNImputer,
            "m2": LaggedTransportTemporalImputer,
            "m3": MultiScaleTemporalTransportImputer,
            "m13": ObservationAwareMultiScaleTemporalTransportImputer,
        }[self.mechanism]
        if self.temporal_operator == "gru_attention":
            # The attention operator is an additive upgrade to the M1 GRU-D
            # path.  Keeping this replacement here (rather than branching in
            # training or prediction) means all existing visibility and
            # masking code is shared with the historical GRU implementation.
            cls = ObservationAwareAttentionTemporalTransportGCNImputer
        elif self.temporal_operator == "gru_attention_river":
            cls = RiverLagAttentionTemporalTransportGCNImputer
        if self.spatial_variant in ("msgonly", "msgdual"):
            if self.mechanism != "m1":
                raise ValueError("message-only source isolation uses M1 memory")
            cls = MessageOnlyTemporalTransportImputer
        model = cls(spatial, lookback=self.lookback,
                    temporal_hidden=self.temporal_hidden, chunk_months=self.chunk_months,
                    **({"attention_heads": self.attention_heads,
                        "attention_dropout": self.attention_dropout}
                       if self.temporal_operator in ("gru_attention", "gru_attention_river") else {}),
                    **({"lag_mode": self.lag_mode} if self.mechanism == "m2" else {}))
        model._edge_index, model._edge_attr = edge_index, edge_attr
        # Observation support features can remain matched between river and
        # no-message controls while the spatial encoder receives no edges.
        model._feature_edge_index = feature_edge_index
        return model

    @staticmethod
    def _cells(cells: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        return cells // t, cells % t

    def _forward(self, model, x, inputs, dataset, age=None):
        env_raw = inputs.env_raw
        if self.spatial_variant == "msgonly" and self.message_feature_mode != "all":
            # The message-only branch is used for source isolation.  Its
            # input channels are masked here, before any edge aggregation,
            # while the local RF base and the no-message null stay unchanged.
            base_channels = inputs.xt_static.shape[-1]
            context_count = {"none": 0, "global": 2, "directional": 6, "all": 8}[inputs.context_mode]
            target_base_end = 10 + context_count
            target_stats_start = base_channels
            target_stats_end = base_channels + len(OBS_FEATURE_NAMES)
            channel_mask = torch.ones_like(x)
            if self.message_feature_mode == "target_only":
                channel_mask.zero_()
                channel_mask[..., 8:target_base_end] = 1.0
                channel_mask[..., target_stats_start:target_stats_end] = 1.0
                env_raw = torch.zeros_like(env_raw) if env_raw is not None else None
            else:  # hydro_ecology: remove target values and all target statistics
                channel_mask[..., 8:target_base_end] = 0.0
                channel_mask[..., target_stats_start:target_stats_end] = 0.0
                age = torch.zeros_like(age) if age is not None else None
            x = x * channel_mask
        if self.mechanism in ("m1", "m2", "m13"):
            return model.forward_sequence(x, model._edge_index, model._edge_attr,
                                          env_raw, age_seq=age).T
        return model.forward_sequence(x, model._edge_index, model._edge_attr,
                                      env_raw).T

    def _make_input(self, inputs, visible, model, y_feed=None):
        if self.mechanism in ("m1", "m2", "m3", "m13"):
            return build_observation_features(
                inputs, visible, model._feature_edge_index, y_feed=y_feed
            )
        values = inputs.y_model if y_feed is None else y_feed
        return fill_target_channel(
            inputs.xt_static, values, visible,
            context_mode=inputs.context_mode, edge_index=model._edge_index,
        ), None

    def fit(self, dataset: dict, split: dict[str, np.ndarray]) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        inputs = build_temporal_inputs(
            dataset, split, target_transform=self.target_transform,
            env_groups=self.env_groups, env_encoder=self.env_encoder,
            context_mode=self.context_mode,
        )
        model = self._build_model(inputs, dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        _, t = inputs.y_model.shape
        train_cells = inputs.train_cells
        val_cells = torch.as_tensor(np.asarray(split.get("val", []), dtype=np.int64))
        base_train = inputs.base_visible.clone()
        if val_cells.numel():
            base_train.reshape(-1)[val_cells] = False
        best_loss, best_state, bad = float("inf"), None, 0
        masking_history = []
        for epoch in range(self.max_epochs):
            context, targets, mode = sample_training_view(
                train_cells, *inputs.y_model.shape, rng=rng,
                mode="point" if self.masking == "point" else None,
            )
            masking_history.append(mode)
            visible = base_train.clone()
            visible.reshape(-1)[context] = True
            x, age = self._make_input(inputs, visible, model)
            model.train()
            pred = self._forward(model, x, inputs, dataset, age)
            ti, tj = self._cells(targets, t)
            loss = F.mse_loss(pred[ti, tj], inputs.y_model[ti, tj])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch + 1}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                eval_visible = base_train.clone()
                eval_visible.reshape(-1)[train_cells] = True
                eval_x, eval_age = self._make_input(inputs, eval_visible, model)
                eval_pred = self._forward(model, eval_x, inputs, dataset, eval_age)
                if val_cells.numel():
                    vi, vj = self._cells(val_cells, t)
                    criterion = F.mse_loss(eval_pred[vi, vj], inputs.y_model[vi, vj]).item()
                else:
                    criterion = float(loss.item())
            if not np.isfinite(criterion):
                raise FloatingPointError(f"nonfinite validation loss at epoch {epoch + 1}")
            if self.epoch_callback:
                self.epoch_callback({"epoch": epoch + 1, "loss": float(loss.detach()),
                                     "val_loss": criterion, "masking": mode,
                                     "target_count": len(targets)})
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
        self._bundle = UpgradeBundle(
            model=model, inputs=inputs, dataset=dataset, split=split,
            best_val_loss=best_loss, epochs_run=epoch + 1,
            masking_history=masking_history,
        )

    def _visible_input(self, *, only_visible=None, extra_visible=None,
                       extra_values=None):
        bundle = self._bundle
        inputs = bundle.inputs
        n, t = inputs.y_model.shape
        if only_visible is not None:
            visible = torch.zeros(n * t, dtype=torch.bool)
            cells = torch.as_tensor(np.asarray(only_visible, dtype=np.int64)).reshape(-1)
            if cells.numel():
                if cells.min() < 0 or cells.max() >= n * t:
                    raise ValueError("only_visible out of range")
                visible[cells] = True
            visible = visible.reshape(n, t)
        else:
            visible = inputs.base_visible.clone()
            visible.reshape(-1)[inputs.train_cells] = True
        y_feed = inputs.y_model.clone()
        if extra_visible is not None:
            cells = torch.as_tensor(np.asarray(extra_visible, dtype=np.int64)).reshape(-1)
            if cells.numel() and (cells.min() < 0 or cells.max() >= n * t):
                raise ValueError("extra_visible out of range")
            visible.reshape(-1)[cells] = True
            if extra_values is not None:
                values = torch.as_tensor(np.asarray(extra_values, dtype=np.float32)).reshape(-1)
                if values.numel() != cells.numel() or not torch.isfinite(values).all():
                    raise ValueError("extra_values must be finite and align with extra_visible")
                raw = transform_target(values, inputs.target_transform)
                y_feed.reshape(-1)[cells] = (raw - inputs.target_mu) / inputs.target_sd
        return visible, y_feed

    def predict(self, *, only_visible=None, extra_visible=None,
                extra_values=None) -> np.ndarray:
        if not hasattr(self, "_bundle"):
            raise RuntimeError("call fit() before predict()")
        bundle = self._bundle
        model, inputs = bundle.model, bundle.inputs
        visible, y_feed = self._visible_input(
            only_visible=only_visible, extra_visible=extra_visible,
            extra_values=extra_values,
        )
        x, age = self._make_input(inputs, visible, model, y_feed=y_feed)
        model.eval()
        with torch.no_grad():
            pred_std = self._forward(model, x, inputs, bundle.dataset, age)
        train_i, train_j = self._cells(inputs.train_cells, inputs.y_model.shape[1])
        train_values = inputs.y_model[train_i, train_j]
        lo = train_values.min()
        hi = torch.quantile(train_values, .995) if inputs.target_transform == "log1p" else train_values.max()
        pred_std = pred_std.clamp(lo, hi)
        pred_model = pred_std * inputs.target_sd + inputs.target_mu
        return inverse_target(pred_model, inputs.target_transform).numpy()

    def fit_predict(self, dataset: dict, split: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        self.fit(dataset, split)
        return self.predict(**kwargs)


__all__ = [
    "MECHANISMS",
    "OBS_FEATURE_NAMES",
    "SPATIAL_VARIANTS",
    "TARGET_TRANSFORMS",
    "GraphUpgradeModel",
    "build_observation_features",
    "observation_statistics",
    "sample_training_view",
]
