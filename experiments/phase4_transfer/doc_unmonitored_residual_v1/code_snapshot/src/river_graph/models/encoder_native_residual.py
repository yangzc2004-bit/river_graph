"""Native DOC correction with selected, end-to-end spatial self-path tuning.

This is the empty-edge expert used by the existing residual model. It does
not add river messages. Raw input views and ecological normalization remain
caller-frozen; only the requested self/ecology parameters are unfrozen.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from river_graph.models.episodic_temporal_adapter import (
    _parameter_distance,
    _state_copy,
)
from river_graph.models.hydro import TransportGCNImputer
from river_graph.models.native_temporal_residual import NativeTemporalResidual


class EncoderNativeResidual(NativeTemporalResidual):
    """Reuse the native residual protocol while differentiating raw encoding.

    Inputs contain raw[N,T,C], env[N,E], age[N,T], support[N,T,3] and
    extra[N,T,F]. The default head uses the existing 30 flow/regime/context
    features and interactions (0,2,4,28). All spatial dropout stays disabled,
    including during fitting. No label-based transformation is fitted here.

    Optional ``daily_history[N,T,8]`` enters the existing recurrent state through
    a zero-initialized, bias-free projection. ``current_only`` injects it at the
    queried month; ``full_history`` injects it at every valid causal window step.
    The default ``off`` path and its checkpoint schema remain unchanged.
    """

    def __init__(self, spatial, temporal, decay, *, encoder_mode="frozen",
                 encoder_learning_rate=1e-5, hydro_sequence_mode="off", **kwargs):
        if (type(spatial) is not TransportGCNImputer or spatial.residual
                or spatial.jumping_knowledge or not len(spatial.convs)):
            raise ValueError("expected the baseline TransportGCNImputer without residual or JK")
        if encoder_mode not in ("frozen", "last_self", "last_self_ecology"):
            raise ValueError("encoder_mode must be frozen, last_self or last_self_ecology")
        if not np.isfinite(encoder_learning_rate) or encoder_learning_rate <= 0:
            raise ValueError("encoder_learning_rate must be finite and positive")
        if encoder_mode == "last_self_ecology" and spatial.env_encoder is None:
            raise ValueError("last_self_ecology requires the existing ecological encoder")
        if hydro_sequence_mode not in ("off", "current_only", "full_history"):
            raise ValueError("hydro_sequence_mode must be off, current_only or full_history")
        kwargs.setdefault("extra_dim", 30)
        kwargs.setdefault("interaction_indices", (0, 2, 4, 28))
        kwargs.setdefault("tail_weight", 2)
        super().__init__(temporal, decay, **kwargs)
        if spatial.head.in_features != self.hidden_size:
            raise ValueError("spatial hidden size must match the GRU")
        if any(parameter.dtype != self.dtype for parameter in spatial.parameters()):
            raise ValueError("spatial, GRU and decay parameters must share their dtype")
        self.encoder_mode = encoder_mode
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.spatial = copy.deepcopy(spatial).cpu().eval()
        self.spatial.requires_grad_(False)
        if encoder_mode != "frozen":
            self.spatial.convs[-1].self_lin.requires_grad_(True)
        if encoder_mode == "last_self_ecology":
            self.spatial.env_encoder.requires_grad_(True)
        self._initial_spatial_state = _state_copy(self.spatial)
        self._initial_last_self_state = _state_copy(self.spatial.convs[-1].self_lin)
        self._initial_ecology_state = (_state_copy(self.spatial.env_encoder)
                                       if self.spatial.env_encoder is not None else None)
        env_dim = spatial.env_encoder[0].in_features if spatial.env_encoder is not None else 0
        env_emb = spatial.env_encoder[0].out_features if env_dim else 32
        self.spatial_architecture = {
            "in_channels": spatial.convs[0].self_lin.in_features - (env_emb if env_dim else 0),
            "edge_dim": spatial.convs[0].gate_up[0].in_features,
            "hidden": self.hidden_size, "layers": len(spatial.convs), "dropout": spatial.dropout,
            "env_dim": env_dim, "env_emb": env_emb, "edge_direction": spatial.edge_direction,
            "gate_mode": spatial.gate_mode, "residual": False, "jumping_knowledge": False}
        self.encoder_trainable_parameter_count_ = sum(
            parameter.numel() for parameter in self.spatial.parameters() if parameter.requires_grad)
        self.trainable_parameter_count_ += self.encoder_trainable_parameter_count_
        self.hydro_sequence_mode = hydro_sequence_mode
        self.hydro_projection = None
        if hydro_sequence_mode != "off":
            # Constructor randomness is discarded without changing the caller's RNG.
            with torch.random.fork_rng():
                self.hydro_projection = nn.Linear(8, self.hidden_size, bias=False, dtype=self.dtype)
            nn.init.zeros_(self.hydro_projection.weight)
            self._initial_hydro_projection_state = _state_copy(self.hydro_projection)
            self.trainable_parameter_count_ += self.hydro_projection.weight.numel()

    def _prepare_inputs(self, inputs):
        required = ("raw", "env", "age", "support")
        if not isinstance(inputs, dict) or not all(key in inputs for key in required):
            raise ValueError("inputs require raw, env, age and support arrays")
        arrays = {key: torch.as_tensor(inputs[key], dtype=self.dtype).detach().cpu() for key in required}
        shape = arrays["raw"].shape
        if (len(shape) != 3 or min(shape) < 1 or shape[-1] != self.spatial_architecture["in_channels"]
                or arrays["env"].shape != (shape[0], self.spatial_architecture["env_dim"])
                or arrays["age"].shape != shape[:2] or arrays["support"].shape != (*shape[:2], 3)
                or not all(torch.isfinite(value).all() for value in arrays.values())):
            raise ValueError("raw/env/age/support must be finite aligned station/month tensors")
        if self.extra_dim:
            if "extra" not in inputs:
                raise ValueError("extra inputs are required when extra_dim is positive")
            extra = torch.as_tensor(inputs["extra"], dtype=self.dtype).detach().cpu()
            if extra.shape != (*shape[:2], self.extra_dim) or not torch.isfinite(extra).all():
                raise ValueError("extra inputs must be finite [station, month, extra_dim]")
            arrays["extra"] = extra
        if self.hydro_projection is not None:
            if "daily_history" not in inputs:
                raise ValueError("daily_history is required for an active hydro_sequence_mode")
            history = torch.as_tensor(inputs["daily_history"], dtype=self.dtype).detach().cpu()
            if history.shape != (*shape[:2], 8) or not torch.isfinite(history).all():
                raise ValueError("daily_history must be finite aligned [station, month, 8]")
            arrays["daily_history"] = history
        return arrays

    def _hidden_cells(self, inputs, cells):
        """Gather causal windows, encode valid dates, then apply the existing GRU-D."""
        cells = torch.as_tensor(cells, dtype=torch.long)
        n, t = inputs["age"].shape
        if cells.ndim != 1 or not len(cells) or (cells < 0).any() or (cells >= n * t).any():
            raise ValueError("cells must be a nonempty valid flat station/month vector")
        station_ids, month_ids = cells // t, cells % t
        raw_months = month_ids[:, None] - torch.arange(self.lookback - 1, -1, -1)[None, :]
        valid = raw_months >= 0
        months = raw_months.clamp_min(0)
        stations = station_ids[:, None].expand_as(months)
        # Encode no padding or future month. Empty edges make the flattened
        # batch exactly a collection of independent spatial self paths.
        self.spatial.eval()
        encoded = self.spatial.encode_nodes(
            inputs["raw"][stations[valid], months[valid]],
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, self.spatial_architecture["edge_dim"]), dtype=self.dtype),
            inputs["env"][stations[valid]])
        valid_rows = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
        sequence = encoded.new_zeros((len(cells) * self.lookback, self.hidden_size))
        sequence = sequence.index_copy(0, valid_rows, encoded).reshape(len(cells), self.lookback, -1)
        if self.hydro_projection is not None:
            use_history = valid.clone()
            if self.hydro_sequence_mode == "current_only":
                use_history[:, :-1] = False
            projected = self.hydro_projection(
                inputs["daily_history"][stations[use_history], months[use_history]])
            history_rows = torch.nonzero(use_history.reshape(-1), as_tuple=False).reshape(-1)
            hydro_sequence = sequence.new_zeros((len(cells) * self.lookback, self.hidden_size))
            hydro_sequence = hydro_sequence.index_copy(0, history_rows, projected).reshape_as(sequence)
            sequence = sequence + hydro_sequence
        decay_input = torch.cat([inputs["age"][stations, months, None],
                                 inputs["support"][stations, months]], dim=-1)
        gamma = torch.exp(-torch.relu(self.decay(decay_input)))
        state = encoded.new_zeros((len(cells), self.hidden_size))
        for step in range(self.lookback):
            indicator = valid[:, step, None]
            step_input = torch.cat([sequence[:, step] * indicator, indicator.to(self.dtype)], dim=-1)
            updated = self.temporal(step_input, gamma[:, step] * state)
            state = torch.where(indicator, updated, state)
        return state

    def _delta_cells(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        hidden = self._hidden_cells(inputs, cells)
        if self.extra_dim:
            months = inputs["age"].shape[1]
            extra = inputs["extra"][cells // months, cells % months]
            pieces = [hidden, extra]
            if self.interaction_indices:
                interaction = hidden.unsqueeze(-1) * extra[:, self.interaction_indices].unsqueeze(1)
                pieces.append(interaction.flatten(1))
            hidden = torch.cat(pieces, dim=-1)
        delta = self.head(hidden).squeeze(-1).double()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite encoder native residual")
        return delta

    def _encoder_distances(self):
        return {
            "spatial_parameter_distance": _parameter_distance(self.spatial, self._initial_spatial_state),
            "last_self_parameter_distance": _parameter_distance(
                self.spatial.convs[-1].self_lin, self._initial_last_self_state),
            "ecology_parameter_distance": (_parameter_distance(self.spatial.env_encoder,
                                                               self._initial_ecology_state)
                                           if self.spatial.env_encoder is not None else 0.0)}

    def _hydro_distances(self):
        if self.hydro_projection is None:
            return {}
        return {"hydro_projection_parameter_distance": _parameter_distance(
                    self.hydro_projection, self._initial_hydro_projection_state),
                "hydro_projection_parameter_norm": float(
                    self.hydro_projection.weight.detach().double().norm())}

    def _source_weights(self, source_cells, source_y, source_tail, months):
        """Full-source weights; subclasses may change the training estimand."""
        return 1.0 + (self.tail_weight - 1.0) * source_tail.double()

    def _training_errors(self, prediction, truth, tail, weights):
        """Unnormalized cell losses; the fit loop retains one source denominator."""
        return weights * (prediction - truth).abs()

    def fit(self, source_inputs, source_base_native, source_truth, source_mask,
            val_inputs, val_base_native, val_truth, val_query_mask, *, tail_threshold,
            selection_role, progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        if not np.isfinite(tail_threshold) or tail_threshold < 0:
            raise ValueError("tail_threshold must be finite and nonnegative")
        source, validation = self._prepare_inputs(source_inputs), self._prepare_inputs(val_inputs)
        source_cells, source_base, source_y = self._observed_labels(
            source, source_base_native, source_truth, source_mask)
        val_cells, val_base, val_y = self._observed_labels(
            validation, val_base_native, val_truth, val_query_mask)
        self.tail_threshold_ = float(tail_threshold)
        source_tail = source_y >= self.tail_threshold_
        weights = self._source_weights(source_cells, source_y, source_tail, source["age"].shape[1])
        mean_weight = float(weights.mean())
        self.n_source_cells_, self.n_validation_query_ = len(source_cells), len(val_cells)
        self.n_source_tail_cells_ = int(source_tail.sum())
        self.n_source_stations_ = len(np.unique(source_cells // source["age"].shape[1]))
        self.n_validation_stations_ = len(np.unique(val_cells // validation["age"].shape[1]))
        self.temporal.load_state_dict(self._initial_temporal_state)
        self.decay.load_state_dict(self._initial_decay_state)
        self.spatial.load_state_dict(self._initial_spatial_state)
        if self.hydro_projection is not None:
            self.hydro_projection.load_state_dict(self._initial_hydro_projection_state)
        self.spatial.eval()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        backbone = [parameter for module in (self.temporal, self.decay)
                    for parameter in module.parameters() if parameter.requires_grad]
        if self.hydro_projection is not None:
            backbone += list(self.hydro_projection.parameters())
        encoder = [parameter for parameter in self.spatial.parameters() if parameter.requires_grad]
        parameters = backbone + list(self.head.parameters()) + encoder
        groups = [{"params": backbone, "lr": self.learning_rate}] if backbone else []
        groups.append({"params": self.head.parameters(), "lr": self.head_learning_rate})
        if encoder:
            groups.append({"params": encoder, "lr": self.encoder_learning_rate})
        optimizer = torch.optim.Adam(groups)
        self.trace_, self.best_epoch_, self.selected_scale_ = [], 0, 0.0
        self.optimizer_steps_ = 0
        best, stale, best_state = float("inf"), 0, None
        modules = (self.temporal, self.decay, self.head, self.spatial)
        if self.hydro_projection is not None:
            modules += (self.hydro_projection,)

        def evaluate(epoch, train_loss):
            nonlocal best, stale, best_state
            with torch.no_grad():
                delta = torch.cat([
                    self._delta_cells(validation, val_cells[start:start + self.batch_size])
                    for start in range(0, len(val_cells), self.batch_size)])
                scores = [{"scale": scale, "mae": float(
                    (self._combine(val_base, delta, scale) - val_y).abs().mean())}
                    for scale in self.scales]
            chosen = min(scores, key=lambda row: (row["mae"], row["scale"]))
            if not np.isfinite(chosen["mae"]):
                raise FloatingPointError("nonfinite encoder native validation MAE")
            improved = chosen["mae"] < best
            if improved:
                best, stale = chosen["mae"], 0
                self.best_epoch_, self.selected_scale_ = epoch, chosen["scale"]
                best_state = [_state_copy(module) for module in modules]
            else:
                stale += 1
            row = {"epoch": epoch, "training_loss": train_loss,
                   "validation_mae": chosen["mae"], "validation_scale": chosen["scale"],
                   "validation_candidates": scores, "is_best": improved, "stale_epochs": stale,
                   "best_epoch": self.best_epoch_, "best_scale": self.selected_scale_,
                   "best_validation_mae": best,
                   "temporal_parameter_distance": _parameter_distance(self.temporal, self._initial_temporal_state),
                   "decay_parameter_distance": _parameter_distance(self.decay, self._initial_decay_state),
                   "head_parameter_norm": float(torch.sqrt(sum(
                       parameter.detach().double().square().sum() for parameter in self.head.parameters()))),
                   **self._encoder_distances(), **self._hydro_distances()}
            self.trace_.append(row)
            if improved:
                self.validation_metrics_ = copy.deepcopy(row)
            if progress is not None:
                progress(copy.deepcopy(row))

        evaluate(0, None)
        for epoch in range(1, self.epochs + 1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
            order = rng.permutation(len(source_cells))
            weighted_error = 0.0
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                optimizer.zero_grad()
                delta = self._delta_cells(source, source_cells[rows])
                prediction = self._combine(source_base[rows], delta, 1.0)
                errors = self._training_errors(prediction, source_y[rows], source_tail[rows], weights[rows])
                loss = errors.mean() / mean_weight
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite encoder native training loss")
                loss.backward()
                if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                    raise FloatingPointError("nonfinite encoder native residual gradient")
                nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                self.optimizer_steps_ += 1
                if not all(torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError("nonfinite encoder native residual parameter")
                weighted_error += float(errors.detach().sum())
            evaluate(epoch, weighted_error / float(weights.sum()))
            if stale >= self.patience:
                break
        self.epochs_run_ = len(self.trace_) - 1
        for module, state in zip(modules, best_state):
            module.load_state_dict(state)
        self.spatial.eval()
        return self

    def _config(self):
        config = {**super()._config(), "encoder_mode": self.encoder_mode,
                  "encoder_learning_rate": self.encoder_learning_rate,
                  "extra_dim": self.extra_dim, "interaction_indices": list(self.interaction_indices)}
        if self.hydro_projection is not None:
            config["hydro_sequence_mode"] = self.hydro_sequence_mode
        return config

    def to_dict(self):
        summary = super().to_dict()
        summary.update({"model_class": "EncoderNativeResidual", "spatial_architecture": self.spatial_architecture.copy(),
                        "encoder_trainable_parameter_count": self.encoder_trainable_parameter_count_,
                        **self._encoder_distances()})
        summary["protocol"].update({
            "initialization": "provided original spatial/GRU/decay; zero scalar head",
            "spatial_mode": "eval throughout; empty edges; frozen input normalization",
            "encoder_selection": self.encoder_mode})
        if self.hydro_projection is not None:
            summary.update({"hydro_projection_trainable_parameter_count": self.hydro_projection.weight.numel(),
                            **self._hydro_distances()})
            summary["protocol"].update({
                "hydro_sequence_mode": self.hydro_sequence_mode,
                "hydro_sequence_injection": "encoded state plus bias-free Linear(8,H) before existing GRU",
                "hydro_sequence_initialization": "zero projection; original encoded state at initialization",
                "hydro_sequence_learning_rate": self.learning_rate,
                "hydro_sequence_scope": "valid causal window steps only; no padding or future dates"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload.update({"model_class": "EncoderNativeResidual", "spatial_architecture": self.spatial_architecture.copy(),
                        "spatial": _state_copy(self.spatial),
                        "initial_spatial": copy.deepcopy(self._initial_spatial_state)})
        if self.hydro_projection is not None:
            payload.update({"hydro_projection": _state_copy(self.hydro_projection),
                            "initial_hydro_projection": copy.deepcopy(self._initial_hydro_projection_state)})
        return payload

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1 or payload.get("model_class") != "EncoderNativeResidual":
            raise ValueError("unsupported encoder native residual checkpoint")
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(payload["dtype"])
        if dtype is None:
            raise ValueError("unsupported encoder native residual dtype")
        h = int(payload["hidden_size"])
        with torch.random.fork_rng():
            spatial = TransportGCNImputer(**payload["spatial_architecture"]).to(dtype=dtype)
            temporal = nn.GRUCell(h + 1, h, bias=payload["temporal_bias"]).to(dtype=dtype)
            decay = nn.Linear(4, h, bias=payload["decay_bias"]).to(dtype=dtype)
        for module, name in ((spatial, "spatial"), (temporal, "temporal"), (decay, "decay")):
            module.load_state_dict(payload["initial_" + name])
            if not all(torch.isfinite(parameter).all() for parameter in module.parameters()):
                raise ValueError("checkpoint contains nonfinite initial weights")
        obj = cls(spatial, temporal, decay, **payload["config"])
        names = ("spatial", "temporal", "decay", "head")
        if obj.hydro_projection is not None:
            if not all(key in payload for key in ("hydro_projection", "initial_hydro_projection")):
                raise ValueError("active hydro sequence checkpoint lacks projection weights")
            obj.hydro_projection.load_state_dict(payload["initial_hydro_projection"])
            if not torch.isfinite(obj.hydro_projection.weight).all() or torch.count_nonzero(obj.hydro_projection.weight):
                raise ValueError("hydro projection initialization must contain finite zero weights")
            obj._initial_hydro_projection_state = _state_copy(obj.hydro_projection)
            names += ("hydro_projection",)
        elif "hydro_projection" in payload or "initial_hydro_projection" in payload:
            raise ValueError("off hydro sequence checkpoint cannot contain projection weights")
        for name in names:
            module = getattr(obj, name)
            module.load_state_dict(payload[name])
            if not all(torch.isfinite(parameter).all() for parameter in module.parameters()):
                raise ValueError("checkpoint contains nonfinite residual weights")
        summary = payload["summary"]
        for name in ("best_epoch", "epochs_run", "optimizer_steps", "n_source_cells", "n_source_tail_cells",
                     "n_validation_query", "n_source_stations", "n_validation_stations"):
            setattr(obj, name + "_", int(summary[name]))
        obj.selected_scale_, obj.tail_threshold_ = float(summary["selected_scale"]), float(summary["tail_threshold"])
        if obj.selected_scale_ not in obj.scales or not np.isfinite(obj.tail_threshold_) or obj.tail_threshold_ < 0:
            raise ValueError("invalid serialized scale or tail threshold")
        obj.trace_, obj.validation_metrics_ = copy.deepcopy(summary["trace"]), copy.deepcopy(summary["validation_metrics"])
        obj.spatial.eval()
        return obj
