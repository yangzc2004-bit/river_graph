"""Support-episodic tuning of a copied observation-aware temporal encoder.

Spatial encodings and the two-dimensional readout remain frozen. Only the
existing GRUCell and observation decay are updated. Their rolling recurrent
states are causal; station normalization uses fixed calendar anchors over the
record and therefore represents retrospective reconstruction, not forecasting.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.episodic_station_adapter import (
    TRAIN_K,
    TRAIN_RIDGE,
    EpisodicStationProjector,
    _episode_queries,
    _integer,
    _tensor_episode_loss,
    stratified_support_months,
)


def gathered_rolling_states(temporal, decay, inputs, station_ids, month_ids, *, lookback):
    """Return differentiable states for the requested station/month pairs.

    Inputs are detached tensors in station-major layout. Each requested month
    starts with a fresh zero state and processes its own rolling window. Left
    padding performs no recurrent update, matching the released M1 model.
    """
    encoded, age, support = (inputs[key] for key in ("encoded", "age", "support"))
    station_ids = torch.as_tensor(station_ids, dtype=torch.long, device=encoded.device)
    month_ids = torch.as_tensor(month_ids, dtype=torch.long, device=encoded.device)
    if (station_ids.ndim != 1 or month_ids.shape != station_ids.shape
            or (station_ids < 0).any() or (station_ids >= len(encoded)).any()
            or (month_ids < 0).any() or (month_ids >= encoded.shape[1]).any()):
        raise ValueError("requested station/month pairs must be valid aligned vectors")
    offsets = torch.arange(lookback - 1, -1, -1, device=encoded.device)
    raw_months = month_ids[:, None] - offsets[None, :]
    valid = raw_months >= 0
    months = raw_months.clamp_min(0)
    stations = station_ids[:, None]
    sequence = encoded[stations, months]
    decay_input = torch.cat([age[stations, months, None], support[stations, months]], dim=-1)
    gamma = torch.exp(-torch.relu(decay(decay_input)))
    state = encoded.new_zeros((len(station_ids), encoded.shape[-1]))
    for step in range(lookback):
        indicator = valid[:, step, None]
        step_input = torch.cat([
            sequence[:, step] * indicator, indicator.to(encoded.dtype)], dim=-1)
        updated = temporal(step_input, gamma[:, step] * state)
        state = torch.where(indicator, updated, state)
    return state


def anchor_normalize(raw_basis, anchor_months, *, scale_floor, return_stats=False):
    """Center by anchors and divide by one station scalar RMS across both axes.

    This removes a common positive gain of the two-dimensional representation
    above the numerical floor. It does not separately whiten its two axes.
    """
    anchors = raw_basis[:, anchor_months]
    mean = anchors.mean(dim=1, keepdim=True)
    squared = (anchors - mean).square().mean(dim=(1, 2), keepdim=True)
    # Clamp the variance before sqrt to avoid a NaN derivative at zero RMS.
    scale = squared.clamp_min(scale_floor**2).sqrt()
    result = (raw_basis - mean) / scale
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite anchor-normalized temporal basis")
    if return_stats:
        rms = squared.detach().sqrt().reshape(-1).cpu().numpy()
        return result, {"station_rms": rms, "floor_hit": rms < scale_floor}
    return result


def _state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _parameter_distance(module, initial):
    return float(torch.sqrt(sum(
        (value.detach().cpu().double() - initial[key].double()).square().sum()
        for key, value in module.state_dict().items())))


class EpisodicTemporalAdapter:
    """Fine-tune copied GRU/decay weights through support-fitted ridge heads.

    Training uses equal-station native MAE, averaged over K=3/5 and ridge=1/10
    with support-mean alpha=1. Checkpoint selection pools source-validation
    query cells using the same four losses. Epoch zero is a candidate.

    Each input mapping has ``encoded[S,T,H]``, ``age[S,T]`` and
    ``support[S,T,3]``. Source labels/base are consulted only where masks are
    true. No labels enter ``transform`` or the calendar anchor selection.
    """

    def __init__(self, temporal, decay, readout, lookback=12, epochs=30, patience=5,
                 learning_rate=1e-4, batch_size=8, seed=42, anchor_count=32,
                 scale_floor=1e-4):
        if (not isinstance(temporal, nn.GRUCell) or not isinstance(decay, nn.Linear)
                or temporal.input_size != temporal.hidden_size + 1
                or decay.in_features != 4 or decay.out_features != temporal.hidden_size):
            raise ValueError("expected the existing GRUCell(H+1,H) and Linear(4,H) decay")
        self.hidden_size = temporal.hidden_size
        self.lookback = _integer(lookback, "lookback")
        self.epochs = _integer(epochs, "epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed", minimum=0)
        self.anchor_count = _integer(anchor_count, "anchor_count", minimum=2)
        for name, value in (("learning_rate", learning_rate), ("scale_floor", scale_floor)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        self.temporal = copy.deepcopy(temporal).cpu()
        self.decay = copy.deepcopy(decay).cpu()
        self.dtype = next(self.temporal.parameters()).dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("GRU and decay must use float32 or float64")
        if next(self.decay.parameters()).dtype != self.dtype:
            raise ValueError("GRU and decay must use the same dtype")
        for module in (self.temporal, self.decay):
            module.requires_grad_(True)
        self.initial_temporal = copy.deepcopy(self.temporal).requires_grad_(False)
        self.initial_decay = copy.deepcopy(self.decay).requires_grad_(False)
        self._initial_temporal_state = _state_copy(self.temporal)
        self._initial_decay_state = _state_copy(self.decay)
        self.readout = torch.as_tensor(readout, dtype=torch.float64).detach().cpu().clone()
        if self.readout.shape != (self.hidden_size, 2) or not torch.isfinite(self.readout).all():
            raise ValueError("fixed readout must be finite [hidden, 2]")
        self.trainable_parameter_count_ = sum(
            parameter.numel() for module in (self.temporal, self.decay)
            for parameter in module.parameters())

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, dict) or not all(
                key in inputs for key in ("encoded", "age", "support")):
            raise ValueError("inputs require encoded, age and support arrays")
        result = {key: torch.as_tensor(inputs[key], dtype=self.dtype).detach().cpu()
                  for key in ("encoded", "age", "support")}
        shape = result["encoded"].shape
        if (len(shape) != 3 or min(shape) < 1 or shape[-1] != self.hidden_size
                or result["age"].shape != shape[:2] or result["support"].shape != (*shape[:2], 3)
                or not all(torch.isfinite(value).all() for value in result.values())):
            raise ValueError("inputs must be finite aligned station/month tensors")
        return result

    def anchor_months(self, n_months):
        """Fixed label-free calendar anchors, including the record endpoints."""
        n_months = _integer(n_months, "n_months")
        return np.linspace(0, n_months - 1, min(n_months, self.anchor_count), dtype=np.int64)

    def _basis_for_rows(self, inputs, station_ids, *, selected_months=None, initial=False,
                        return_stats=False):
        """Compute only selected/anchor windows, scattering to a small 2D grid."""
        station_ids = torch.as_tensor(station_ids, dtype=torch.long)
        batch = {key: value[station_ids] for key, value in inputs.items()}
        n, t = batch["age"].shape
        anchors = torch.as_tensor(self.anchor_months(t))
        if selected_months is None:
            selected = torch.ones((n, t), dtype=torch.bool)
        else:
            selected = torch.as_tensor(selected_months, dtype=torch.bool).clone()
            if selected.shape != (n, t):
                raise ValueError("selected months must align with selected station rows")
        selected[:, anchors] = True
        rows, months = torch.nonzero(selected, as_tuple=True)
        temporal = self.initial_temporal if initial else self.temporal
        decay = self.initial_decay if initial else self.decay
        hidden = gathered_rolling_states(temporal, decay, batch, rows, months,
                                        lookback=self.lookback)
        projected = hidden.double() @ self.readout
        raw_basis = projected.new_zeros((n * t, 2)).index_copy(
            0, rows * t + months, projected).reshape(n, t, 2)
        return anchor_normalize(raw_basis, anchors, scale_floor=self.scale_floor,
                                return_stats=return_stats)

    def fit(self, source_inputs, source_base_z, source_truth, source_mask,
            validation_inputs, validation_base_native, validation_truth,
            validation_mask, *, selection_role, progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        source = self._prepare_inputs(source_inputs)
        validation = self._prepare_inputs(validation_inputs)
        source_y, source_z, source_observed = EpisodicStationProjector._label_arrays(
            source["encoded"], source_base_z, source_truth, source_mask, base_native=False)
        val_y, val_z, val_observed = EpisodicStationProjector._label_arrays(
            validation["encoded"], validation_base_native, validation_truth,
            validation_mask, base_native=True)
        self.source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) > 5)
        self.excluded_source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) <= 5)
        if not len(self.source_station_ids_):
            raise ValueError("no source station has five supports and at least one query")
        source = {key: value[self.source_station_ids_] for key, value in source.items()}
        source_y = torch.as_tensor(source_y[self.source_station_ids_])
        source_z = torch.as_tensor(source_z[self.source_station_ids_])
        source_observed = source_observed[self.source_station_ids_]
        val_y, val_z = torch.as_tensor(val_y), torch.as_tensor(val_z)
        schedule, query = support_schedule(np.flatnonzero(val_observed), val_observed.shape[1])
        if not len(query) or len(schedule) != len(val_observed):
            raise ValueError("every validation station needs five supports and at least one query")
        val_support = torch.as_tensor(np.stack([
            schedule[i] % val_observed.shape[1] for i in range(len(val_observed))]))
        val_query = torch.zeros_like(val_y, dtype=torch.bool)
        val_query.view(-1)[torch.as_tensor(query)] = True
        self.n_source_stations_ = len(source_y)
        self.n_validation_stations_, self.n_validation_query_ = len(val_y), len(query)
        self.n_source_months_, self.n_validation_months_ = source_y.shape[1], val_y.shape[1]
        self.temporal.load_state_dict(self._initial_temporal_state)
        self.decay.load_state_dict(self._initial_decay_state)
        parameters = list(self.temporal.parameters()) + list(self.decay.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.trace_, self.best_epoch_ = [], 0
        best_score, stale = float("inf"), 0
        best_temporal, best_decay = None, None

        def evaluate(epoch, training_mae, training_rms=None):
            nonlocal best_score, stale, best_temporal, best_decay
            total = np.zeros(len(TRAIN_K) * len(TRAIN_RIDGE), dtype=float)
            validation_rms = []
            with torch.no_grad():
                for start in range(0, len(val_y), self.batch_size):
                    rows = np.arange(start, min(start + self.batch_size, len(val_y)))
                    basis, stats = self._basis_for_rows(
                        validation, rows, selected_months=val_observed[rows], return_stats=True)
                    validation_rms.extend(stats["station_rms"].tolist())
                    _, losses = _tensor_episode_loss(
                        basis, val_z[rows], val_y[rows], val_support[rows], val_query[rows], pooled=True)
                    total += losses.numpy() * int(val_query[rows].sum())
            components = total / self.n_validation_query_
            score = float(components.mean())
            improved = score < best_score
            if improved:
                best_score, stale, self.best_epoch_ = score, 0, epoch
                best_temporal, best_decay = _state_copy(self.temporal), _state_copy(self.decay)
            else:
                stale += 1
            row = {"epoch": epoch, "training_mae": training_mae, "validation_mae": score,
                   "validation_by_k_ridge": [
                       {"k": k, "ridge_strength": strength, "mae": float(value)}
                       for (k, strength), value in zip(
                           [(k, r) for k in TRAIN_K for r in TRAIN_RIDGE], components)],
                   "is_best": improved, "stale_epochs": stale,
                   "temporal_parameter_distance": _parameter_distance(
                       self.temporal, self._initial_temporal_state),
                   "decay_parameter_distance": _parameter_distance(self.decay, self._initial_decay_state)}
            for name, rms in (("source", training_rms), ("validation", validation_rms)):
                row[name + "_anchor_floor_hits"] = (
                    int((np.asarray(rms) < self.scale_floor).sum()) if rms is not None else None)
                row[name + "_anchor_rms_min"] = float(np.min(rms)) if rms is not None else None
                row[name + "_anchor_rms_max"] = float(np.max(rms)) if rms is not None else None
            self.trace_.append(row)
            if improved:
                self.validation_metrics_ = copy.deepcopy(row)
            if progress is not None:
                progress(copy.deepcopy(row))

        evaluate(0, None)
        for epoch in range(1, self.epochs + 1):
            support = stratified_support_months(source_observed, seed=self.seed, epoch=epoch)
            query_mask = torch.as_tensor(_episode_queries(source_observed, support))
            support = torch.as_tensor(support)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 2**31 - 1]))
            order = rng.permutation(len(source_y))
            total = 0.0
            training_rms = []
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                optimizer.zero_grad()
                basis, stats = self._basis_for_rows(
                    source, rows, selected_months=source_observed[rows], return_stats=True)
                training_rms.extend(stats["station_rms"].tolist())
                loss, _ = _tensor_episode_loss(
                    basis, source_z[rows], source_y[rows], support[rows], query_mask[rows], pooled=False)
                loss.backward()
                if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                    raise FloatingPointError("nonfinite episodic temporal gradient")
                nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError("nonfinite episodic temporal parameter")
                total += float(loss.detach()) * len(rows)
            evaluate(epoch, total / len(source_y), training_rms)
            if stale >= self.patience:
                break
        self.epochs_run_ = len(self.trace_) - 1
        self.temporal.load_state_dict(best_temporal)
        self.decay.load_state_dict(best_decay)
        return self

    def _transform(self, inputs, *, initial):
        prepared = self._prepare_inputs(inputs)
        result = np.empty((*prepared["age"].shape, 2), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(result), self.batch_size):
                rows = np.arange(start, min(start + self.batch_size, len(result)))
                result[rows] = self._basis_for_rows(prepared, rows, initial=initial).numpy()
        return result

    def transform(self, inputs):
        """Return the selected temporal basis for complete station timelines."""
        return self._transform(inputs, initial=False)

    def transform_initial(self, inputs):
        """Matched frozen-weight control using the identical anchor normalization."""
        return self._transform(inputs, initial=True)

    def normalization_stats(self, inputs, initial=False):
        """Label-free per-station anchor RMS, evaluating anchor windows only."""
        prepared = self._prepare_inputs(inputs)
        rms = np.empty(len(prepared["age"]), dtype=float)
        with torch.no_grad():
            for start in range(0, len(rms), self.batch_size):
                rows = np.arange(start, min(start + self.batch_size, len(rms)))
                _, stats = self._basis_for_rows(
                    prepared, rows, selected_months=np.zeros((len(rows), prepared["age"].shape[1]),
                                                            dtype=bool),
                    initial=initial, return_stats=True)
                rms[rows] = stats["station_rms"]
        hit = rms < self.scale_floor
        return {"anchor_months": self.anchor_months(prepared["age"].shape[1]).tolist(),
                "station_rms": rms.tolist(), "floor_hit": hit.tolist(),
                "station_count": len(rms), "floor_hits": int(hit.sum()),
                "scale_floor": self.scale_floor, "initial": bool(initial)}

    def _config(self):
        return {name: getattr(self, name) for name in (
            "lookback", "epochs", "patience", "learning_rate", "batch_size", "seed",
            "anchor_count", "scale_floor")}

    def to_dict(self):
        """JSON summary; use to_payload for the reconstructable torch checkpoint."""
        if not hasattr(self, "epochs_run_"):
            raise RuntimeError("fit the episodic temporal adapter before serialization")
        return {"version": 1, "config": self._config(), "hidden_size": self.hidden_size,
                "dtype": str(self.dtype), "trainable_parameter_count": self.trainable_parameter_count_,
                "fixed_readout_shape": list(self.readout.shape),
                "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_,
                "trace": copy.deepcopy(self.trace_),
                "validation_metrics": copy.deepcopy(self.validation_metrics_),
                "temporal_parameter_distance": _parameter_distance(
                    self.temporal, self._initial_temporal_state),
                "decay_parameter_distance": _parameter_distance(self.decay, self._initial_decay_state),
                "selected_validation_anchor_floor_hits": self.validation_metrics_["validation_anchor_floor_hits"],
                "selected_source_anchor_floor_hits": self.validation_metrics_["source_anchor_floor_hits"],
                "source_station_ids": self.source_station_ids_.tolist(),
                "excluded_source_station_ids": self.excluded_source_station_ids_.tolist(),
                "n_source_stations": self.n_source_stations_,
                "n_validation_stations": self.n_validation_stations_,
                "n_validation_query": self.n_validation_query_,
                "n_source_months": self.n_source_months_,
                "n_validation_months": self.n_validation_months_,
                "source_anchor_months": self.anchor_months(self.n_source_months_).tolist(),
                "validation_anchor_months": self.anchor_months(self.n_validation_months_).tolist(),
                "protocol": {"alpha": 1.0, "k_values": list(TRAIN_K),
                             "ridge_strengths": list(TRAIN_RIDGE), "optimizer": "Adam",
                             "gradient_clip_norm": 1.0,
                             "training_loss": "raw_mae_equal_station_mean_k_ridge",
                             "validation_loss": "raw_mae_pooled_query_mean_k_ridge",
                             "source_support": "stratified_observed_rank_five_bins",
                             "validation_support": "spatial_fewshot.support_schedule",
                             "normalization": "fixed_calendar_anchor_mean_and_station_scalar_rms",
                             "source_anchor_statistics": "training batches before their optimizer steps; epoch0 not computed",
                             "gradient_finite_policy": "fail on nonfinite; no skipped or clipped prediction values",
                             "selection_role": "source_validation"}}

    def to_payload(self):
        return {"version": 1, "config": self._config(), "hidden_size": self.hidden_size,
                "dtype": str(self.dtype), "temporal_bias": self.temporal.bias,
                "decay_bias": self.decay.bias is not None,
                "readout": self.readout.clone(), "temporal": _state_copy(self.temporal),
                "decay": _state_copy(self.decay),
                "initial_temporal": _state_copy(self.initial_temporal),
                "initial_decay": _state_copy(self.initial_decay), "summary": self.to_dict()}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1:
            raise ValueError("unsupported episodic temporal checkpoint version")
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(payload["dtype"])
        if dtype is None:
            raise ValueError("unsupported episodic temporal dtype")
        h = _integer(payload["hidden_size"], "hidden_size")
        temporal = nn.GRUCell(h + 1, h, bias=payload["temporal_bias"]).to(dtype=dtype)
        decay = nn.Linear(4, h, bias=payload["decay_bias"]).to(dtype=dtype)
        temporal.load_state_dict(payload["initial_temporal"])
        decay.load_state_dict(payload["initial_decay"])
        obj = cls(temporal, decay, payload["readout"], **payload["config"])
        obj.temporal.load_state_dict(payload["temporal"])
        obj.decay.load_state_dict(payload["decay"])
        if not all(torch.isfinite(parameter).all() for module in (
                obj.temporal, obj.decay, obj.initial_temporal, obj.initial_decay)
                   for parameter in module.parameters()):
            raise ValueError("checkpoint contains nonfinite temporal weights")
        summary = payload["summary"]
        for name in ("best_epoch", "epochs_run", "n_source_stations", "n_validation_stations",
                     "n_validation_query", "n_source_months", "n_validation_months"):
            setattr(obj, name + "_", int(summary[name]))
        obj.trace_, obj.validation_metrics_ = copy.deepcopy(summary["trace"]), copy.deepcopy(
            summary["validation_metrics"])
        obj.source_station_ids_ = np.asarray(summary["source_station_ids"], dtype=np.int64)
        obj.excluded_source_station_ids_ = np.asarray(summary["excluded_source_station_ids"], dtype=np.int64)
        return obj
