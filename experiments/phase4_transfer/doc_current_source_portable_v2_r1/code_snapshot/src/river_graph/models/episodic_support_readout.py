"""Train only the two-dimensional support readout of frozen temporal states."""

from __future__ import annotations

import copy

import numpy as np
import torch

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
from river_graph.models.episodic_temporal_adapter import anchor_normalize

PROJECTION_BATCH_SIZE = 2048


class EpisodicSupportReadout:
    """Supervise a readout without updating any hidden-state expert.

    The readout is P=P0+||P0||F*D, with D initially zero. For hidden width 64,
    exactly 128 scalar parameters are optimized. No PCA, whitening, QR or new
    recurrent weights are fitted. A station scalar RMS across fixed calendar
    anchors controls the projected basis scale, exactly as in the older v4
    temporal adapter. That feature-only normalization is retrospective.

    Training and validation inputs may have different station counts. Both
    baselines are native DOC, including any frozen neural correction. An OOF
    forest component does not make a source-trained neural correction OOF;
    source episode losses are supervised training diagnostics.
    """

    def __init__(self, readout, anchor_count=32, scale_floor=1e-4, seed=42,
                 max_epochs=30, patience=5, lr=1e-3, batch_size=32):
        initial = torch.as_tensor(readout, dtype=torch.float64).detach().cpu().clone()
        if (initial.ndim != 2 or initial.shape[1] != 2 or initial.shape[0] < 2
                or not torch.isfinite(initial).all() or not torch.isfinite(initial.norm())
                or not float(initial.norm()) > 0):
            raise ValueError("readout must be finite nonzero [hidden, 2]")
        self.initial_readout = initial
        self.hidden_size = initial.shape[0]
        self.initial_norm = float(initial.norm())
        self.anchor_count = _integer(anchor_count, "anchor_count", minimum=2)
        self.seed = _integer(seed, "seed", minimum=0)
        self.max_epochs = _integer(max_epochs, "max_epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        for name, value in (("scale_floor", scale_floor), ("lr", lr)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        self.trainable_parameter_count_ = initial.numel()

    def anchor_months(self, n_months):
        n_months = _integer(n_months, "n_months")
        return np.linspace(0, n_months-1, min(n_months, self.anchor_count), dtype=np.int64)

    def _hidden(self, hidden):
        values = torch.as_tensor(hidden, dtype=torch.float64).detach().cpu()
        if (values.ndim != 3 or min(values.shape) < 1 or values.shape[-1] != self.hidden_size
                or not torch.isfinite(values).all()):
            raise ValueError("hidden must be finite [station, month, readout hidden width]")
        return values

    def _basis(self, hidden, readout, *, return_stats=False):
        n, months, _ = hidden.shape
        flat = hidden.reshape(-1, self.hidden_size)
        # Match the refresh helper's station-major projection batching exactly.
        projected = torch.cat([flat[start:start+PROJECTION_BATCH_SIZE] @ readout
                               for start in range(0, len(flat), PROJECTION_BATCH_SIZE)])
        return anchor_normalize(projected.reshape(n, months, 2),
                                torch.as_tensor(self.anchor_months(months)),
                                scale_floor=self.scale_floor, return_stats=return_stats)

    def fit(self, source_hidden, source_base_native, source_truth, source_mask,
            validation_hidden, validation_base_native, validation_truth, validation_mask,
            *, selection_role="source_validation", progress=None):
        """Learn from source episodes; select checkpoints on held validation.

        Source stations receive equal weight. Validation pools fixed query
        cells. Both losses average K={3,5} and ridge={1,10}, with alpha=1.
        Five support candidates are excluded from queries even when K=3.
        Values outside each supplied mask never enter a label transformation,
        reduction, episode or checkpoint choice and may contain NaNs.
        """
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        source_x, validation_x = self._hidden(source_hidden), self._hidden(validation_hidden)
        if source_x.shape[1] != validation_x.shape[1]:
            raise ValueError("source and validation month grids must match")
        source_y, source_z, source_observed = EpisodicStationProjector._label_arrays(
            source_x, source_base_native, source_truth, source_mask, base_native=True)
        val_y, val_z, val_observed = EpisodicStationProjector._label_arrays(
            validation_x, validation_base_native, validation_truth, validation_mask, base_native=True)
        self.source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) > 5)
        self.excluded_source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) <= 5)
        if not len(self.source_station_ids_):
            raise ValueError("no source station has five supports and at least one query")
        if not (val_observed.sum(axis=1) > 5).all():
            raise ValueError("every validation station needs five supports and at least one query")
        source_x = source_x[self.source_station_ids_]
        source_y = torch.as_tensor(source_y[self.source_station_ids_])
        source_z = torch.as_tensor(source_z[self.source_station_ids_])
        source_observed = source_observed[self.source_station_ids_]
        val_y, val_z = torch.as_tensor(val_y), torch.as_tensor(val_z)
        schedule, query = support_schedule(np.flatnonzero(val_observed), val_observed.shape[1])
        support = np.stack([schedule[i] % val_observed.shape[1] for i in range(len(val_observed))])
        validation_support = torch.as_tensor(support)
        validation_query = torch.zeros_like(val_y, dtype=torch.bool)
        validation_query.reshape(-1)[torch.as_tensor(query)] = True
        self.validation_support_months_ = support.copy()
        self.validation_query_cells_ = np.asarray(query).copy()
        self.n_source_stations_, self.n_validation_stations_ = len(source_x), len(validation_x)
        self.n_months_, self.n_validation_query_ = source_x.shape[1], len(query)
        self.n_source_query_ = int(source_observed.sum() - 5 * len(source_x))
        delta = torch.nn.Parameter(torch.zeros_like(self.initial_readout))
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        self.trace_, self.source_episode_schedules_ = [], []
        self.best_epoch_ = 0
        best_score, stale = float("inf"), 0
        best_delta = delta.detach().clone()

        def evaluate(epoch, training_mae, gradient_max=None):
            nonlocal best_score, best_delta, stale
            readout = self.initial_readout + self.initial_norm * delta
            sums = np.zeros(len(TRAIN_K) * len(TRAIN_RIDGE), dtype=float)
            rms = []
            with torch.no_grad():
                for start in range(0, len(validation_x), self.batch_size):
                    sl = slice(start, start+self.batch_size)
                    basis, stats = self._basis(validation_x[sl], readout, return_stats=True)
                    _, losses = _tensor_episode_loss(basis, val_z[sl], val_y[sl],
                                                     validation_support[sl], validation_query[sl], pooled=True)
                    sums += losses.numpy() * int(validation_query[sl].sum())
                    rms.extend(stats["station_rms"].tolist())
            components = sums / self.n_validation_query_
            score = float(components.mean())
            improved = score < best_score
            if improved:
                best_score, stale, self.best_epoch_ = score, 0, epoch
                best_delta = delta.detach().clone()
            else:
                stale += 1
            row = {"epoch": epoch, "training_mae": training_mae, "validation_mae": score,
                   "validation_by_k_ridge": [
                       {"k": k, "ridge_strength": strength, "mae": float(value)}
                       for (k, strength), value in zip(
                           [(k, r) for k in TRAIN_K for r in TRAIN_RIDGE], components, strict=True)],
                   "is_best": improved, "stale_epochs": stale,
                   "delta_norm": float(delta.detach().norm()),
                   "readout_parameter_distance": float((readout.detach()-self.initial_readout).norm()),
                   "readout_norm": float(readout.detach().norm()),
                   "gradient_norm_max_before_clip": gradient_max,
                   "validation_anchor_floor_hits": int((np.asarray(rms) < self.scale_floor).sum()),
                   "validation_anchor_rms_min": float(np.min(rms)),
                   "validation_anchor_rms_max": float(np.max(rms))}
            self.trace_.append(row)
            if improved:
                self.validation_metrics_ = copy.deepcopy(row)
            if progress is not None:
                progress(copy.deepcopy(row))

        evaluate(0, None)
        for epoch in range(1, self.max_epochs+1):
            source_support = stratified_support_months(source_observed, seed=self.seed, epoch=epoch)
            source_query = torch.as_tensor(_episode_queries(source_observed, source_support))
            self.source_episode_schedules_.append({"epoch": epoch,
                "support_months": source_support.tolist()})
            source_support = torch.as_tensor(source_support)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 2**31-1]))
            order = rng.permutation(len(source_x))
            total, gradient_max = 0.0, 0.0
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start+self.batch_size]
                optimizer.zero_grad()
                readout = self.initial_readout + self.initial_norm * delta
                basis = self._basis(source_x[rows], readout)
                loss, _ = _tensor_episode_loss(basis, source_z[rows], source_y[rows],
                                             source_support[rows], source_query[rows], pooled=False)
                loss.backward()
                if delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise FloatingPointError("nonfinite support-readout gradient")
                grad = torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0, error_if_nonfinite=True)
                gradient_max = max(gradient_max, float(grad))
                optimizer.step()
                if not torch.isfinite(delta).all():
                    raise FloatingPointError("nonfinite support-readout parameters")
                total += float(loss.detach()) * len(rows)
            evaluate(epoch, total/len(source_x), gradient_max)
            if stale >= self.patience:
                break
        self.epochs_run_ = len(self.trace_)-1
        self.delta_ = best_delta
        self.readout_ = self.initial_readout + self.initial_norm * self.delta_
        return self

    def transform(self, hidden):
        """Return selected label-free normalized basis, flat [station*month,2]."""
        if not hasattr(self, "readout_"):
            raise RuntimeError("fit the readout before transform")
        with torch.no_grad():
            return self._basis(self._hidden(hidden), self.readout_).reshape(-1, 2).numpy().copy()

    def transform_initial(self, hidden):
        """Return the exact fixed-readout control, without fitting."""
        with torch.no_grad():
            return self._basis(self._hidden(hidden), self.initial_readout).reshape(-1, 2).numpy().copy()

    def _config(self):
        return {name: getattr(self, name) for name in
                ("anchor_count", "scale_floor", "seed", "max_epochs", "patience", "lr", "batch_size")}

    def to_dict(self):
        if not hasattr(self, "readout_"):
            raise RuntimeError("fit the readout before serialization")
        return {"version": 1, "model_class": "EpisodicSupportReadout", "config": self._config(),
                "hidden_size": self.hidden_size, "output_dim": 2,
                "trainable_parameter_count": self.trainable_parameter_count_,
                "initial_readout_norm": self.initial_norm,
                "readout_parameter_distance": float((self.readout_-self.initial_readout).norm()),
                "delta_norm": float(self.delta_.norm()), "readout_norm": float(self.readout_.norm()),
                "best_epoch": self.best_epoch_, "epochs_run": self.epochs_run_,
                "trace": copy.deepcopy(self.trace_), "validation_metrics": copy.deepcopy(self.validation_metrics_),
                "n_source_stations": self.n_source_stations_, "n_validation_stations": self.n_validation_stations_,
                "n_months": self.n_months_, "n_source_query": self.n_source_query_,
                "n_validation_query": self.n_validation_query_,
                "source_station_ids": self.source_station_ids_.tolist(),
                "excluded_source_station_ids": self.excluded_source_station_ids_.tolist(),
                "source_episode_schedules": copy.deepcopy(self.source_episode_schedules_),
                "validation_support_months": self.validation_support_months_.tolist(),
                "validation_query_cells": self.validation_query_cells_.tolist(),
                "protocol": {"parameterization": "P=P0+frobenius_norm(P0)*D", "delta_initialization": "zero",
                    "alpha": 1.0, "k_values": list(TRAIN_K), "ridge_strengths": list(TRAIN_RIDGE),
                    "training_loss": "raw_mae_equal_station_mean_k_ridge",
                    "validation_loss": "raw_mae_pooled_query_mean_k_ridge",
                    "selection_role": "source_validation", "source_base_scale": "native",
                    "source_base_role": "frozen forest OOF plus source-trained native neural correction; not full neural OOF",
                    "source_support": "stratified_observed_rank_five_bins", "queries": "all five supports excluded",
                    "projection_batch_size": PROJECTION_BATCH_SIZE, "gradient_clip_norm": 1.0,
                    "hidden_experts_updated": False, "pca_whitening_qr": False,
                    "normalization": "retrospective fixed calendar anchors and one station scalar RMS"}}

    def summary(self):
        return self.to_dict()

    def to_payload(self):
        summary = self.to_dict()
        return {"version": 1, "model_class": "EpisodicSupportReadout", "config": self._config(),
                "initial_readout": self.initial_readout.clone(), "delta": self.delta_.clone(),
                "summary": summary}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1 or payload.get("model_class") != "EpisodicSupportReadout":
            raise ValueError("unsupported support-readout checkpoint")
        obj = cls(payload["initial_readout"], **payload["config"])
        delta = torch.as_tensor(payload["delta"], dtype=torch.float64).detach().cpu().clone()
        if delta.shape != obj.initial_readout.shape or not torch.isfinite(delta).all():
            raise ValueError("invalid saved readout delta")
        obj.delta_, obj.readout_ = delta, obj.initial_readout + obj.initial_norm*delta
        summary = copy.deepcopy(payload["summary"])
        if summary["config"] != obj._config():
            raise ValueError("checkpoint configuration disagrees with its summary")
        for name in ("best_epoch", "epochs_run", "n_source_stations", "n_validation_stations",
                     "n_months", "n_source_query", "n_validation_query"):
            setattr(obj, name+"_", int(summary[name]))
        for name in ("trace", "validation_metrics", "source_episode_schedules"):
            setattr(obj, name+"_", copy.deepcopy(summary[name]))
        for name in ("source_station_ids", "excluded_source_station_ids",
                     "validation_support_months", "validation_query_cells"):
            setattr(obj, name+"_", np.asarray(summary[name], dtype=np.int64))
        if obj.to_dict() != summary:
            raise ValueError("saved readout coefficients and summary disagree")
        return obj
