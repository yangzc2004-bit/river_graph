"""Supervised station adaptation on frozen source feature representations.

Only a two-row projection is learned. Its input is source-fitted, whitened PCA
of within-station feature dynamics; the source experts are never updated here.
Support ridge heads are solved inside each episode. Training gives stations
equal weight, while checkpoint selection pools source-validation query cells.
All five support candidates are excluded from queries at both K=3 and K=5.

Whole-timeline feature centering and record-spanning support are retrospective
reconstruction operations. They do not implement prospective forecasting.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.support_shape_adapter import StationTemporalBasis

TRAIN_K = (3, 5)
TRAIN_RIDGE = (1.0, 10.0)
SUPPORT_ORDER = np.array([0, 2, 4, 1, 3])


def _integer(value, name, *, minimum=1):
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def stratified_support_months(mask, *, seed, epoch):
    """Sample one observed month per chronological rank bin, without labels.

    Random streams depend only on seed, epoch and source row identity, so the
    GRU and tree representations receive the same episodes. Columns are nested
    in the order first, middle, last, first quarter, third quarter.
    """
    observed = np.asarray(mask)
    if observed.ndim != 2 or observed.dtype.kind != "b":
        raise ValueError("mask must be a two-dimensional boolean array")
    seed = _integer(seed, "seed", minimum=0)
    epoch = _integer(epoch, "epoch", minimum=0)
    result = np.empty((len(observed), 5), dtype=np.int64)
    for station, row in enumerate(observed):
        months = np.flatnonzero(row)
        if len(months) <= 5:
            raise ValueError("each episodic station needs five supports and at least one query")
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, station]))
        sampled = np.array([rng.choice(part) for part in np.array_split(months, 5)])
        result[station] = sampled[SUPPORT_ORDER]
    return result


def _episode_queries(mask, support):
    query = mask.copy()
    query[np.arange(len(mask))[:, None], support] = False
    if not query.any(axis=1).all():
        raise ValueError("each episodic station needs at least one query")
    return query


def _tensor_episode_loss(projected, base_z, truth, support, query_mask, *, pooled):
    """Differentiable mean-loss ridge, matching SupportShapeAdapter at alpha=1.

    Returns the mean of four raw MAEs plus their K/ridge components. No upper
    clipping is applied before or after expm1; overflow fails explicitly.
    """
    rows = torch.arange(len(projected), device=projected.device)[:, None]
    eye = torch.eye(projected.shape[-1], dtype=projected.dtype, device=projected.device)
    losses = []
    for k in TRAIN_K:
        chosen = support[:, :k]
        basis = projected[rows, chosen]
        mean_basis = basis.mean(dim=1, keepdim=True)
        centered = basis - mean_basis
        residual = torch.log1p(truth[rows, chosen]) - base_z[rows, chosen]
        mean_residual = residual.mean(dim=1, keepdim=True)
        gram = centered.transpose(1, 2) @ centered / k
        rhs = (centered.transpose(1, 2) @ (residual - mean_residual).unsqueeze(-1)) / k
        for strength in TRAIN_RIDGE:
            coefficients = torch.linalg.solve(gram + strength * eye, rhs)
            delta = ((projected - mean_basis) @ coefficients).squeeze(-1)
            z = base_z + mean_residual + delta
            # Unobserved/padded cells are not evaluated, including their exp.
            prediction = torch.expm1(torch.where(query_mask, z, 0.0)).clamp_min(0.0)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite episodic prediction; no upper clipping is applied")
            absolute = torch.where(query_mask, (prediction - truth).abs(), 0.0)
            if pooled:
                loss = absolute.sum() / query_mask.sum()
            else:
                loss = (absolute.sum(dim=1) / query_mask.sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite episodic raw MAE")
            losses.append(loss)
    stacked = torch.stack(losses)
    return stacked.mean(), stacked


class EpisodicStationProjector:
    """Learn a two-dimensional support-adaptable subspace of frozen features.

    ``source_truth`` and validation values are native nonnegative DOC. Source
    base predictions are already log1p, normally from the saved station OOF
    forest. Validation base predictions are native. Only masks select labels;
    unobserved label entries can be NaN. Prediction is a label-free projection.

    The default 2x16 projection has 32 raw parameters and 29 orthonormal degrees
    of freedom. QR retraction prevents scaling the basis to weaken ridge loss.
    Near-zero source PCA modes are zeroed and excluded from the learned rows.
    """

    def __init__(self, n_components=16, output_dim=2, epochs=100, patience=15,
                 learning_rate=0.01, batch_size=32, seed=42):
        self.n_components = _integer(n_components, "n_components", minimum=2)
        self.output_dim = _integer(output_dim, "output_dim", minimum=2)
        if self.output_dim != 2:
            raise ValueError("the episodic projection has exactly two output dimensions")
        self.epochs = _integer(epochs, "epochs", minimum=0)
        self.patience = _integer(patience, "patience")
        self.batch_size = _integer(batch_size, "batch_size")
        self.seed = _integer(seed, "seed", minimum=0)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)

    @staticmethod
    def _label_arrays(features, base, truth, mask, *, base_native):
        shape = np.shape(features)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("features must be [station, month, feature]")
        observed = np.asarray(mask)
        values, prediction = np.asarray(truth, dtype=float), np.asarray(base, dtype=float)
        if (observed.dtype.kind != "b" or observed.shape != shape[:2]
                or values.shape != shape[:2] or prediction.shape != shape[:2]):
            raise ValueError("base, truth and boolean mask must align with feature station-month rows")
        if (not np.isfinite(values[observed]).all() or (values[observed] < 0).any()
                or not np.isfinite(prediction[observed]).all()
                or (base_native and (prediction[observed] < 0).any())):
            raise ValueError("observed labels and base predictions must be finite and valid")
        # Unselected labels never enter a reduction, a transform or an episode.
        values = np.where(observed, values, 0.0)
        prediction = np.where(observed, prediction, 0.0)
        if base_native:
            prediction = np.log1p(prediction)
        return values, prediction, observed.copy()

    def _whiten(self, features):
        result = self.basis_.transform(features)
        result[..., ~self.active_modes_] = 0.0
        return result

    def fit(self, source_features, source_base_z, source_truth, source_mask,
            validation_features, validation_base_native, validation_truth,
            validation_mask, *, selection_role, progress=None):
        if selection_role != "source_validation":
            raise ValueError("checkpoint selection requires source_validation")
        source_y, source_z, source_observed = self._label_arrays(
            source_features, source_base_z, source_truth, source_mask, base_native=False)
        validation_y, validation_z, validation_observed = self._label_arrays(
            validation_features, validation_base_native, validation_truth,
            validation_mask, base_native=True)
        if np.shape(source_features)[-1] != np.shape(validation_features)[-1]:
            raise ValueError("source and validation feature dimensions must match")
        self.source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) > 5)
        self.excluded_source_station_ids_ = np.flatnonzero(source_observed.sum(axis=1) <= 5)
        if not len(self.source_station_ids_):
            raise ValueError("no source station has five supports and at least one query")
        # Fit all source covariates; no validation features or labels set PCA.
        self.basis_ = StationTemporalBasis(self.n_components).fit_station_sequences(source_features)
        self.eigenvalue_floor_ = max(1e-8, 1e-5 * float(self.basis_.eigenvalues_[0]))
        self.basis_.eigenvalue_floor = self.eigenvalue_floor_
        self.active_modes_ = self.basis_.eigenvalues_ >= self.eigenvalue_floor_
        self.effective_rank_ = int(self.active_modes_.sum())
        if self.effective_rank_ < 2:
            raise ValueError("source PCA effective rank must be at least two")
        source_x = torch.as_tensor(self._whiten(source_features)[self.source_station_ids_])
        validation_x = torch.as_tensor(self._whiten(validation_features))
        source_y = torch.as_tensor(source_y[self.source_station_ids_])
        source_z = torch.as_tensor(source_z[self.source_station_ids_])
        source_observed = source_observed[self.source_station_ids_]
        validation_y, validation_z = torch.as_tensor(validation_y), torch.as_tensor(validation_z)
        schedule, query = support_schedule(
            np.flatnonzero(validation_observed), validation_observed.shape[1])
        if not len(query) or len(schedule) != len(validation_observed):
            raise ValueError("every validation station needs five supports and at least one query")
        validation_support = torch.as_tensor(np.stack([
            schedule[i] % validation_observed.shape[1] for i in range(len(validation_observed))]))
        validation_query = torch.zeros_like(validation_y, dtype=torch.bool)
        validation_query.view(-1)[torch.as_tensor(query)] = True
        self.n_source_stations_ = len(source_x)
        self.n_validation_stations_ = len(validation_x)
        self.n_validation_query_ = len(query)
        self.projection_ = np.eye(self.n_components, dtype=float)[:2].copy()
        projection = torch.nn.Parameter(torch.as_tensor(self.projection_.copy()))
        optimizer = torch.optim.Adam([projection], lr=self.learning_rate)
        active = torch.as_tensor(self.active_modes_)
        self.trace_, self.best_epoch_ = [], 0
        best_score, stale = float("inf"), 0

        def evaluate(epoch, training_mae):
            nonlocal best_score, stale
            with torch.no_grad():
                loss, components = _tensor_episode_loss(
                    validation_x @ projection.T, validation_z, validation_y,
                    validation_support, validation_query, pooled=True)
            score = float(loss)
            improved = score < best_score
            if improved:
                best_score, stale, self.best_epoch_ = score, 0, epoch
                self.projection_ = projection.detach().numpy().copy()
            else:
                stale += 1
            component_metrics = [
                {"k": k, "ridge_strength": strength, "mae": float(value)}
                for (k, strength), value in zip(
                    [(k, r) for k in TRAIN_K for r in TRAIN_RIDGE], components)]
            row = {"epoch": epoch, "training_mae": training_mae,
                   "validation_mae": score, "validation_by_k_ridge": component_metrics,
                   "is_best": improved, "stale_epochs": stale}
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
            order = rng.permutation(len(source_x))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                stations = torch.as_tensor(order[start:start + self.batch_size])
                optimizer.zero_grad()
                loss, _ = _tensor_episode_loss(
                    source_x[stations] @ projection.T, source_z[stations], source_y[stations],
                    support[stations], query_mask[stations], pooled=False)
                loss.backward()
                if projection.grad is None or not torch.isfinite(projection.grad).all():
                    raise FloatingPointError("nonfinite episodic projection gradient")
                optimizer.step()
                with torch.no_grad():
                    projection[:, ~active] = 0.0
                    q, r = torch.linalg.qr(projection.T, mode="reduced")
                    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
                    projection.copy_((q * signs).T)
                    if not torch.isfinite(projection).all():
                        raise FloatingPointError("nonfinite episodic projection after retraction")
                total += float(loss.detach()) * len(stations)
            evaluate(epoch, total / len(source_x))
            if stale >= self.patience:
                break
        self.epochs_run_ = len(self.trace_) - 1
        return self

    def transform(self, features):
        """Project full station feature timelines without reading any labels."""
        if not hasattr(self, "projection_"):
            raise RuntimeError("fit the episodic projector before transform")
        result = self._whiten(features) @ self.projection_.T
        if not np.isfinite(result).all():
            raise ValueError("episodic projected features are not finite")
        return result

    def transform_pca(self, features):
        """Return the first two axes of the exact same frozen source whitener."""
        if not hasattr(self, "projection_"):
            raise RuntimeError("fit the episodic projector before transform")
        return self._whiten(features)[..., :2]

    def to_dict(self):
        if not hasattr(self, "epochs_run_"):
            raise RuntimeError("fit the episodic projector before serialization")
        return {"version": 1, "config": {
            name: getattr(self, name) for name in (
                "n_components", "output_dim", "epochs", "patience", "learning_rate",
                "batch_size", "seed")},
            "basis": self.basis_.to_dict(), "projection": self.projection_.tolist(),
            "projection_parameter_displacement": float(np.linalg.norm(
                self.projection_ - np.eye(self.n_components)[:2])),
            "projection_subspace_displacement": float(np.linalg.norm(
                self.projection_.T @ self.projection_
                - np.diag([1.0, 1.0] + [0.0] * (self.n_components - 2)))),
            "active_modes": self.active_modes_.tolist(), "effective_rank": self.effective_rank_,
            "eigenvalue_floor": self.eigenvalue_floor_, "best_epoch": self.best_epoch_,
            "epochs_run": self.epochs_run_, "trace": copy.deepcopy(self.trace_),
            "validation_metrics": copy.deepcopy(self.validation_metrics_),
            "source_station_ids": self.source_station_ids_.tolist(),
            "excluded_source_station_ids": self.excluded_source_station_ids_.tolist(),
            "n_source_stations": self.n_source_stations_,
            "n_validation_stations": self.n_validation_stations_,
            "n_validation_query": self.n_validation_query_,
            "protocol": {"alpha": 1.0, "k_values": list(TRAIN_K),
                         "ridge_strengths": list(TRAIN_RIDGE),
                         "training_loss": "raw_mae_equal_station_mean_k_ridge",
                         "validation_loss": "raw_mae_pooled_query_mean_k_ridge",
                         "source_support": "stratified_observed_rank_five_bins",
                         "validation_support": "spatial_fewshot.support_schedule",
                         "selection_role": "source_validation"}}

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported episodic-projector version")
        obj = cls(**state["config"])
        obj.basis_ = StationTemporalBasis.from_dict(state["basis"])
        obj.projection_ = np.asarray(state["projection"], dtype=float)
        obj.active_modes_ = np.asarray(state["active_modes"], dtype=bool)
        obj.effective_rank_ = int(state["effective_rank"])
        obj.eigenvalue_floor_ = float(state["eigenvalue_floor"])
        if (obj.basis_.n_components != obj.n_components
                or obj.projection_.shape != (2, obj.n_components)
                or obj.active_modes_.shape != (obj.n_components,)
                or not np.isfinite(obj.projection_).all()
                or obj.effective_rank_ != obj.active_modes_.sum() or obj.effective_rank_ < 2
                or not np.array_equal(obj.active_modes_,
                                      obj.basis_.eigenvalues_ >= obj.eigenvalue_floor_)
                or obj.eigenvalue_floor_ != obj.basis_.eigenvalue_floor
                or not np.allclose(obj.projection_ @ obj.projection_.T, np.eye(2), atol=1e-10)
                or not np.allclose(obj.projection_[:, ~obj.active_modes_], 0.0, atol=1e-12)):
            raise ValueError("invalid serialized episodic projection")
        obj.best_epoch_, obj.epochs_run_ = int(state["best_epoch"]), int(state["epochs_run"])
        obj.trace_, obj.validation_metrics_ = copy.deepcopy(state["trace"]), copy.deepcopy(
            state["validation_metrics"])
        obj.source_station_ids_ = np.asarray(state["source_station_ids"], dtype=np.int64)
        obj.excluded_source_station_ids_ = np.asarray(state["excluded_source_station_ids"], dtype=np.int64)
        obj.n_source_stations_ = int(state["n_source_stations"])
        obj.n_validation_stations_ = int(state["n_validation_stations"])
        obj.n_validation_query_ = int(state["n_validation_query"])
        return obj
