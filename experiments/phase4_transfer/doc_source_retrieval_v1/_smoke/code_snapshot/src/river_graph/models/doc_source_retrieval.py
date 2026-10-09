"""Source-only, portable DOC residual profiles and sparse donor attention.

These are ecological retrieval links, not physical river edges. Donor profiles
estimate the forest residual already modelled by the temporal expert; callers
blend the two experts rather than double-counting their corrections.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn


class SourceResidualBank:
    """Fit donor profiles from selected station-blocked OOF forest errors."""

    def __init__(self, *, ridge=10.0, max_donors=20):
        if ridge <= 0 or max_donors < 1:
            raise ValueError("positive ridge and donor count required")
        self.ridge, self.max_donors = float(ridge), int(max_donors)

    def fit(self, ecology, hydro, hydro_valid, months, cells, oof_prediction, truth,
            *, station_names, excluded_station_names=()):
        ecology = np.asarray(ecology, dtype=np.float64)
        hydro = np.asarray(hydro, dtype=np.float64)
        valid = np.asarray(hydro_valid, dtype=bool)
        cells = np.asarray(cells, dtype=np.int64)
        months = np.asarray(months, dtype="datetime64[M]")
        names = np.asarray(station_names, dtype=str)
        if (ecology.ndim != 2 or ecology.shape[1] != 9 or hydro.shape != (*valid.shape[:2], 2)
                or valid.shape != hydro.shape or hydro.shape[:2] != (len(names), len(months))
                or len(ecology) != len(names) or cells.ndim != 1 or not len(cells)
                or cells.min() < 0 or cells.max() >= hydro.shape[0]*hydro.shape[1]
                or len(np.unique(cells)) != len(cells) or len(np.unique(names)) != len(names)):
            raise ValueError("aligned ecology/hydro/calendar, unique stations and selected cells required")
        p, y = np.asarray(oof_prediction, dtype=float), np.asarray(truth, dtype=float)
        if p.shape != cells.shape or y.shape != cells.shape or not np.isfinite([p, y]).all() or np.any(p < 0):
            raise ValueError("finite selected OOF predictions/truth required")
        ids, inverse, counts = np.unique(cells // len(months), return_inverse=True, return_counts=True)
        if set(names[ids]) & set(map(str, excluded_station_names)):
            raise ValueError("excluded episode stations cannot enter the donor bank")
        self.station_names_ = names[ids].copy()
        weights = 1 / (len(ids)*counts[inverse])
        eco = ecology[ids]
        seen = np.isfinite(eco) & (eco != -1)
        self.eco_median_ = np.array([np.median(eco[seen[:, j], j]) if seen[:, j].any() else 0
                                     for j in range(9)])
        self.eco_scale_ = np.array([max(np.subtract(*np.percentile(eco[seen[:, j], j], [75, 25])), 1e-6)
                                    if seen[:, j].any() else 1 for j in range(9)])
        self.eco_active_ = seen.any(0)
        self.ecology_, self.ecology_valid_ = self._ecology(eco)
        selected_hydro, selected_valid = hydro[ids], valid[ids]
        transformed = self._hydro_transform(selected_hydro, selected_valid)
        self.hydro_mean_, self.hydro_scale_ = np.zeros(2), np.ones(2)
        for j in range(2):
            values = transformed[..., j][selected_valid[..., j]]
            if len(values):
                self.hydro_mean_[j], self.hydro_scale_[j] = np.mean(values), max(np.std(values), 1e-6)
        z = np.log1p(p)
        self.context_mean_ = float(weights @ z)
        self.context_scale_ = max(float(np.sqrt(weights @ (z-self.context_mean_)**2)), 1e-6)
        self.residual_scale_ = max(float(weights @ np.abs(y-p)), 1e-6)
        design = self.design(p, hydro[cells//len(months), cells % len(months)],
                             valid[cells//len(months), cells % len(months)], months[cells % len(months)])
        residual = (y-p)/self.residual_scale_
        global_coef = np.linalg.solve(design.T@(weights[:, None]*design)+self.ridge*np.eye(6),
                                      design.T@(weights*residual))
        profiles = []
        for station in range(len(ids)):
            rows = inverse == station
            x, target = design[rows], residual[rows]
            # Shrink a donor's response to a station-balanced population profile.
            profiles.append(np.linalg.solve(x.T@x/rows.sum()+self.ridge*np.eye(6),
                            x.T@target/rows.sum()+self.ridge*global_coef))
        self.profiles_ = np.asarray(profiles)
        normalized_hydro = self.hydro_features(selected_hydro, selected_valid)[..., :2]
        means = np.sum(normalized_hydro, axis=1)/np.maximum(selected_valid.sum(1), 1)
        coverage = selected_valid.mean(1)
        self.keys_ = np.column_stack([self.ecology_, self.ecology_valid_, means, coverage,
                                      self.profiles_]).astype(np.float32)
        self.source_counts_ = counts.copy()
        return self

    @staticmethod
    def _hydro_transform(hydro, valid):
        values = np.where(valid, hydro, 0).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("visible hydro must be finite")
        values[..., 1] = np.log1p(np.maximum(values[..., 1], 0))
        return values

    def _ecology(self, values):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values) & (values != -1)
        z = (np.where(valid, values, self.eco_median_)-self.eco_median_)/self.eco_scale_
        z[..., ~self.eco_active_] = 0
        return (z/(1+np.abs(z))).astype(np.float32), valid.astype(np.float32)

    def hydro_features(self, hydro, valid):
        valid = np.asarray(valid, dtype=bool)
        z = (self._hydro_transform(hydro, valid)-self.hydro_mean_)/self.hydro_scale_
        z = np.where(valid, z/(1+np.abs(z)), 0)
        return np.concatenate([z, valid], axis=-1).astype(np.float32)

    def design(self, context, hydro, valid, months):
        phase = 2*math.pi*(np.asarray(months, dtype="datetime64[M]").astype(int) % 12)/12
        z = (np.log1p(context)-self.context_mean_)/self.context_scale_
        h = self.hydro_features(hydro, valid)[..., :2]
        return np.column_stack([np.ones(len(z)), z/(1+np.abs(z)), np.sin(phase), np.cos(phase), h])

    def query(self, ecology, hydro, hydro_valid, months, context, hidden, *, station_names):
        """Prepare arbitrary new stations; exclude identity-matched source donors."""
        names = np.asarray(station_names, dtype=str)
        ecology, eco_valid = self._ecology(ecology)
        context, hidden = np.asarray(context), np.asarray(hidden)
        if (context.ndim != 1 or hidden.ndim != 2 or hidden.shape[0] != len(context)
                or len(names) != len(context) or len(ecology) != len(context)
                or not np.isfinite(hidden).all()):
            raise ValueError("selected query rows require aligned finite hidden states and station names")
        design = self.design(context, hydro, hydro_valid, months)
        query = np.column_stack([ecology, eco_valid, self.hydro_features(hydro, hydro_valid),
                                 design[:, 1:4], hidden]).astype(np.float32)
        distance = np.mean((ecology[:, None]-self.ecology_[None])**2, axis=-1)
        same = names[:, None] == self.station_names_[None]
        distance[same] = np.inf
        k = min(self.max_donors, len(self.station_names_))
        # Stable bank order resolves tied ecological distances without a target label.
        donors = np.argsort(distance, axis=1, kind="stable")[:, :k]
        finite = np.take_along_axis(np.isfinite(distance), donors, axis=1)
        if not finite.any(1).all():
            raise ValueError("no eligible donor remains for a query")
        profiles = self.profiles_[donors]
        residual = np.einsum("bkd,bd->bk", profiles, design)
        value = np.concatenate([residual[..., None], np.broadcast_to(design[:, None], profiles.shape)], axis=-1)
        return {"query": query, "keys": self.keys_[donors], "values": value.astype(np.float32),
                "valid": finite, "donor_names": self.station_names_[donors],
                "nearest_distance": np.take_along_axis(distance, donors[:, :1], axis=1).ravel()}

    def to_dict(self):
        return {"version": 1, "ridge": self.ridge, "max_donors": self.max_donors,
                **{key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in self.__dict__.items() if key.endswith("_")}}

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported source bank")
        obj = cls(ridge=state["ridge"], max_donors=state["max_donors"])
        for key, value in state.items():
            if key.endswith("_"):
                setattr(obj, key, np.asarray(value) if isinstance(value, list) else value)
        return obj


class SourceRetrievalAttention(nn.Module):
    """Two sparse32-dimensional heads; zero projection preserves old memory."""

    def __init__(self, query_dim, key_dim=28, *, heads=2, head_dim=32, dropout=.1, seed=42):
        super().__init__()
        self.settings = {"query_dim": int(query_dim), "key_dim": int(key_dim), "heads": int(heads),
                         "head_dim": int(head_dim), "dropout": float(dropout), "seed": int(seed)}
        self.heads, self.head_dim = int(heads), int(head_dim)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.q = nn.Linear(query_dim, heads*head_dim)
            self.k = nn.Linear(key_dim, heads*head_dim)
            self.projection = nn.Linear(heads*7, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys, values, valid, *, diagnostics=False):
        q = self.q(query).reshape(len(query), self.heads, self.head_dim)
        k = self.k(keys).reshape(len(query), keys.shape[1], self.heads, self.head_dim)
        score = torch.einsum("bhd,bkhd->bhk", q, k)/math.sqrt(self.head_dim)
        score = score.masked_fill(~valid[:, None], -torch.inf)
        weights = torch.softmax(score, dim=-1)
        if not torch.isfinite(weights).all():
            raise ValueError("each query needs at least one valid donor")
        attended = torch.einsum("bhk,bkd->bhd", self.dropout(weights), values)
        delta = self.projection(attended.flatten(1)).squeeze(-1)
        if diagnostics:
            entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(1)
            return delta, {"entropy": entropy, "effective_donors": entropy.exp(),
                           "max_weight": weights.max(-1).values.mean(1), "weights": weights}
        return delta

    def to_payload(self):
        return {"version": 1, "settings": self.settings.copy(),
                "state_dict": {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1:
            raise ValueError("unsupported retrieval checkpoint")
        obj = cls(**payload["settings"])
        obj.load_state_dict(payload["state_dict"])
        return obj.eval()


def fit_retrieval(episodes, validation, *, residual_scale, seed=42, epochs=30, patience=5,
                  batch_size=512, progress=None):
    """Fit on pseudo-target source labels; select using source-validation only."""
    model = SourceRetrievalAttention(episodes[0]["query"].shape[1], episodes[0]["keys"].shape[-1], seed=seed)
    tensors = []
    for episode in [*episodes, validation]:
        selected = {name: torch.as_tensor(episode[name], dtype=torch.bool if name == "valid" else torch.float32)
                    for name in ("query", "keys", "values", "valid", "static", "context", "temporal", "truth")}
        tensors.append(selected)
    validation_t = tensors.pop()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, stale, state, trace, selected_gamma = np.inf, 0, None, [], 0.0
    selected_epoch = 0

    def evaluate(epoch, loss):
        nonlocal best, stale, state, selected_gamma, selected_epoch
        model.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, len(validation_t["query"]), batch_size):
                rows = slice(start, start+batch_size)
                chunks.append(model(*(validation_t[key][rows] for key in ("query", "keys", "values", "valid"))))
            memory = validation_t["static"] + residual_scale*torch.cat(chunks)
            scores = []
            for gamma in (0., .25, .5, 1.):
                prediction = (validation_t["context"]+(1-gamma)*(validation_t["temporal"]-validation_t["context"])
                              +gamma*memory).clamp_min(0)
                scores.append({"gamma": gamma, "mae": float((prediction-validation_t["truth"]).abs().mean())})
        choice = min(scores, key=lambda row: (row["mae"], row["gamma"]))
        if choice["mae"] < best:
            best, stale, state = choice["mae"], 0, copy.deepcopy(model.state_dict())
            selected_gamma, selected_epoch = choice["gamma"], epoch
        else:
            stale += 1
        row = {"epoch": epoch, "training_loss": loss, "validation_mae": choice["mae"],
               "gamma": choice["gamma"], "best_epoch": selected_epoch, "candidates": scores}
        trace.append(row)
        if progress:
            progress(row)

    evaluate(0, None)
    for epoch in range(1, epochs+1):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        errors = []
        for episode_index, episode in enumerate(tensors):
            order = rng.permutation(len(episode["query"]))
            # Episodes are equally represented; the caller supplies station-balanced weights.
            weights = torch.as_tensor(episodes[episode_index]["weights"], dtype=torch.float32)
            total, count = 0., 0
            for start in range(0, len(order), batch_size):
                rows = order[start:start+batch_size]
                optimizer.zero_grad()
                delta = model(*(episode[key][rows] for key in ("query", "keys", "values", "valid")))
                pred = (episode["context"][rows]+episode["static"][rows]+residual_scale*delta).clamp_min(0)
                loss = (weights[rows]*(pred-episode["truth"][rows]).abs()).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(rows)
                count += len(rows)
            errors.append(total/count)
        evaluate(epoch, float(np.mean(errors)))
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, {"selection_role": "source_validation", "best_epoch": selected_epoch,
                   "gamma": selected_gamma, "validation_mae": best, "trace": trace,
                   "epochs_run": len(trace)-1, "residual_scale": residual_scale}
