"""Real upstream path messages added to the existing DOC retrieval residual.

Receiving labels never enter a message bank. Source training banks use the
existing double-held tree references; this is joint neural fitting, not a
post-hoc fit to in-sample residuals of the complete neural predictor.
"""
from __future__ import annotations

import copy
import heapq
import math

import numpy as np
import torch
from torch import nn

from river_graph.models.relative_source_attention import relative_source_residual_grid
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)

LAGS = (0, 1, 3)
PATH_DIM = 8


def upstream_paths(edges, structures, receivers, eligible, *, candidates=20, max_km=3000.):
    """Find nearest eligible ancestors through the directed monitored graph.

    Unavailable intermediate stations remain physical waypoints, never donors.
    Path length, storage length and junction counts are additive atlas summaries.
    """
    if type(candidates) is not int or candidates < 1:
        raise ValueError("positive integer candidate count required")
    eligible = set(map(str, eligible))
    parents = {}
    for row in edges.itertuples(index=False):
        if not row.mainstem_connected:
            continue
        distance = float(row.path_length_km)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("positive finite mapped river path length required")
        parents.setdefault(str(row.target), []).append((str(row.source), distance,
            float(row.path_major_junctions), float(row.path_storage_km)))
    physical = structures.set_index("station")
    physical.index = physical.index.astype(str)
    output = []
    for receiver in map(str, receivers):
        if receiver not in physical.index:
            raise ValueError("receiving station missing physical attributes")
        queue, visited, selected = [(0., receiver, 0., 0.)], set(), []
        while queue:
            distance, station, major, storage = heapq.heappop(queue)
            if station in visited or distance > max_km:
                continue
            visited.add(station)
            if station != receiver and station in eligible:
                selected.append((station, distance, major, storage))
            for source, length, junctions, water in parents.get(station, []):
                heapq.heappush(queue, (distance+length, source, major+junctions, storage+water))
        selected.sort(key=lambda item: (item[1], item[0]))
        output.append(selected[:candidates])
    return output


def path_candidates(edges, structures, receivers, bank_names, residual, *, eligible=None,
                    candidates=20, rewired=False, seed=42, max_age=12):
    """Gather causal observed source departures at lags 0, 1, 3.

    A retained last observation is explicitly aged and invalid after max_age;
    no missing DOC is presented as a fresh observation. Rewiring keeps receiver
    path slots/attributes, replaces donors by eligible non-ancestors closest in
    log drainage area, and does not claim the resulting paths are real rivers.
    """
    names, receivers = np.asarray(bank_names, str), np.asarray(receivers, str)
    residual = np.asarray(residual, float)
    if (residual.ndim != 2 or residual.shape[0] != len(names) or len(set(names)) != len(names)
            or np.isinf(residual).any() or type(max_age) is not int or max_age < 0):
        raise ValueError("unique source bank with finite-or-missing residuals and nonnegative age required")
    allowed = set(names if eligible is None else map(str, eligible))
    if not allowed <= set(names):
        raise ValueError("eligible donors must belong to the saved source bank")
    paths = upstream_paths(edges, structures, receivers, allowed, candidates=candidates)
    # Exclude all eligible physical ancestors from the rewiring pool, not just
    # the first twenty. Node degrees/candidate counts are kept on the receiver.
    all_ancestors = upstream_paths(edges, structures, receivers, allowed,
                                   candidates=max(1, len(names))) if rewired else paths
    physical = structures.set_index("station")
    physical.index = physical.index.astype(str)
    lookup = {name: i for i, name in enumerate(names)}
    n, t, c = len(receivers), residual.shape[1], candidates
    owner = np.full((n, c), -1, np.int64)
    features = np.zeros((n, c, PATH_DIM))
    values, valid = np.zeros((n, t, c, len(LAGS))), np.zeros((n, t, c, len(LAGS)), bool)
    ages = np.zeros_like(values)
    rng = np.random.default_rng(seed)
    last_value, last_age = np.zeros_like(residual), np.full_like(residual, max_age+1)
    for station in range(len(names)):
        previous, age = 0., max_age+1
        for month in range(t):
            if np.isfinite(residual[station, month]):
                previous, age = residual[station, month], 0
            else:
                age += 1
            last_value[station, month], last_age[station, month] = previous, age
    for row, receiver in enumerate(receivers):
        target = physical.loc[receiver]
        ancestors = {item[0] for item in all_ancestors[row]}
        unused = sorted(allowed-{receiver}-ancestors)
        if rewired and len(unused) < len(paths[row]):
            raise ValueError("not enough eligible non-ancestor sources for matched rewiring")
        for column, (source, distance, major, storage) in enumerate(paths[row]):
            donor = physical.loc[source]
            source_area, target_area = float(donor.drainage_area_km2), float(target.drainage_area_km2)
            if not np.isfinite([source_area, target_area]).all() or min(source_area, target_area) <= 0:
                raise ValueError("positive mapped source/receiver drainage area required")
            features[row, column] = [np.log1p(distance)/np.log1p(3000),
                np.log1p(major)/np.log1p(100), min(1., storage/distance),
                min(1., source_area/target_area), float(target.largest_minor_area_share_5km),
                float(target.stream_order)/10, float(target.storage_fraction_20km),
                np.log1p(source_area)/np.log1p(1e6)]
            if rewired:
                areas = np.array([float(physical.loc[name].drainage_area_km2) for name in unused])
                selected = int(np.argmin(np.abs(np.log1p(areas)-np.log1p(source_area))+rng.uniform(0, .1, len(areas))))
                source = unused.pop(selected)
            owner[row, column] = lookup[source]
            for index, lag in enumerate(LAGS):
                months = np.arange(lag, t)
                observed_age = last_age[lookup[source], months-lag]
                usable = observed_age <= max_age
                valid[row, months, column, index] = usable
                values[row, months, column, index] = np.where(usable, last_value[lookup[source], months-lag], 0.)
                ages[row, months, column, index] = np.where(usable, observed_age, max_age+1)
    if not np.isfinite(features).all():
        raise ValueError("finite physical river attributes required")
    return {"river_owner": owner, "river_path": features, "river_values": values,
            "river_valid": valid, "river_age": ages}


def nested_path_candidates(truth, names, source_cells, folds, references, edges, structures,
                           *, candidates=20, rewired=False, seed=42):
    """Exclude the receiving fold from every donor value and its forest fit."""
    truth, cells, names = np.asarray(truth), np.asarray(source_cells), np.asarray(names, str)
    if truth.ndim != 2 or cells.ndim != 1 or cells.dtype.kind not in 'iu' or not len(cells):
        raise ValueError("aligned truth and integer permitted source cells required")
    t, ids = truth.shape[1], np.unique(cells//truth.shape[1])
    folds = [np.asarray(fold, np.int64) for fold in folds]
    if len(folds) < 3 or not np.array_equal(np.sort(np.concatenate(folds)), ids):
        raise ValueError("folds must partition permitted source stations")
    output, records = None, []
    for a, query in enumerate(folds):
        donor_cells = cells[~np.isin(cells//t, query)]
        oof = np.full(truth.shape, np.nan)
        for b, donor in enumerate(folds):
            if a == b:
                continue
            saved = references[tuple(sorted((a, b)))]
            hidden = set(query.tolist()+donor.tolist())
            expected = cells[np.isin(cells//t, list(hidden))]
            if (set(saved["hidden_stations"]) != hidden or set(saved["fitted_stations"]) != set(ids)-hidden
                    or not np.array_equal(saved["cells"], expected)
                    or np.shape(saved["pred_z"]) != expected.shape):
                raise ValueError("river source reference must exclude query AND donor folds")
            take = np.isin(expected//t, donor)
            oof.ravel()[expected[take]] = np.asarray(saved["pred_z"])[take]
        donor_ids, residual = relative_source_residual_grid(truth, oof, donor_cells)
        bank = np.full((len(ids), t), np.nan)
        bank[np.searchsorted(ids, donor_ids)] = residual
        arrays = path_candidates(edges, structures, names[query], names[ids], bank,
            eligible=names[donor_ids], candidates=candidates, rewired=rewired, seed=seed+a)
        if output is None:
            output = {key: np.zeros((len(ids), *value.shape[1:]), dtype=value.dtype) for key, value in arrays.items()}
        for key in output:
            output[key][np.searchsorted(ids, query)] = arrays[key]
        records.append({"query_fold": a, "query_station_ids": query.tolist(),
            "library_station_ids": donor_ids.tolist(), "query_labels_in_library_or_reference": False})
    return output, records


class _RiverMessage(nn.Module):
    """Sparse source by lag attention with conveyance/mixing/storage readouts."""

    def __init__(self, hidden, ecology, dtype, seed, mode):
        super().__init__()
        self.mode = mode
        with torch.random.fork_rng():
            torch.manual_seed(seed+2719)
            self.query = nn.Linear(hidden+ecology+8, 32, bias=False, dtype=dtype)
            self.encoder = nn.Sequential(nn.Linear(PATH_DIM+8+1+3, 32, dtype=dtype), nn.Tanh())
            self.key = nn.Linear(32, 32, bias=False, dtype=dtype)
            self.gate = nn.Linear(32, 3, dtype=dtype)
            self.gain = nn.Linear(32, 3, dtype=dtype)
            self.output = nn.Linear(3, 1, bias=False, dtype=dtype)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, path, hydro, values, valid, age):
        b, c, l = values.shape
        lag = torch.eye(l, dtype=values.dtype)[None, None].expand(b, c, -1, -1)
        physical = path[:, :, None].expand(-1, -1, l, -1)
        if self.mode == "simple":
            physical = physical*0  # identical capacity, no path conditioning
        encoded = self.encoder(torch.cat([physical, hydro, age[..., None]/13, lag], -1))
        scores = torch.einsum('bd,bcld->bcl', self.query(query), self.key(encoded))/math.sqrt(32)
        weights = torch.softmax(torch.cat([scores.reshape(b, -1).masked_fill(~valid.reshape(b, -1), -torch.inf),
            torch.zeros((b, 1), dtype=values.dtype)], -1), -1)
        allocated = weights[:, :-1].reshape(b, c, l)
        gate = torch.softmax(self.gate(encoded), -1)
        area = physical[..., 3].clamp_min(0).sqrt()
        distance = physical[..., 0]
        storage = physical[..., 2]
        components = torch.stack([values*torch.exp(-distance), values*(.5+.5*area),
                                  values*torch.exp(-age/12)*(1+storage)], -1)
        if self.mode == "simple":
            components = values[..., None].expand(-1, -1, -1, 3)
        state = (allocated[..., None]*gate*components*(1+torch.tanh(self.gain(encoded)))).sum((1, 2))
        delta = self.output(state).squeeze(-1)
        if self.mode == "none":
            delta = delta*0
        return delta, {"river_prior_mass": weights[:, -1],
            "river_entropy": -(weights*weights.clamp_min(1e-30).log()).sum(-1),
            "river_lag_mass": allocated.sum(1), "river_support": valid.sum((1, 2)),
            "river_component_state": state}


class RiverStructureResidual(AvailableSourceAttentionResidual):
    """Jointly train the original local/retrieval paths plus a river message."""

    def __init__(self, spatial, temporal, decay, *, river_mode="structure", **kwargs):
        if river_mode not in ("none", "simple", "structure"):
            raise ValueError("river_mode must be none, simple or structure")
        super().__init__(spatial, temporal, decay, **kwargs)
        self.river_config = {"river_mode": river_mode}
        self.head.river = _RiverMessage(self.hidden_size, self.spatial_architecture["env_dim"],
                                      self.dtype, self.seed, river_mode)
        self._initial_attention_head = copy.deepcopy(self.head.state_dict())
        self.river_parameter_count_ = sum(p.numel() for p in self.head.river.parameters())
        self.trainable_parameter_count_ += self.river_parameter_count_

    def _prepare_inputs(self, inputs):
        arrays = super()._prepare_inputs(inputs)
        n, t = arrays["age"].shape
        owner, valid = np.asarray(inputs["river_owner"]), np.asarray(inputs["river_valid"])
        if owner.dtype.kind not in 'iu' or valid.dtype != np.bool_ or owner.ndim != 2 or owner.shape[0] != n:
            raise ValueError("integer river owners and boolean valid masks must align with receivers")
        c = owner.shape[1]
        if c < 1 or (owner < -1).any() or (owner >= arrays["donor_hydro_bank"].shape[0]).any():
            raise ValueError("river owner outside permitted source bank")
        arrays["river_owner"] = torch.as_tensor(owner, dtype=torch.long)
        arrays["river_valid"] = torch.as_tensor(valid)
        for key, shape in (("river_path", (n, c, PATH_DIM)),
                           ("river_values", (n, t, c, len(LAGS))), ("river_age", (n, t, c, len(LAGS)))):
            arrays[key] = torch.as_tensor(inputs[key], dtype=self.dtype)
            if arrays[key].shape != shape or not torch.isfinite(arrays[key]).all():
                raise ValueError("finite aligned river path/value/age arrays required")
        if (valid.shape != (n, t, c, len(LAGS)) or (valid & (owner[:, None, :, None] == -1)).any()
                or torch.count_nonzero(arrays["river_values"][~arrays["river_valid"]])
                or (arrays["river_age"] < 0).any()):
            raise ValueError("invalid river sources must have zero hidden values")
        for i, lag in enumerate(LAGS):
            if valid[:, :lag, :, i].any():
                raise ValueError("pre-calendar river lag cannot be visible")
        return arrays

    def _river_cells(self, inputs, cells, hidden):
        t = inputs["age"].shape[1]
        station, month = cells//t, cells % t
        owner = inputs["river_owner"][station]
        dates = month[:, None, None]-torch.tensor(LAGS)[None, None]
        hydro = inputs["donor_hydro_bank"][owner.clamp_min(0)[:, :, None], dates.clamp_min(0)]
        hydro = hydro*(owner >= 0)[:, :, None, None]*(dates >= 0)[..., None]
        start = self.attention_config["daily_start"]
        query = torch.cat([hidden, inputs["env"][station], inputs["extra"][station, month, start:start+8]], -1)
        delta, diagnostic = self.head.river(query, inputs["river_path"][station], hydro,
            inputs["river_values"][station, month], inputs["river_valid"][station, month],
            inputs["river_age"][station, month])
        return delta*(1+inputs["attention_reference"][station, month]), diagnostic

    def _delta_cells(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        hidden = self._hidden_cells(inputs, cells)
        t = inputs["age"].shape[1]
        extra = inputs["extra"][cells//t, cells % t]
        pieces = [hidden, extra]
        if self.interaction_indices:
            pieces.append((hidden.unsqueeze(-1)*extra[:, self.interaction_indices].unsqueeze(1)).flatten(1))
        state, _ = self._attention_cells(inputs, cells, hidden)
        river, _ = self._river_cells(inputs, cells, hidden)
        delta = (self.head.linear(torch.cat(pieces, -1)).squeeze(-1)+self.head.output(state).squeeze(-1)+river).double()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite river correction")
        return delta

    def river_diagnostics(self, inputs, cells):
        arrays = self._prepare_inputs(inputs)
        cells = np.asarray(cells)
        n, t = arrays["age"].shape
        if cells.ndim != 1 or cells.dtype.kind not in 'iu' or (cells < 0).any() or (cells >= n*t).any():
            raise ValueError("aligned integer diagnostic cells required")
        chunks = {}
        with torch.no_grad():
            for offset in range(0, len(cells), self.batch_size):
                selected = torch.as_tensor(cells[offset:offset+self.batch_size], dtype=torch.long)
                hidden = self._hidden_cells(arrays, selected)
                delta, diag = self._river_cells(arrays, selected, hidden)
                for key, value in {**diag, "river_delta_unscaled": delta}.items():
                    chunks.setdefault(key, []).append(value.numpy())
        return {key: np.concatenate(value) for key, value in chunks.items()}

    def to_dict(self):
        summary = super().to_dict()
        summary.update({"river_config": self.river_config, "river_parameters": self.river_parameter_count_,
                        "river_output_norm": float(self.head.river.output.weight.detach().norm())})
        summary["protocol"]["river_training"] = "joint neural fitting on station-OOF environmental base; double-held donor residuals"
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload["river_config"] = self.river_config.copy()
        return payload

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1 or payload.get("model_class") != cls.__name__:
            raise ValueError("unsupported river structure checkpoint")
        plain = copy.deepcopy(payload)
        plain["model_class"] = "AvailableSourceAttentionResidual"
        for key in ("head", "initial_attention_head"):
            plain[key] = {name: value for name, value in plain[key].items() if not name.startswith('river.')}
        base = AvailableSourceAttentionResidual.from_payload(plain)
        obj = cls(base.spatial, base.temporal, base.decay, **base._config(),
                  **base.attention_config, **payload["river_config"])
        river = obj.head.river
        obj.__dict__ = base.__dict__.copy()
        obj.river_config = payload["river_config"].copy()
        obj.head.river = river
        obj.river_parameter_count_ = sum(p.numel() for p in river.parameters())
        obj.trainable_parameter_count_ += obj.river_parameter_count_
        obj.head.load_state_dict(payload["head"])
        initial = copy.deepcopy(payload["initial_attention_head"])
        if (set(initial) != set(obj.head.state_dict()) or not all(torch.isfinite(v).all() for v in initial.values())
                or torch.count_nonzero(initial["river.output.weight"])):
            raise ValueError("finite initial river state with zero projection required")
        if not all(torch.isfinite(p).all() for p in obj.head.parameters()):
            raise ValueError("nonfinite river checkpoint")
        obj._initial_attention_head = initial
        return obj
