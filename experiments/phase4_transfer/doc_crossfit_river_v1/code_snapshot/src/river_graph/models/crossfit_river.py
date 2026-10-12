"""Controlled whole-station cross-fitting of a new DOC local/river backbone.

No legacy supervised weights or DOC-valued input channels are consumed. The
same local fit is used at deployment by both correction-training strategies.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from river_graph.models.dynamic_river_state import (
    DynamicRiverState,
    hydro_history_valid,
)
from river_graph.models.river_structure_residual import LAGS, upstream_paths

RELATIONS = ("directed", "generic", "nonconnected")
WAYS = ("fitted", "crossfit")


class LocalEnvironmentGRU(nn.Module):
    """Ecological MLP + causal GRU correction to covariate-only ExtraTrees."""

    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.environment = nn.Sequential(nn.Linear(15, 24), nn.Tanh())
            self.temporal = nn.GRU(14, 32, batch_first=True)
            self.readout = nn.Linear(56, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, windows, environment):
        _, hidden = self.temporal(windows)
        states = torch.cat([self.environment(environment), hidden[-1]], -1)
        return self.readout(states).squeeze(-1), states


def covariates(view):
    """Only measured hydro, masks, calendar, ecology and daily descriptors."""
    raw = np.asarray(view["raw"], np.float32)
    daily = np.asarray(view["extra"][..., 30:38], np.float32)
    env = np.concatenate([raw[:, 0, 6:8], raw[:, 0, 10:14], view["env"]], -1)
    temporal = np.concatenate([raw[..., :6], daily], -1).astype(np.float32)
    n, t = raw.shape[:2]
    forest = np.concatenate([np.broadcast_to(env[:, None], (n, t, 15)), temporal], -1)
    available = hydro_history_valid(raw, daily)
    return {"env": env, "temporal": temporal, "forest": forest.reshape(n*t, -1),
            "daily": daily, "hydro": raw[..., :4], "available": available}


def local_batch(inputs, cells):
    t = inputs["temporal"].shape[1]
    row, month = np.asarray(cells)//t, np.asarray(cells) % t
    dates = month[:, None]-np.arange(11, -1, -1)[None]
    windows = inputs["temporal"][row[:, None], dates.clip(min=0)].copy()
    windows[dates < 0] = 0
    return torch.from_numpy(windows), torch.from_numpy(inputs["env"][row])


def relation_banks(edges, physical, names, receivers, eligible, available, seed):
    """Matched 10 upstream + 10 downstream slots; all controls label-free.

    Nonconnected replacement donors have no directed path in either direction,
    even beyond the primary 3000-km cap, and no receiving COMID alias. Generic
    attention chooses from all eligible donors without checking connectivity.
    Only directed banks use real path attributes; null banks retain the same
    pseudo-path slots as a donor-identity negative control.
    """
    names, receivers, eligible = np.asarray(names, str), np.asarray(receivers), np.asarray(eligible)
    rev = edges.rename(columns={"source": "target", "target": "source"})
    target_names, bank_names = names[receivers], names[eligible]
    up = upstream_paths(edges, physical, target_names, bank_names, candidates=10)
    down = upstream_paths(rev, physical, target_names, bank_names, candidates=10)
    all_up = upstream_paths(edges, physical, target_names, bank_names,
                            candidates=max(1, len(eligible)), max_km=np.inf)
    all_down = upstream_paths(rev, physical, target_names, bank_names,
                              candidates=max(1, len(eligible)), max_km=np.inf)
    attrs = physical.set_index("station")
    attrs.index = attrs.index.astype(str)
    lookup = {name: i for i, name in enumerate(names)}
    areas = np.log1p(attrs.loc[names, "drainage_area_km2"].to_numpy(float))
    comids = attrs.loc[names, "comid"].to_numpy()
    owners = {arm: np.full((len(names), 20), -1, np.int64) for arm in RELATIONS}
    paths = np.zeros((len(names), 20, 8), np.float32)
    records = []
    rng = np.random.default_rng(seed)
    for j, receiver in enumerate(receivers):
        related = {lookup[x[0]] for x in (*all_up[j], *all_down[j])}
        aliases = {comids[receiver]}
        null_pool = [i for i in eligible if i not in related and comids[i] not in aliases]
        generic_pool = [i for i in eligible if comids[i] not in aliases]
        for direction, selected in ((1, up[j]), (-1, down[j])):
            for slot, (name, km, junctions, storage) in enumerate(selected):
                column = slot if direction == 1 else slot+10
                original = lookup[name]
                owners["directed"][receiver, column] = original
                paths[receiver, column] = [float(direction), np.log1p(km)/np.log1p(3000),
                    np.log1p(junctions)/np.log1p(100), min(1., storage/km),
                    np.clip(areas[original]-areas[receiver], -5, 5)/5,
                    areas[original]/np.log1p(1e6),
                    float(attrs.loc[name, "stream_order"])/10,
                    float(attrs.loc[name, "storage_fraction_20km"])]
                row = {"receiver": int(receiver), "slot": column, "direction": direction,
                       "directed": int(original), "path_km": float(km)}
                for arm, pool in (("generic", generic_pool), ("nonconnected", null_pool)):
                    if not pool:
                        raise ValueError("insufficient matched control donors")
                    similarity = [abs(areas[i]-areas[original])+2*(1-
                        (available[i] & available[original]).sum()/max(1,
                        (available[i] | available[original]).sum())) for i in pool]
                    score = np.asarray(similarity)
                    if arm == "generic":
                        score = score+rng.uniform(0, .1, len(pool))
                    selected_i = pool.pop(int(np.argmin(score)))
                    owners[arm][receiver, column] = selected_i
                    row[arm] = int(selected_i)
                records.append(row)
    # One common candidate-date support mask, determined without any DOC.
    dates = np.arange(available.shape[1])[:, None]-np.asarray(LAGS)[None]
    valid = np.ones((len(names), available.shape[1], 20, 3), bool)
    for owner in owners.values():
        valid &= (owner[:, None, :, None] >= 0) & (dates[None, :, None] >= 0) & available[
            owner.clip(min=0)[:, None, :, None], dates.clip(min=0)[None, :, None]]
    return {"owners": owners, "paths": paths, "valid": valid, "records": records}


def river_batch(inputs, bank, anchor, cells, arm):
    """Gather equal-supported messages lazily; no future or label channels."""
    cells = np.asarray(cells)
    t = inputs["temporal"].shape[1]
    row, month = cells//t, cells % t
    owners = bank["owners"][arm][row]
    dates = month[:, None, None]-np.asarray(LAGS)[None, None]
    valid = bank["valid"][row, month]
    donor = owners.clip(min=0)[..., None]
    d = dates.clip(min=0)
    states = anchor["states"][donor, d]
    hydro = inputs["daily"][donor, d]
    previous = inputs["daily"][donor, (dates-1).clip(min=0)]
    previous = np.where((dates >= 1)[..., None], previous, 0)
    current = inputs["daily"][row, month]
    path = bank["paths"][row]
    if arm == "generic":
        path = np.zeros_like(path)
    features = np.concatenate([np.broadcast_to(path[:, :, None], (*valid.shape, 8)),
        hydro, hydro-previous, hydro-current[:, None, None],
        np.zeros((*valid.shape, 1), np.float32),
        np.broadcast_to(np.eye(3, dtype=np.float32)[None, None], (*valid.shape, 3))], -1)
    logbase = anchor["logpred"].ravel()[cells]
    query = np.concatenate([anchor["states"][row, month], logbase[:, None]/5], -1)
    return {"query": torch.from_numpy(query.astype(np.float32)),
        "features": torch.from_numpy(np.where(valid[..., None], features, 0).astype(np.float32)),
        "states": torch.from_numpy(np.where(valid[..., None], states, 0).astype(np.float32)),
        "valid": torch.from_numpy(valid)}, torch.from_numpy(logbase.astype(np.float32))


def new_river(seed):
    return DynamicRiverState(84, 36, 83, seed=seed, heads=2, dimensions=32,
                             epochs=30, patience=30, allocation="dynamic")


def native(log_values):
    return torch.expm1(log_values.clamp(0, 15))
