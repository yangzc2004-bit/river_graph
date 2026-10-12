"""Matched input/loss experiment within the directional RiverDOCNet backbone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from river_graph.models.river_relation_comparison import RiverRelationModel

INPUTS = ("monthly", "daily")
LOSSES = ("log", "joint")


class DynamicRiverDOC(RiverRelationModel):
    """Same MLP, GRU width, directional heads and readout; eight extra channels.

    Original-input controls receive eight zero channels. All four neural arms
    have identical tensors and seeded initialization. Added channels are daily
    discharge descriptors, not DOC or station identity information.
    """

    def __init__(self, **kwargs):
        super().__init__("direction_heads", **kwargs)
        old = self.temporal
        with torch.random.fork_rng(devices=[]):
            expanded = nn.GRU(16, old.hidden_size, batch_first=True)
        with torch.no_grad():
            expanded.weight_ih_l0[:, :8].copy_(old.weight_ih_l0)
            expanded.weight_hh_l0.copy_(old.weight_hh_l0)
            expanded.bias_ih_l0.copy_(old.bias_ih_l0)
            expanded.bias_hh_l0.copy_(old.bias_hh_l0)
        self.temporal = expanded


def dynamic_inputs(original, daily, view):
    if view not in INPUTS:
        raise ValueError("undeclared dynamic input view")
    daily = np.asarray(daily, dtype=np.float32)
    times, stations, lookback, channels = original["windows"].shape
    if daily.shape != (stations, times, 8) or channels != 8:
        raise ValueError("aligned station/month daily descriptors required")
    if not np.isfinite(daily).all() or np.any((daily < 0) | (daily > 1)):
        raise ValueError("bounded daily flow descriptors and availability required")
    extra = daily if view == "daily" else np.zeros_like(daily)
    padded = np.pad(extra, ((0, 0), (lookback-1, 0), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=1)
    windows = np.ascontiguousarray(windows.transpose(1, 0, 3, 2))
    return {**original, "windows": torch.cat([original["windows"], torch.from_numpy(windows)], -1)}


def label_scales(train_y):
    valid = torch.isfinite(train_y)
    values = train_y[valid]
    if not len(values):
        raise ValueError("training DOC required")
    adjacent = valid[1:] & valid[:-1]
    deltas = (train_y[1:]-train_y[:-1])[adjacent]
    return {"log_mean": float(torch.log1p(values).mean()),
            "log_std": max(float(torch.log1p(values).std()), .1),
            "native_std": max(float(values.std()), 1.),
            "delta_rms": max(float(deltas.square().mean().sqrt()), .5) if len(deltas) else .5,
            "training_cells": int(valid.sum()), "training_adjacent_pairs": int(adjacent.sum())}


def reconstruction_loss(z, truth, previous_z, previous_truth, scales, denominators, loss):
    """Only finite training-role truth contributes; no query labels are inputs."""
    if loss not in LOSSES:
        raise ValueError("undeclared objective")
    valid = torch.isfinite(truth)
    target = (torch.log1p(truth[valid])-scales["log_mean"])/scales["log_std"]
    log_loss = (z[valid]-target).square().sum()/denominators["cells"]
    if loss == "log":
        return log_loss
    native = torch.expm1(torch.clamp(z*scales["log_std"]+scales["log_mean"], 0., 15.))
    native_loss = ((native[valid]-truth[valid])/scales["native_std"]).square().sum()/denominators["cells"]
    pair = valid & torch.isfinite(previous_truth)
    if pair.any():
        previous = torch.expm1(torch.clamp(previous_z*scales["log_std"]+scales["log_mean"], 0., 15.))
        delta_error = ((native[pair]-previous[pair])-(truth[pair]-previous_truth[pair]))/scales["delta_rms"]
        delta_loss = delta_error.square().sum()/denominators["pairs"]
    else:
        delta_loss = z.sum()*0
    return log_loss+.25*native_loss+.25*delta_loss


def change_metrics(frame):
    """Supplementary diagnostics; demeaning is evaluation-only."""
    y, p = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    local = pd.DataFrame({"station": frame.station.to_numpy(), "y": y, "p": p})
    centered = local[["y", "p"]]-local.groupby("station")[["y", "p"]].transform("mean")
    a, b = centered.y.to_numpy(), centered.p.to_numpy()
    within = float(np.square(a).sum())
    log = pd.DataFrame({"station": local.station, "y": np.log1p(y), "p": np.log1p(p)})
    log_center = log[["y", "p"]]-log.groupby("station")[["y", "p"]].transform("mean")
    log_within = float(np.square(log_center.y).sum())
    ordered = frame.sort_values(["station", "month"])
    delta = ordered.groupby("station")[["month", "y_true", "y_pred"]].diff()
    delta = delta[delta.month == 1]
    correlation = lambda x,z: float(np.corrcoef(x,z)[0,1]) if len(x)>2 and np.std(x)>1e-6 and np.std(z)>1e-6 else None
    return {"station_centered_r2": 1-float(np.square(b-a).sum())/within if within else None,
            "log1p_station_centered_r2": 1-float(np.square(log_center.p-log_center.y).sum())/log_within if log_within else None,
            "station_centered_correlation": correlation(a,b),
            "within_station_sd_ratio": float(np.sqrt(np.square(b).sum()/within)) if within else None,
            "monthly_delta_mae": float(np.abs(delta.y_pred-delta.y_true).mean()) if len(delta) else None,
            "monthly_delta_correlation": correlation(delta.y_true,delta.y_pred),
            "zero_change_delta_mae": float(np.abs(delta.y_true).mean()) if len(delta) else None,
            "monthly_pairs": len(delta)}
