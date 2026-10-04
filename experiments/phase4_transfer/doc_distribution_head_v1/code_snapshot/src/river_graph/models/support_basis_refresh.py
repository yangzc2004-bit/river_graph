"""Read the updated expert through the existing two-dimensional support head.

This is inference, not an additional learned adapter. The selected neural
weights, source-derived readout and historical calendar-anchor definition stay
fixed. Raw projected states are causal within the expert's input window;
station anchor normalization is deliberately retrospective, as in v4.
"""

from __future__ import annotations

import numpy as np
import torch

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import (
    EpisodicTemporalAdapter,
    anchor_normalize,
)


def refresh_support_basis(
    model: EncoderNativeResidual,
    inputs: dict,
    temporal_adapter: EpisodicTemporalAdapter,
    *,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Project selected expert states with an unchanged historical readout.

    ``inputs`` uses the expert's existing raw/env/age/support/extra schema and,
    for active modes, daily_history. The caller supplies the correct visibility
    view; this function never reads labels, creates a split, or fits statistics.
    Input station/month order is preserved. Source-fold views may be passed in
    exactly the same way as full-cohort views.

    Returns float64 ``basis[N,T,2]`` and ``raw_basis[N,T,2]``, integer
    ``anchor_months[A]``, float64 ``station_anchor_mean[N,2]``,
    ``station_rms[N]`` and ``station_scale[N]``, plus bool ``floor_hit[N]``.
    The scale equals sqrt(max(anchor_variance, scale_floor**2)), matching
    :func:`anchor_normalize` exactly. It is one scalar per station, not separate
    axis whitening. The raw state is computed in bounded station-major batches;
    only the two projected channels are retained for the full grid.

    No parameters, gradients, training flags or random-generator state change.
    No new normalization is learned from labels. Calendar anchors can include
    months after a query, so only ``raw_basis`` has causal-prefix invariance.
    """
    if not isinstance(model, EncoderNativeResidual):
        raise TypeError("model must be an EncoderNativeResidual")
    if not isinstance(temporal_adapter, EpisodicTemporalAdapter):
        raise TypeError("temporal_adapter must be the existing EpisodicTemporalAdapter")
    if (not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool)
            or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    readout = temporal_adapter.readout.detach().cpu()
    if (readout.dtype != torch.float64 or readout.shape != (model.hidden_size, 2)
            or not torch.isfinite(readout).all()):
        raise ValueError("saved readout must be finite float64 [expert hidden_size, 2]")
    if not np.isfinite(temporal_adapter.scale_floor) or temporal_adapter.scale_floor <= 0:
        raise ValueError("saved anchor scale floor must be finite and positive")
    prepared = model._prepare_inputs(inputs)
    n, months = prepared["age"].shape
    anchors = temporal_adapter.anchor_months(months)
    roots = (model.spatial, model.temporal, model.decay, model.head, model.hydro_projection)
    modules = {id(module): module for root in roots if root is not None for module in root.modules()}
    training_flags = [(module, module.training) for module in modules.values()]
    try:
        for root in roots:
            if root is not None:
                root.eval()
        with torch.inference_mode():
            raw = torch.empty((n * months, 2), dtype=torch.float64)
            for start in range(0, n * months, int(batch_size)):
                end = min(start + int(batch_size), n * months)
                cells = torch.arange(start, end, dtype=torch.long)
                hidden = model._hidden_cells(prepared, cells)
                if hidden.shape != (end - start, model.hidden_size) or not torch.isfinite(hidden).all():
                    raise FloatingPointError("expert produced invalid hidden states")
                raw[start:end] = hidden.double() @ readout
            raw = raw.reshape(n, months, 2)
            if not torch.isfinite(raw).all():
                raise FloatingPointError("nonfinite projected support basis")
            basis, stats = anchor_normalize(
                raw, torch.as_tensor(anchors), scale_floor=temporal_adapter.scale_floor,
                return_stats=True)
            anchor_values = raw[:, anchors]
            mean = anchor_values.mean(dim=1, keepdim=True)
            variance = (anchor_values - mean).square().mean(dim=(1, 2), keepdim=True)
            scale = variance.clamp_min(temporal_adapter.scale_floor**2).sqrt()
            return {
                "basis": basis.numpy().copy(),
                "raw_basis": raw.numpy().copy(),
                "anchor_months": anchors.copy(),
                "station_anchor_mean": mean[:, 0].numpy().copy(),
                "station_rms": stats["station_rms"].copy(),
                "station_scale": scale.reshape(-1).numpy().copy(),
                "floor_hit": stats["floor_hit"].copy(),
            }
    finally:
        # Set individual flags rather than recursively train(), preserving
        # callers' deliberately mixed train/eval state within the encoder.
        for module, training in training_flags:
            module.training = training
