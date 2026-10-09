"""Source-only water-quality supervision for the existing ecological GRU.

Auxiliary labels are targets, never input features. The copied backbone and
disposable heads share station-blocked auxiliary validation; the ordinary DOC
model is subsequently constructed from the warm backbone without these heads.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from river_graph.models.episodic_temporal_adapter import _parameter_distance


def source_auxiliary_targets(auxiliary, source_ids, site_no, months):
    """Slice allowed source rows before reading target values or availability."""
    source_ids = np.asarray(source_ids, dtype=np.int64)
    shape = (len(site_no), len(months))
    if (source_ids.ndim != 1 or not len(source_ids)
            or len(np.unique(source_ids)) != len(source_ids)
            or (source_ids < 0).any() or (source_ids >= shape[0]).any()):
        raise ValueError("source station identities must be unique valid indices")
    values, masks = [], []
    for name in ("ph", "spec_conductance"):
        dataset = auxiliary[name]
        for key, expected in (("site_no", site_no), ("months", months)):
            np.testing.assert_array_equal(dataset[key], expected)
        if tuple(dataset["y"].shape) != shape or tuple(dataset["y_mask"].shape) != shape:
            raise ValueError("auxiliary station-month grids must align with DOC")
        # Index first: receiving labels, including availability, are not examined.
        target = np.asarray(dataset["y"][source_ids], dtype=np.float64)
        valid = np.asarray(dataset["y_mask"][source_ids], dtype=bool)
        if not np.isfinite(target[valid]).all():
            raise ValueError("available source auxiliary labels must be finite")
        if name == "spec_conductance" and (target[valid] < 0).any():
            raise ValueError("valid source conductance must be nonnegative")
        safe = np.zeros_like(target)
        safe[valid] = np.log1p(target[valid]) if name == "spec_conductance" else target[valid]
        values.append(safe)
        masks.append(valid)
    return np.stack(values, axis=-1), np.stack(masks, axis=-1)


def prepare_auxiliary_labels(target, valid, validation_stations, *, shuffle=False, seed=42):
    """Fit scalers on auxiliary fit stations; shuffle their labels only."""
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    held = np.asarray(validation_stations, dtype=bool)
    if (target.ndim != 3 or target.shape[-1] != 2 or valid.shape != target.shape
            or held.shape != (target.shape[0],) or not held.any() or held.all()
            or not np.isfinite(target[valid]).all()):
        raise ValueError("aligned source targets need nonempty fit/held stations")
    fit_mask = valid & ~held[:, None, None]
    val_mask = valid & held[:, None, None]
    mean, scale = np.zeros(2), np.ones(2)
    normalized = np.zeros_like(target)
    random = np.random.default_rng(seed)
    for channel in range(2):
        fit = target[..., channel][fit_mask[..., channel]]
        if not len(fit) or not val_mask[..., channel].any():
            raise ValueError("both auxiliary targets need fit and validation labels")
        mean[channel], scale[channel] = fit.mean(), max(fit.std(), 1e-6)
        normalized[..., channel][valid[..., channel]] = (target[..., channel][valid[..., channel]]-mean[channel])/scale[channel]
        if shuffle:
            normalized[..., channel][fit_mask[..., channel]] = random.permutation(normalized[..., channel][fit_mask[..., channel]])
    return normalized, fit_mask, val_mask, mean, scale


def pretrain_source_auxiliary(model, inputs, target, valid, validation_stations, *,
                             shuffle=False, seed=42, epochs=30, batch_size=512, progress=None):
    """Train all source fit cells for a fixed budget, select on held source sites.

    Losses give the two indicators equal weight despite different availability.
    No DOC checkpoint, receiving label or auxiliary value enters the input view.
    The supplied DOC model is unchanged. Its copied trainable backbone uses the
    same parameter selection and learning rates as ordinary DOC residual fitting.
    """
    if epochs < 0 or batch_size < 1:
        raise ValueError("epochs must be nonnegative and batch size positive")
    warm = copy.deepcopy(model)
    data = warm._prepare_inputs(inputs)
    if np.asarray(target).shape != (*data["age"].shape, 2):
        raise ValueError("source auxiliary labels must align with input views")
    normalized, fit_mask, val_mask, mean, scale = prepare_auxiliary_labels(
        target, valid, validation_stations, shuffle=shuffle, seed=seed)
    labels = torch.as_tensor(normalized, dtype=warm.dtype)
    fit_t, val_t = torch.as_tensor(fit_mask), torch.as_tensor(val_mask)
    months = normalized.shape[1]
    fit_cells, val_cells = np.flatnonzero(fit_mask.any(-1)), np.flatnonzero(val_mask.any(-1))
    fit_counts, val_counts = fit_mask.sum((0, 1)), val_mask.sum((0, 1))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = nn.Linear(warm.hidden_size, 2, dtype=warm.dtype)
    modules = (warm.spatial, warm.temporal, warm.decay, head)
    before = [copy.deepcopy(module.state_dict()) for module in modules]
    encoder = [p for p in warm.spatial.parameters() if p.requires_grad]
    backbone = [p for module in (warm.temporal, warm.decay) for p in module.parameters() if p.requires_grad]
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if encoder:
        groups.append({"params": encoder, "lr": warm.encoder_learning_rate})
    if backbone:
        groups.append({"params": backbone, "lr": warm.learning_rate})
    parameters = encoder+backbone+list(head.parameters())
    optimizer = torch.optim.Adam(groups)
    weights = torch.as_tensor(len(fit_cells)/(2*fit_counts), dtype=warm.dtype)
    best, best_epoch, saved, trace, steps = np.inf, 0, None, [], 0

    def evaluate(epoch, training_loss):
        nonlocal best, best_epoch, saved
        total = np.zeros(2)
        with torch.no_grad():
            for start in range(0, len(val_cells), batch_size):
                cells = val_cells[start:start+batch_size]
                rows, dates = cells//months, cells % months
                errors = (head(warm._hidden_cells(data, cells))-labels[rows, dates]).square()
                total += (errors*val_t[rows, dates]).sum(0).double().numpy()
        per_target = total/val_counts
        score = float(per_target.mean())
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite auxiliary validation loss")
        if score < best:
            best, best_epoch = score, epoch
            saved = [copy.deepcopy(module.state_dict()) for module in modules]
        row = {"epoch": epoch, "training_mse": training_loss, "validation_mse": score,
               "validation_ph_mse": float(per_target[0]), "validation_log_ec_mse": float(per_target[1]),
               "best_epoch": best_epoch, "best_validation_mse": best}
        trace.append(row)
        if progress:
            progress(copy.deepcopy(row))

    evaluate(0, None)
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(fit_cells)
        total = np.zeros(2)
        for start in range(0, len(order), batch_size):
            cells = order[start:start+batch_size]
            rows, dates = cells//months, cells % months
            optimizer.zero_grad()
            errors = (head(warm._hidden_cells(data, cells))-labels[rows, dates]).square()*fit_t[rows, dates]
            loss = (errors*weights).sum()/len(cells)
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("nonfinite source auxiliary training")
            nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            total += errors.detach().sum(0).double().numpy()
            steps += 1
        evaluate(epoch, float((total/fit_counts).mean()))
    for module, state in zip(modules, saved, strict=True):
        module.load_state_dict(state)
    warm.spatial.eval()
    summary = {"selection_role": "internal_source_station_fold", "auxiliary_targets": ["ph", "log1p_spec_conductance"],
        "source_label_shuffle": bool(shuffle), "seed": seed, "best_epoch": best_epoch,
        "epochs_run": epochs, "optimizer_steps": steps, "validation_mse": best,
        "target_mean": mean.tolist(), "target_scale": scale.tolist(),
        "fit_label_counts": fit_counts.tolist(), "validation_label_counts": val_counts.tolist(),
        "validation_station_positions": np.flatnonzero(validation_stations).tolist(),
        "source_stations": len(target), "receiving_chemistry_used": False,
        "spatial_parameter_distance": _parameter_distance(warm.spatial, before[0]),
        "temporal_parameter_distance": _parameter_distance(warm.temporal, before[1]),
        "decay_parameter_distance": _parameter_distance(warm.decay, before[2]), "trace": trace}
    return warm, summary, {"spatial": saved[0], "temporal": saved[1], "decay": saved[2], "auxiliary_head": saved[3]}
