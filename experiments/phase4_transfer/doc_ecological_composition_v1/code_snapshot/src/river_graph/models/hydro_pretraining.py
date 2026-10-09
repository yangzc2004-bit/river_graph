"""Masked hydro reconstruction inside the existing ecological encoder/GRU."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def masked_hydro_inputs(inputs, hidden_mask):
    """Remove both hydro values and visibility; never reveal a masked target."""
    result = {name: np.asarray(value).copy() for name, value in inputs.items()}
    mask = np.asarray(hidden_mask, dtype=bool)
    if mask.shape != (*result["age"].shape, 2):
        raise ValueError("hydro masking must align with station/month/two channels")
    for channel, value_col, visibility_col in ((0, 0, 1), (1, 2, 3)):
        result["raw"][..., value_col][mask[..., channel]] = 0
        result["raw"][..., visibility_col][mask[..., channel]] = 0
    return result


def pretrain_hydro(model, inputs, hydro, hydro_valid, *, seed=42, epochs=30,
                   patience=5, batch_size=512, max_training_cells=8192, progress=None):
    """Return a copied pretrained encoder/GRU and a source-only reconstruction trace.

    The caller supplies source-station views only. All reserved validation hydro
    targets remain masked during training. Daily head descriptors are not consumed
    by ``_hidden_cells``. The original DOC head and supplied model are unchanged.
    """
    pretrained = copy.deepcopy(model)
    target = np.asarray(hydro, dtype=np.float64).copy()
    valid = np.asarray(hydro_valid, dtype=bool)
    if target.shape != (*np.asarray(inputs["age"]).shape, 2) or valid.shape != target.shape:
        raise ValueError("source hydro and views must align")
    if not np.isfinite(target[valid]).all():
        raise ValueError("visible source hydro must be finite")
    target[..., 1] = np.log1p(np.maximum(target[..., 1], 0))
    rng = np.random.default_rng(seed)
    validation_mask = valid & (rng.random(valid.shape) < .2)
    training_valid = valid & ~validation_mask
    if not validation_mask.any() or not training_valid.any():
        raise ValueError("source hydro needs masked training and validation targets")
    mean, scale = np.zeros(2), np.ones(2)
    for channel in range(2):
        values = target[..., channel][training_valid[..., channel]]
        if len(values):
            mean[channel], scale[channel] = values.mean(), max(values.std(), 1e-6)
    normalized = np.where(valid, (target-mean)/scale, 0)
    targets = torch.as_tensor(normalized, dtype=pretrained.dtype)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = nn.Linear(pretrained.hidden_size, 2, dtype=pretrained.dtype)
    modules = (pretrained.spatial, pretrained.temporal, pretrained.decay, head)
    encoder = [parameter for parameter in pretrained.spatial.parameters() if parameter.requires_grad]
    backbone = [parameter for module in (pretrained.temporal, pretrained.decay)
                for parameter in module.parameters() if parameter.requires_grad]
    parameters = encoder+backbone+list(head.parameters())
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if encoder:
        groups.append({"params": encoder, "lr": pretrained.encoder_learning_rate})
    if backbone:
        groups.append({"params": backbone, "lr": pretrained.learning_rate})
    optimizer = torch.optim.Adam(groups)
    months = target.shape[1]
    val_cells = np.flatnonzero(validation_mask.any(-1))
    if len(val_cells) > 4096:
        val_cells = np.sort(rng.choice(val_cells, 4096, replace=False))
    validation_inputs = pretrained._prepare_inputs(masked_hydro_inputs(inputs, validation_mask))
    val_mask_t = torch.as_tensor(validation_mask, dtype=torch.bool)
    best, state, trace, stale, best_epoch = np.inf, None, [], 0, 0

    def evaluate(epoch, training_loss):
        nonlocal best, state, stale, best_epoch
        total, count = 0., 0
        with torch.no_grad():
            for start in range(0, len(val_cells), batch_size):
                cells = val_cells[start:start+batch_size]
                prediction = head(pretrained._hidden_cells(validation_inputs, cells))
                rows, dates = cells//months, cells % months
                mask = val_mask_t[rows, dates]
                errors = (prediction-targets[rows, dates]).square()[mask]
                total += float(errors.sum())
                count += int(mask.sum())
        score = total/count
        if not np.isfinite(score):
            raise FloatingPointError("nonfinite hydro reconstruction")
        if score < best:
            best, state, stale, best_epoch = score, [copy.deepcopy(module.state_dict()) for module in modules], 0, epoch
        else:
            stale += 1
        row = {"epoch": epoch, "training_mse": training_loss, "validation_mse": score, "best_epoch": best_epoch}
        trace.append(row)
        if progress:
            progress(row)

    evaluate(0, None)
    for epoch in range(1, epochs+1):
        random = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        hidden = training_valid & (random.random(valid.shape) < .2)
        cells = np.flatnonzero(hidden.any(-1))
        random.shuffle(cells)
        cells = cells[:max_training_cells]
        if not len(cells):
            break
        training = pretrained._prepare_inputs(masked_hydro_inputs(inputs, hidden | validation_mask))
        selected_mask = torch.as_tensor(hidden, dtype=torch.bool)
        total, count = 0., 0
        for start in range(0, len(cells), batch_size):
            batch = cells[start:start+batch_size]
            rows, dates = batch//months, batch % months
            optimizer.zero_grad()
            prediction = head(pretrained._hidden_cells(training, batch))
            errors = (prediction-targets[rows, dates]).square()[selected_mask[rows, dates]]
            loss = errors.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(errors.detach().sum())
            count += len(errors)
        evaluate(epoch, total/count)
        if stale >= patience:
            break
    for module, saved in zip(modules, state):
        module.load_state_dict(saved)
    pretrained.spatial.eval()
    return pretrained, {"selection_role": "masked_source_hydro", "best_epoch": best_epoch,
                        "epochs_run": len(trace)-1, "validation_mse": best,
                        "target_mean": mean.tolist(), "target_scale": scale.tolist(),
                        "mask_fraction": .2, "validation_channels": int(validation_mask.sum()),
                        "source_stations": len(target), "trace": trace}
