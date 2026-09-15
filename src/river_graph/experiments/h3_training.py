"""T08/T09: one shared trainer for the ENV, H2X and H3A arms.

Why a dedicated trainer instead of GCNDocModel.fit_predict: the H3 protocol
needs (a) an ENV arm with no graph at all, (b) per-epoch re-masking and
selection that keep the internal validation roles hidden, and (c) the H3A
components exported separately.  Rewriting the loop here keeps the frozen
GCNDocModel behaviour, and the frozen gate-run identities, untouched.

What is shared across the three arms and therefore cannot differ between them:

  * the same month-wise update loop (one month = one full-graph step);
  * the same per-epoch re-masking of train cells (fresh random half visible as
    context, other half carries the loss);
  * the same visible-role arrangement, read straight from the mask;
  * the same loss (log1p MSE on the current month's train targets);
  * the same Adam settings, gradient clipping, early stopping and clipping of
    the exported prediction.

Visible roles, in one place:

  training forward   context + the visible half of train
  selection forward  context + train (+ val_context when the mask exposes it)
  validation forward context + train (+ val_context when the mask exposes it)
  never visible      val (the internal validation target), test (outer test)

val never enters an input or the loss: it can only influence model selection
through its score.  E2a simply has no val_context key, so E2a and E2b differ
only by that one arrangement and may select different best weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from river_graph.models.gcn import GCNDocModel
from river_graph.models.h3 import (
    ENV_FEATURE_NAMES,
    EnvironmentalPredictor,
    H3ResidualImputer,
    build_env_features,
    make_h2x_trunk,
    reset_linear_parameters,
)

ARMS = ("env", "h2x", "h3a", "h3a_no_message")
GRAPH_ARMS = ("h2x", "h3a", "h3a_no_message")
RESIDUAL_ARMS = ("h3a", "h3a_no_message")
DEFAULT_PROTOCOL = "configs/h3a_v1.json"


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict:
    """Read and sanity-check the frozen machine-readable protocol."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    required = (
        "protocol_version",
        "arms",
        "init",
        "training",
        "prediction",
        "metrics",
        "promotion",
        "seeds",
    )
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError("protocol is missing keys: " + ", ".join(missing))
    for arm in ("env", "h2x", "h3a"):
        if arm not in protocol["arms"]:
            raise ValueError("protocol is missing arm " + arm)
    training = protocol["training"]
    for key in (
        "optimizer",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "max_epochs",
        "patience",
        "early_stop_min_delta",
        "gradient_clip_norm",
        "remask_fraction",
        "loss",
    ):
        if key not in training:
            raise ValueError("training protocol is missing " + key)
    if training["loss"] != "log1p_mse":
        raise ValueError("only the frozen log1p_mse loss is supported")
    if protocol["prediction"]["clip_quantile"] != 0.995:
        raise ValueError("only the frozen 0.995 clip quantile is supported")
    return protocol


def restrict_to_stations(
    dataset: dict, split: dict, n_stations: int
) -> tuple[dict, dict]:
    """First-n_stations subset that keeps the flat-cell-index convention.

    The smoke stage needs a small dataset that is still described by the frozen
    h3a_v1 masks.  Restricting to a station prefix and dropping the cells that
    leave the prefix keeps every surviving flat index (row * n_months + month)
    pointed at the same station/month, so no cell needs remapping.
    """
    n, t = dataset["y"].shape
    keep = np.arange(min(int(n_stations), int(n)))
    allowed = np.zeros(n, dtype=bool)
    allowed[keep] = True

    subset: dict = {}
    for key, value in dataset.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == n:
            subset[key] = value[keep].clone()
        else:
            subset[key] = value
    subset["site_no"] = [dataset["site_no"][int(i)] for i in keep]

    if "edge_index" in dataset:
        source = dataset["edge_index"][0].numpy()
        target = dataset["edge_index"][1].numpy()
        inside = allowed[source] & allowed[target]
        remap = -np.ones(n, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        subset["edge_index"] = torch.tensor(
            np.stack([remap[source[inside]], remap[target[inside]]]), dtype=torch.long
        )
        if "edge_attr" in dataset:
            subset["edge_attr"] = dataset["edge_attr"][inside].clone()

    subset_split = {}
    for key, value in split.items():
        value = np.asarray(value)
        if value.ndim == 1 and value.dtype.kind in "iu":
            subset_split[key] = value[allowed[value // t]]
        # station-name arrays (val_sites) are not cell indices and are dropped
    return subset, subset_split


class H3Trainer:
    """Train one (arm, seed, mask) configuration under the frozen protocol."""

    def __init__(self, arm: str, seed: int, protocol: dict):
        if arm not in ARMS:
            raise ValueError("unknown arm: " + str(arm))
        self.arm = arm
        self.seed = int(seed)
        self.protocol = protocol
        self.model: torch.nn.Module | None = None
        self.training_info_ = None
        self.validation_ = None
        self.epoch_log_: list[dict] = []
        self.best_state_ = None
        self.init_info_: dict = {}

    # ---------------------------------------------------------------- inputs

    def prepare(self, dataset: dict, split: dict) -> None:
        """Build every input tensor the arms share; never mutates the dataset."""
        h2x = self.protocol["arms"]["h2x"]
        probe = GCNDocModel(
            architecture="transport_enc",
            env_encoder=h2x["env_encoder"],
            env_groups=h2x["env_groups"],
            gate_mode=h2x["gate_mode"],
        )
        (
            self.xt,
            self.y,
            base_visible,
            self.train_cells,
            self.stats,
            self.env_raw,
        ) = probe._build_inputs(dataset, split)
        self.env = build_env_features(dataset, split)
        self.edge_index = dataset["edge_index"]
        raw_edge_attr = dataset["edge_attr"]
        self.edge_attr = (raw_edge_attr - raw_edge_attr.mean(0)) / (
            raw_edge_attr.std(0) + 1e-8
        )
        self.n, self.t = self.y.shape
        self.dataset = dataset
        self.split = split

        self.val_cells = torch.as_tensor(np.asarray(split["val"], dtype=np.int64))
        self.val_context_cells = torch.as_tensor(
            np.asarray(split.get("val_context", np.empty(0, dtype=np.int64)),
                       dtype=np.int64)
        )
        self.test_cells = torch.as_tensor(np.asarray(split["test"], dtype=np.int64))
        self.context_cells = torch.as_tensor(
            np.asarray(split.get("context", np.empty(0, dtype=np.int64)),
                       dtype=np.int64)
        )

        # base_visible marks val+context; val and val_context are hidden here so
        # they can never reach an input.  test is never marked visible at all.
        self.base_visible = base_visible.clone()
        self.base_visible.reshape(-1)[self.val_cells] = False
        self.base_visible.reshape(-1)[self.val_context_cells] = False

        # selection / validation forward: context + all train + (val_context)
        self.eval_visible = self.base_visible.clone()
        self.eval_visible.reshape(-1)[self.train_cells] = True
        self.eval_visible.reshape(-1)[self.val_context_cells] = True

        # frozen-evaluation forward: every non-test observation is open
        self.full_visible = self.base_visible.clone()
        self.full_visible.reshape(-1)[self.train_cells] = True
        self.full_visible.reshape(-1)[self.val_context_cells] = True
        self.full_visible.reshape(-1)[self.val_cells] = True

    # ----------------------------------------------------------------- model

    def build_model(self) -> torch.nn.Module:
        """Construct the arm with explicit, frozen sub-seed initialisation."""
        env_cfg = self.protocol["arms"]["env"]
        h2x = self.protocol["arms"]["h2x"]
        init = self.protocol["init"]
        base_seed = self.seed + int(init["base_seed_offset"])
        correction_seed = self.seed + int(init["correction_seed_offset"])
        if self.arm == "env":
            model = EnvironmentalPredictor(
                len(ENV_FEATURE_NAMES),
                env_cfg["hidden"],
                env_cfg["layers"],
                env_cfg["dropout"],
            )
            model.reset_parameters(torch.Generator().manual_seed(base_seed))
            self.init_info_ = {
                "base_seed": base_seed,
                "correction_seed": None,
                "correction_head_zeroed": None,
            }
        else:
            trunk = make_h2x_trunk(
                self.xt.shape[-1],
                self.edge_attr.shape[1],
                hidden=h2x["hidden"],
                layers=h2x["layers"],
                dropout=h2x["dropout"],
                env_dim=0 if self.env_raw is None else self.env_raw.shape[1],
                env_emb=h2x["env_emb"],
                gate_mode=h2x["gate_mode"],
            )
            if self.arm == "h2x":
                reset_linear_parameters(
                    trunk,
                    torch.Generator().manual_seed(
                        self.seed + int(init["h2x_seed_offset"])
                    ),
                )
                model = trunk
                self.init_info_ = {
                    "base_seed": None,
                    "correction_seed": self.seed + int(init["h2x_seed_offset"]),
                    "correction_head_zeroed": False,
                }
            else:
                base = EnvironmentalPredictor(
                    len(ENV_FEATURE_NAMES),
                    env_cfg["hidden"],
                    env_cfg["layers"],
                    env_cfg["dropout"],
                )
                model = H3ResidualImputer(
                    base,
                    trunk,
                    edge_set="empty" if self.arm == "h3a_no_message" else "river",
                )
                self.init_info_ = model.initialise(base_seed, correction_seed)
                self.init_info_["edge_set"] = model.edge_set
        self.model = model
        return model

    # --------------------------------------------------------------- forward

    def forward_month(self, month: int) -> torch.Tensor:
        """Total log1p prediction for one month, shape (N,)."""
        return self.components_month(month)[2]

    def components_month(
        self, month: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(base, correction, total) in log1p space.

        A component the arm does not own is NaN, so an absent environmental
        base can never be read as a real zero contribution.
        """
        if self.arm == "env":
            total = self.model(self.env[month])
            nan = torch.full_like(total, float("nan"))
            return total, nan, total
        if self.arm == "h2x":
            total = self.model(
                self.xt[month], self.edge_index, self.edge_attr, self.env_raw
            )
            nan = torch.full_like(total, float("nan"))
            return nan, total, total
        return self.model.forward_components(
            self.env[month],
            self.xt[month],
            self.edge_index,
            self.edge_attr,
            self.env_raw,
        )

    # ---------------------------------------------------------------- fitting

    def fit(self, dataset: dict, split: dict) -> dict:
        started = time.perf_counter()
        self.prepare(dataset, split)
        self.build_model()
        training = self.protocol["training"]
        prediction_cfg = self.protocol["prediction"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training["lr"],
            betas=tuple(training["betas"]),
            eps=training["eps"],
            weight_decay=training["weight_decay"],
        )
        months_order = np.arange(self.t)

        best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
        self.epoch_log_ = []
        epochs_run = 0
        for epoch in range(training["max_epochs"]):
            epochs_run = epoch + 1
            epoch_started = time.perf_counter()
            perm = rng.permutation(self.train_cells.numpy())
            half = int(len(perm) * training["remask_fraction"])
            context_half = torch.as_tensor(perm[:half])
            target_cells = torch.as_tensor(perm[half:])
            visible = self.base_visible.clone()
            visible.reshape(-1)[context_half] = True
            self._fill_doc_channel(visible)
            tti, ttj = target_cells // self.t, target_cells % self.t

            self.model.train()
            np.random.shuffle(months_order)
            loss_sum, loss_n = 0.0, 0
            for month in months_order:
                selected = tti[ttj == month]
                if len(selected) == 0:
                    continue
                pred = self.forward_month(int(month))
                loss = F.mse_loss(pred[selected], self.y[selected, month])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training["gradient_clip_norm"]
                )
                optimizer.step()
                loss_sum += float(loss.item()) * len(selected)
                loss_n += len(selected)

            self.model.eval()
            with torch.no_grad():
                self._fill_doc_channel(self.eval_visible)
                criterion = self._mse_on(self.val_cells)
            improved = criterion < best_loss - training["early_stop_min_delta"]
            if improved:
                best_loss = criterion
                best_epoch = epochs_run
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            self.epoch_log_.append(
                {
                    "epoch": epochs_run,
                    "train_loss": loss_sum / max(loss_n, 1),
                    "train_cells": int(loss_n),
                    "val_loss": float(criterion),
                    "is_best": bool(improved),
                    "seconds": round(time.perf_counter() - epoch_started, 3),
                }
            )
            if bad >= training["patience"]:
                break

        if best_state is None:
            raise RuntimeError("no validation improvement was ever recorded")
        self.model.load_state_dict(best_state)
        self.best_state_ = {
            k: v.detach().cpu().clone() for k, v in best_state.items()
        }

        lo, hi = self.clip_bounds()
        self.model.eval()
        with torch.no_grad():
            self._fill_doc_channel(self.eval_visible)
            base, correction, total = self._month_vectors(self.val_cells)
        clipped = total.clamp(min=lo, max=hi)
        self.validation_ = {
            "cells": self.val_cells.numpy().copy(),
            "y_true": dataset["y"].reshape(-1)[self.val_cells].numpy().copy(),
            "y_true_log": self.y.reshape(-1)[self.val_cells].numpy().copy(),
            "base_log": base.numpy().copy(),
            "correction_log": correction.numpy().copy(),
            "total_log": total.numpy().copy(),
            "pred_log_clipped": clipped.numpy().copy(),
            "y_pred": np.expm1(clipped.numpy()).copy(),
            "clipped": (clipped != total).numpy().copy(),
            "clip_log_bounds": [float(lo), float(hi)],
        }
        self.training_info_ = {
            "arm": self.arm,
            "seed": self.seed,
            "best_epoch": best_epoch,
            "epochs": epochs_run,
            "best_val_mse_raw": float(best_loss),
            "final_train_loss": self.epoch_log_[-1]["train_loss"],
            "parameter_count": int(
                sum(p.numel() for p in self.model.parameters())
            ),
            "input_channels": int(self.xt.shape[-1]),
            "env_dim": 0 if self.env_raw is None else int(self.env_raw.shape[1]),
            "edge_dim": int(self.edge_attr.shape[1]),
            "env_features": len(ENV_FEATURE_NAMES),
            "n_edges": int(self.edge_index.shape[1]),
            "clip_log_bounds": [float(lo), float(hi)],
            "clip_quantile": prediction_cfg["clip_quantile"],
            "init": dict(self.init_info_),
            "elapsed_seconds": time.perf_counter() - started,
        }
        return self.training_info_

    # ------------------------------------------------------------- utilities

    def _fill_doc_channel(self, visible: torch.Tensor) -> None:
        """Write channels 8/9 for one visibility pattern (in place on xt)."""
        mu, sd = self.stats
        doc_obs = torch.where(visible, self.y, torch.zeros_like(self.y))
        self.xt[:, :, 8] = ((doc_obs - mu) / sd).permute(1, 0)
        self.xt[:, :, 9] = visible.float().permute(1, 0)

    def _mse_on(self, cells: torch.Tensor) -> float:
        """Unclipped log1p MSE over a cell set, grouped by month."""
        if len(cells) == 0:
            raise ValueError("no cells to score")
        ci, cj = cells // self.t, cells % self.t
        loss, count = 0.0, 0
        for month in np.unique(cj.numpy()):
            selected = ci[cj == month]
            pred = self.forward_month(int(month))
            loss += float(F.mse_loss(pred[selected], self.y[selected, month])) * len(
                selected
            )
            count += len(selected)
        return loss / max(count, 1)

    def _month_vectors(
        self, cells: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-cell (base, correction, total) for an arbitrary cell set."""
        base = torch.full((len(cells),), float("nan"))
        correction = torch.full((len(cells),), float("nan"))
        total = torch.empty(len(cells))
        ci, cj = cells // self.t, cells % self.t
        for month in np.unique(cj.numpy()):
            selector = cj == month
            b, c, t = self.components_month(int(month))
            base[selector] = b[ci[selector]]
            correction[selector] = c[ci[selector]]
            total[selector] = t[ci[selector]]
        return base, correction, total

    def clip_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen truncation rule: train-label min and 99.5th percentile."""
        ti, tj = self.train_cells // self.t, self.train_cells % self.t
        values = self.y[ti, tj]
        return values.min(), torch.quantile(values, self.protocol["prediction"]["clip_quantile"])

    def predict_grid(self, role: str = "full") -> dict[str, np.ndarray]:
        """Full (N, T) log1p grids under one visibility arrangement.

        role="full" opens every non-test observation (the frozen-evaluation
        arrangement); role="eval" keeps the internal validation protocol.
        """
        if role == "full":
            visible = self.full_visible
        elif role == "eval":
            visible = self.eval_visible
        else:
            raise ValueError("unknown visibility role: " + str(role))
        self.model.eval()
        base = torch.full((self.n, self.t), float("nan"))
        correction = torch.full((self.n, self.t), float("nan"))
        total = torch.empty(self.n, self.t)
        with torch.no_grad():
            self._fill_doc_channel(visible)
            for month in range(self.t):
                b, c, t = self.components_month(month)
                base[:, month] = b
                correction[:, month] = c
                total[:, month] = t
        lo, hi = self.clip_bounds()
        clipped = total.clamp(min=lo, max=hi)
        return {
            "base_log": base.numpy().copy(),
            "correction_log": correction.numpy().copy(),
            "total_log": total.numpy().copy(),
            "pred_log_clipped": clipped.numpy().copy(),
            "y_pred": np.expm1(clipped.numpy()).copy(),
            "clipped": (clipped != total).numpy().copy(),
        }

    def raw_log_metrics(self) -> dict:
        """Unclipped log1p metrics on the internal validation cells.

        These use the raw log1p output, which is also the early-stopping
        criterion; the headline metrics in evaluate.metrics use the truncated
        prediction instead, matching the frozen benchmark.
        """
        if self.validation_ is None:
            raise RuntimeError("fit() has not run")
        truth = self.validation_["y_true_log"]
        pred = self.validation_["total_log"]
        residual = truth - pred
        return {
            "raw_log_rmse": float(np.sqrt(np.mean(residual**2))),
            "raw_log_mae": float(np.mean(np.abs(residual))),
            "n": len(residual),
        }
