"""Lightweight support encoder for K-shot adaptation (plan Task 3, case B).

Freezes the H2 base grid and learns a small per-query head that blends the
base prediction with local support statistics (value, residual vs base,
river hop distance). A closed-form residual-IDW is provided as a non-learned
control that should already beat a constant mean-bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def residual_idw(
    pred0: np.ndarray,
    y: np.ndarray,
    support: list[int],
    query: list[int],
    hops,
    t: int,
) -> dict[int, float]:
    """Base prediction + hop-weighted support residuals."""
    out: dict[int, float] = {}
    y_flat = y.ravel()
    p0 = pred0.ravel()
    for q in query:
        qrow = q // t
        wsum, rsum = 0.0, 0.0
        for s in support:
            h = hops(s // t, qrow)
            if h is None:
                continue
            w = 1.0 / (h + 1.0)
            wsum += w
            rsum += w * (y_flat[s] - p0[s])
        out[q] = float(p0[q] + (rsum / wsum if wsum > 0 else 0.0))
    return out


def analytic_blend(
    pred0: np.ndarray,
    y: np.ndarray,
    support: list[int],
    query: list[int],
    hops,
    t: int,
) -> dict[int, float]:
    """Closed-form gate: trust local_mean when support is tight and close."""
    out: dict[int, float] = {}
    y_flat = y.ravel()
    p0 = pred0.ravel()
    svals = y_flat[support]
    local_mean = float(svals.mean())
    spread = float(svals.std() + 1e-6)
    rel_spread = spread / (abs(local_mean) + 1.0)
    for q in query:
        qrow = q // t
        hs = [hops(s // t, qrow) for s in support]
        hs = [h for h in hs if h is not None]
        mean_h = float(np.mean(hs)) if hs else 5.0
        # g in (0,1): nearly pure local_mean when support is tight and close;
        # fall back to H2 when support is noisy or far (mean_h <= 1 is local).
        g = float(np.exp(-2.5 * rel_spread - 0.15 * max(0.0, mean_h - 1.0)))
        out[q] = float((1.0 - g) * p0[q] + g * local_mean)
    return out


FEATURE_DIM = 8


def support_features(
    pred0: np.ndarray,
    y: np.ndarray,
    support: list[int],
    q: int,
    hops,
    t: int,
) -> np.ndarray:
    """Per-query feature vector (FEATURE_DIM,)."""
    y_flat = y.ravel()
    p0 = pred0.ravel()
    svals = y_flat[support]
    res = svals - p0[support]
    qrow = q // t
    hs = np.array(
        [hops(s // t, qrow) for s in support], dtype=np.float64
    )
    hs_finite = hs[np.isfinite(hs)] if hs.size else np.array([5.0])
    if hs_finite.size == 0:
        hs_finite = np.array([5.0])
    return np.array(
        [
            len(support),
            float(svals.mean()),
            float(svals.std() + 1e-6),
            float(res.mean()),
            float(res.std() + 1e-6),
            float(hs_finite.min()),
            float(hs_finite.mean()),
            float(p0[q]),
        ],
        dtype=np.float32,
    )


class SupportEncoder(nn.Module):
    """Predict query DOC from support features + H2 base.

    Output is the final mg/L prediction (not a residual), so the net can
    fall back to the base when support is uninformative. Predictions are
    clamped to the training label range to avoid explosive extrapolation.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.feat_mu = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        self.feat_sd = np.ones((1, FEATURE_DIM), dtype=np.float32)
        self.y_lo = 0.0
        self.y_hi = 50.0

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pred = self.net(feats).squeeze(-1)
        return pred.clamp(min=self.y_lo, max=self.y_hi)


@dataclass
class SupportEpisodeBatch:
    feats: torch.Tensor
    target: torch.Tensor
    feat_mu: np.ndarray | None = None
    feat_sd: np.ndarray | None = None


def sample_episodes(
    pred0: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    candidate_rows: list[int],
    hops,
    t: int,
    k_list: tuple[int, ...] = (1, 3, 5),
    n_episodes: int = 256,
    min_query: int = 2,
    seed: int = 0,
) -> SupportEpisodeBatch:
    """Episodic support/query samples on candidate rows (train-only region)."""
    rng = np.random.default_rng(seed)
    y_flat = y.ravel()
    feats, tgts = [], []
    rows = list(candidate_rows)
    if len(rows) < min_query + 1:
        return SupportEpisodeBatch(
            torch.zeros((0, FEATURE_DIM)), torch.zeros(0)
        )
    for _ in range(n_episodes):
        j = int(rng.integers(0, t))
        obs = [i for i in rows if y_mask[i, j]]
        if len(obs) < min_query + 1:
            continue
        obs = [obs[i] for i in rng.permutation(len(obs))]
        k = int(rng.choice([kk for kk in k_list if kk + min_query <= len(obs)] or [1]))
        k = min(k, len(obs) - min_query)
        support = [i * t + j for i in obs[:k]]
        query = [i * t + j for i in obs[k: k + min_query]]
        if not support or not query:
            continue
        for q in query:
            feats.append(support_features(pred0, y, support, q, hops, t))
            tgts.append(float(y_flat[q]))
    if not feats:
        return SupportEpisodeBatch(
            torch.zeros((0, FEATURE_DIM)), torch.zeros(0)
        )
    x = np.stack(feats).astype(np.float32)
    # standardize features so the MLP is scale-stable across regions
    mu = x.mean(axis=0, keepdims=True).astype(np.float32)
    sd = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    x = (x - mu) / sd
    return SupportEpisodeBatch(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(tgts, dtype=torch.float32),
        feat_mu=mu,
        feat_sd=sd,
    )


def fit_support_encoder(
    batch: SupportEpisodeBatch,
    hidden: int = 32,
    lr: float = 5e-3,
    epochs: int = 120,
    seed: int = 0,
    y_lo: float = 0.0,
    y_hi: float = 50.0,
) -> SupportEncoder:
    torch.manual_seed(seed)
    model = SupportEncoder(hidden=hidden)
    model.y_lo = y_lo
    model.y_hi = y_hi
    if batch.feats.shape[0] == 0:
        return model
    if batch.feat_mu is not None:
        model.feat_mu = batch.feat_mu
        model.feat_sd = batch.feat_sd if batch.feat_sd is not None else model.feat_sd
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x, yb = batch.feats, batch.target
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i: i + 64]
            pred = model(x[idx])
            loss = torch.mean(torch.abs(pred - yb[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def encode_predict(
    model: SupportEncoder,
    pred0: np.ndarray,
    y: np.ndarray,
    support: list[int],
    query: list[int],
    hops,
    t: int,
) -> dict[int, float]:
    model.eval()
    out: dict[int, float] = {}
    mu = np.asarray(model.feat_mu, dtype=np.float32)
    sd = np.asarray(model.feat_sd, dtype=np.float32)
    with torch.no_grad():
        for q in query:
            feat = support_features(pred0, y, support, q, hops, t)
            feat = (feat - mu.reshape(-1)) / sd.reshape(-1)
            out[q] = float(model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))[0])
    return out
