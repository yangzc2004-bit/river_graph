"""GCN imputation model for the DOC reconstruction benchmark.

Monthly snapshot imputation: at month t the model sees node features +
the observed-DOC channel (0 where unobserved) and predicts DOC at all
nodes. Loss is computed on train cells only (design.md leakage rules).

Topology variants (the paper's core ablation):
- "river":  real directed river graph, symmetrized for GCN
- "random": same number of edges, uniformly random node pairs
- "none":   no edges (GCN degrades to a per-node MLP)
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

IN_CHANNELS = 10  # temp, flow, temp_m, flow_m, sin, cos, lat, lon, doc_obs, doc_obs_m


class GCNImputer(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels, *([hidden] * layers)]
        for a, b in itertools.pairwise(dims):
            self.convs.append(GCNConv(a, b))
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


def make_edge_index(variant: str, edge_index: torch.Tensor, n_nodes: int,
                    seed: int = 42) -> torch.Tensor:
    """Topology ablation: real river graph / random graph / no edges."""
    if variant == "river":
        return to_undirected(edge_index)
    if variant == "random":
        rng = np.random.default_rng(seed)
        n_edges = edge_index.shape[1]
        pairs = rng.choice(n_nodes * (n_nodes - 1), size=n_edges, replace=False)
        src, dst = pairs // (n_nodes - 1), pairs % (n_nodes - 1)
        dst = np.where(dst >= src, dst + 1, dst)  # skip self-loops
        ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return to_undirected(ei)
    if variant == "none":
        return torch.empty((2, 0), dtype=torch.long)
    raise ValueError(f"unknown variant: {variant}")


class GCNDocModel:
    """fit_predict(dataset, split) interface shared with the baselines."""

    def __init__(self, variant: str = "river", hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, lr: float = 1e-3, max_epochs: int = 200,
                 patience: int = 20, seed: int = 0, architecture: str = "gcn",
                 share_weights: bool = False, edge_dropout: float = 0.0,
                 weight_decay: float = 0.0):
        self.variant = variant
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.architecture = architecture
        self.share_weights = share_weights
        self.edge_dropout = edge_dropout
        self.weight_decay = weight_decay

    def _build_inputs(self, dataset: dict, split: dict[str, np.ndarray]):
        """Fixed features + pieces for the dynamic DOC-obs channel.

        Returns (xt_static, y, base_visible, train_cells) where
        xt_static is (T, N, C) with the two DOC channels left as zeros,
        y is log1p mg/L, base_visible marks val+context cells (always
        visible, never used for loss), and train_cells are flat indices.
        """
        y = torch.log1p(dataset["y"])  # (N, T)
        x = dataset["x"]  # (N, T, 2), already 0-filled
        x_mask = dataset["x_mask"].float()
        n, t = y.shape

        base_visible = torch.zeros(n * t, dtype=torch.bool)
        for key in ("val", "context"):
            if key in split and len(split[key]):
                base_visible[torch.as_tensor(split[key])] = True
        base_visible = base_visible.reshape(n, t)

        months = torch.tensor(
            [int(str(m)[5:7]) for m in dataset["months"]], dtype=torch.float32
        )
        season = torch.stack(
            [torch.sin(2 * np.pi * months / 12), torch.cos(2 * np.pi * months / 12)],
            dim=1,
        )  # (T, 2)
        latlon = dataset["static"]  # (N, 2)

        train_cells = torch.as_tensor(split["train"])
        ti, tj = train_cells // t, train_cells % t

        def standardized(v: torch.Tensor) -> torch.Tensor:
            mu = v[ti, tj].mean() if v.shape == y.shape else v.mean()
            sd = v[ti, tj].std() if v.shape == y.shape else v.std()
            return (v - mu) / (sd + 1e-8)

        feats = [
            standardized(x[:, :, 0]), x_mask[:, :, 0],
            standardized(x[:, :, 1]), x_mask[:, :, 1],
            season[:, 0].expand(n, t), season[:, 1].expand(n, t),
            standardized(latlon[:, 0:1]).expand(n, t),
            standardized(latlon[:, 1:2]).expand(n, t),
            torch.zeros(n, t), torch.zeros(n, t),  # doc_obs channel slots
        ]
        if "regime" in dataset:  # H2: hydrologic regime channels (static)
            reg = dataset["regime"].float()
            reg = (reg - reg.mean(0)) / (reg.std(0) + 1e-8)
            for c in range(reg.shape[1]):
                feats.append(reg[:, c:c + 1].expand(n, t))
        xt_static = torch.stack(feats, dim=-1).permute(1, 0, 2).contiguous()
        # fixed log-space standardization stats for the DOC channel
        doc_mu, doc_sd = y[ti, tj].mean(), y[ti, tj].std() + 1e-8
        return xt_static, y, base_visible, train_cells, (doc_mu, doc_sd)

    @staticmethod
    def _fill_doc_channel(xt, y, visible, stats):
        """Set channels 8/9 (standardized observed DOC + visibility mask)."""
        mu, sd = stats
        doc_obs = torch.where(visible, y, torch.zeros_like(y))
        xt[:, :, 8] = ((doc_obs - mu) / sd).permute(1, 0)
        xt[:, :, 9] = visible.float().permute(1, 0)
        return xt

    def fit_predict(self, dataset: dict, split: dict[str, np.ndarray]) -> np.ndarray:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n, t = dataset["y"].shape
        xt, y, base_visible, train_cells, stats = self._build_inputs(dataset, split)
        if self.architecture == "directed":
            from river_graph.models.hydro import DirectedGCNImputer, make_directed_edges

            ei = make_directed_edges(self.variant, dataset["edge_index"], n)
            model = DirectedGCNImputer(xt.shape[-1], self.hidden, self.layers,
                                       self.dropout, share_weights=self.share_weights,
                                       edge_dropout=self.edge_dropout)
        elif self.architecture == "transport":
            from river_graph.models.hydro import TransportGCNImputer

            ei = dataset["edge_index"]  # raw directed; gates need real edges
            edge_attr = dataset["edge_attr"]
            edge_attr = (edge_attr - edge_attr.mean(0)) / (edge_attr.std(0) + 1e-8)
            model = TransportGCNImputer(xt.shape[-1], edge_attr.shape[1],
                                        self.hidden, self.layers, self.dropout)
        else:
            ei = make_edge_index(self.variant, dataset["edge_index"], n)
            model = GCNImputer(xt.shape[-1], self.hidden, self.layers, self.dropout)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        if self.architecture == "transport":
            def fwd(xb):
                return model(xb, ei, edge_attr)
        else:
            def fwd(xb):
                return model(xb, ei)
        months_idx = np.arange(t)

        # Train with per-epoch re-masking of train cells: half stay visible
        # as context, half are hidden and used for the loss. Without this the
        # model learns to copy the input channel and never learns imputation.
        best_loss, best_state, bad = float("inf"), None, 0
        for _epoch in range(self.max_epochs):
            perm = rng.permutation(train_cells.numpy())
            half = len(perm) // 2
            ctx_cells = torch.as_tensor(perm[:half])
            tgt_cells = torch.as_tensor(perm[half:])
            visible = base_visible.clone()
            visible.reshape(-1)[ctx_cells] = True
            self._fill_doc_channel(xt, y, visible, stats)
            ti, tj = tgt_cells // t, tgt_cells % t

            model.train()
            np.random.shuffle(months_idx)
            tgt_loss = 0.0
            for j in months_idx:
                sel = ti[tj == j]
                if len(sel) == 0:
                    continue
                pred = fwd(xt[j])
                loss = F.mse_loss(pred[sel], y[sel, j])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                # hidden-from-input cells: honest proxy for imputation skill
                tgt_loss += loss.item() * len(sel)
            tgt_loss /= max(len(tgt_cells), 1)
            if tgt_loss < best_loss - 1e-6:
                best_loss = tgt_loss
                # state_dict() returns live references; clone or continued
                # training mutates the "best" state
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        # inference: all train cells become visible context
        visible = base_visible.clone()
        visible.reshape(-1)[train_cells] = True
        self._fill_doc_channel(xt, y, visible, stats)
        model.eval()
        preds = torch.empty(n, t)
        with torch.no_grad():
            for j in range(t):
                preds[:, j] = fwd(xt[j])
        # clamp to the observed train range: tree baselines (RF) cannot
        # extrapolate beyond training targets by construction, so the GNN
        # gets a similar physical bound. Use the 99.5th percentile, not the
        # max — the max is a single flood-event spike (445 mg/L) and letting
        # pathological extrapolations clamp there still destroys mg/L R2.
        ti, tj = train_cells // t, train_cells % t
        lo = y[ti, tj].min()
        hi = torch.quantile(y[ti, tj], 0.995)
        preds = preds.clamp(min=lo, max=hi)
        return np.expm1(preds.numpy())
