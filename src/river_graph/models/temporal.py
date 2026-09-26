"""Temporal extensions of the released H2X spatial graph model.

H2X-T keeps the TransportGCNImputer spatial trunk unchanged and applies a
causal GRU to a rolling sequence of monthly node representations.  The
implementation is intentionally independent of the DOC-specific benchmark
wrapper so it can be used for each validated analyte with the same model
class.
"""

from __future__ import annotations

import torch
from torch import nn

from river_graph.models.hydro import TransportGCNImputer


class TemporalTransportGCNImputer(nn.Module):
    """H2X spatial encoder followed by a causal rolling GRU.

    Parameters
    ----------
    spatial:
        An initialized :class:`TransportGCNImputer`.  Its encoder and
        prediction head are reused directly.
    lookback:
        Number of monthly representations supplied to the GRU for each
        prediction.  The current month is the final step of every window.
    temporal_hidden:
        GRU output width.  It is kept equal to the spatial hidden width in
        the released H2X-T configuration so the original prediction head can
        be reused without a new output projection.
    """

    def __init__(self, spatial: TransportGCNImputer, *, lookback: int = 12,
                 temporal_hidden: int = 64, chunk_months: int = 256):
        super().__init__()
        if lookback < 1:
            raise ValueError("lookback must be positive")
        if temporal_hidden < 1:
            raise ValueError("temporal_hidden must be positive")
        if chunk_months < 1:
            raise ValueError("chunk_months must be positive")
        self.spatial = spatial
        self.lookback = int(lookback)
        self.temporal_hidden = int(temporal_hidden)
        self.chunk_months = int(chunk_months)
        spatial_hidden = spatial.head.in_features
        if temporal_hidden != spatial_hidden:
            raise ValueError(
                "temporal_hidden must equal the spatial head width when the "
                "existing H2X prediction head is reused"
            )
        self.temporal = nn.GRU(
            input_size=spatial_hidden + 1,  # + history-valid flag
            hidden_size=temporal_hidden,
            num_layers=1,
        )

    @property
    def head(self) -> nn.Linear:
        """Expose the spatial head for callers that inspect model width."""
        return self.spatial.head

    def encode_months(self, x_seq: torch.Tensor,
                      edge_index: torch.Tensor,
                      edge_attr: torch.Tensor,
                      env_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a chronological sequence of snapshots.

        ``x_seq`` has shape ``[T, N, C]`` and the result has shape
        ``[T, N, H]``.  The same spatial parameters are applied at every
        month.
        """
        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape [time, nodes, channels]")
        t, n, channels = x_seq.shape
        # Treat each month as a disconnected copy of the same graph.  This
        # keeps the spatial operator exactly unchanged while replacing a
        # Python loop over 654 monthly calls with a few batched calls.
        outputs = []
        for start in range(0, t, self.chunk_months):
            stop = min(start + self.chunk_months, t)
            count = stop - start
            x_batch = x_seq[start:stop].reshape(count * n, channels)
            if edge_index.numel():
                offsets = torch.arange(count, device=edge_index.device) * n
                batched_edges = (
                    edge_index[:, None, :].expand(-1, count, -1)
                    + offsets[None, :, None]
                ).reshape(2, -1)
                batched_attr = edge_attr.repeat(count, 1)
            else:
                batched_edges = edge_index
                batched_attr = edge_attr
            env_batch = env_raw.repeat(count, 1) if env_raw is not None else None
            encoded = self.spatial.encode_nodes(
                x_batch, batched_edges, batched_attr, env_batch
            )
            outputs.append(encoded.reshape(count, n, -1))
        return torch.cat(outputs, dim=0)

    def _temporal_windows(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        """Apply rolling causal GRU windows to ``[T, N, H]`` hidden states."""
        if hidden_seq.ndim != 3:
            raise ValueError("hidden_seq must have shape [time, nodes, hidden]")
        t, n, h = hidden_seq.shape
        outputs: list[torch.Tensor] = []
        device = hidden_seq.device
        for start in range(0, t, self.chunk_months):
            stop = min(start + self.chunk_months, t)
            months = torch.arange(start, stop, device=device)
            offsets = torch.arange(self.lookback - 1, -1, -1, device=device)
            raw_idx = months[:, None] - offsets[None, :]
            valid = (raw_idx >= 0).float()
            idx = raw_idx.clamp(min=0)
            # [chunk, lookback, nodes, hidden] -> [lookback, chunk*nodes, hidden]
            seq = hidden_seq[idx].permute(1, 0, 2, 3).reshape(
                self.lookback, (stop - start) * n, h
            )
            valid_feature = valid.T[:, :, None].expand(
                self.lookback, stop - start, n
            ).reshape(self.lookback, (stop - start) * n, 1)
            gru_in = torch.cat([seq, valid_feature], dim=-1)
            gru_out, _ = self.temporal(gru_in)
            last = gru_out[-1].reshape(stop - start, n, self.temporal_hidden)
            outputs.append(self.spatial.predict_from_hidden(last))
        return torch.cat(outputs, dim=0)

    def forward_sequence(self, x_seq: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_attr: torch.Tensor,
                         env_raw: torch.Tensor | None = None) -> torch.Tensor:
        """Predict every month in a chronological input sequence."""
        hidden = self.encode_months(x_seq, edge_index, edge_attr, env_raw)
        return self._temporal_windows(hidden)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                env_raw: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_sequence(x_seq, edge_index, edge_attr, env_raw)
