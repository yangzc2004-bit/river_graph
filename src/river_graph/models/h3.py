"""H3-A models: an independent environmental base plus a static river-network
residual branch, added in log1p space (z_hat = base + correction).

Three pieces:

* build_env_features -- the single 21-column non-DOC environmental view shared
  by ENV and by the H3A base branch.  It slices the already-standardised tensor
  that GCNDocModel._build_inputs produces, so there is no second imputation or
  standardisation implementation to drift out of sync.
* EnvironmentalPredictor -- the ENV arm: two 64-wide MLP layers + scalar head.
* H3ResidualImputer -- the H3A arm: the same base branch plus a static
  TransportGCNImputer correction branch whose output head starts at exactly
  zero.  forward_components() exposes base / correction / total separately.

Initialisation is explicit.  Every nn.Linear in a module is re-drawn from a
caller-supplied torch.Generator using PyTorch's default Linear scheme, so the
parameters depend on the frozen sub-seeds rather than on module construction
order.  This is what makes ENV and the H3A base branch bit-identical, and it
reproduces the legacy H2X parameters exactly (see tests/test_h3.py).

Nothing in this module imports or edits river_graph.models.gcn or
river_graph.models.hydro beyond calling them; the frozen gate identities stay
untouched.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

# The 21 non-DOC columns, in the frozen order of configs/h3a_v1.json.
ENV_FEATURE_NAMES = (
    "temp_std",
    "temp_missing",
    "flow_std",
    "flow_missing",
    "season_sin",
    "season_cos",
    "lat_std",
    "lon_std",
    "regime_0",
    "regime_1",
    "regime_2",
    "regime_3",
    "regime_4",
    "regime_5",
    "regime_6",
    "regime_7",
    "regime_8",
    "regime_9",
    "regime_10",
    "regime_11",
    "regime_12",
)
N_ENV_FEATURES = len(ENV_FEATURE_NAMES)

# Column positions inside the _build_inputs tensor: 0-7 are the eight base
# environmental channels, 8/9 are the DOC value and DOC visibility slots, and
# 10.. are the standardised ecological/hydrological attributes.
_BASE_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7)
_DOC_CHANNELS = (8, 9)
_REGIME_START = 10
N_REGIME_FEATURES = 13


def env_feature_index(n_regime: int = N_REGIME_FEATURES) -> list[int]:
    """Column positions of the 21 ENV features inside the _build_inputs tensor."""
    return [*_BASE_CHANNELS, *range(_REGIME_START, _REGIME_START + n_regime)]


def build_env_features(dataset: dict, split: dict) -> torch.Tensor:
    """(T, N, 21) standardised non-DOC environmental view for one dataset.

    Reuses GCNDocModel._build_inputs for the standardisation, then drops the
    two DOC channels.  The result therefore cannot depend on DOC values or on
    DOC visibility, and the caller's tensors are never modified in place.
    """
    from river_graph.models.gcn import GCNDocModel

    if "regime" not in dataset:
        raise ValueError("the ENV view needs the ecological regime block")
    n_regime = int(dataset["regime"].shape[1])
    probe = GCNDocModel()  # env_groups=None, env_encoder=False
    xt = probe._build_inputs(dataset, split)[0]
    if xt.shape[-1] != _REGIME_START + n_regime:
        raise ValueError(
            "unexpected _build_inputs channel layout: "
            + str(xt.shape[-1])
            + " channels for "
            + str(n_regime)
            + " regime columns"
        )
    for channel in _DOC_CHANNELS:
        if not torch.equal(xt[:, :, channel], torch.zeros_like(xt[:, :, channel])):
            raise ValueError("DOC slots are not zero before slicing")
    env = xt[:, :, env_feature_index(n_regime)]
    if env.shape[-1] != N_ENV_FEATURES:
        raise ValueError("ENV view must have 21 columns, got " + str(env.shape[-1]))
    return env.contiguous()


def reset_linear_parameters(module: nn.Module, generator: torch.Generator) -> int:
    """Re-draw every nn.Linear inside a module from an explicit generator.

    Uses exactly PyTorch's nn.Linear.reset_parameters scheme, so a model built
    under torch.manual_seed(s) and the same model re-drawn from
    Generator().manual_seed(s) hold identical parameters.
    """
    drawn = 0
    for sub in module.modules():
        if not isinstance(sub, nn.Linear):
            continue
        nn.init.kaiming_uniform_(sub.weight, a=math.sqrt(5), generator=generator)
        if sub.bias is not None:
            fan_in = sub.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(sub.bias, -bound, bound, generator=generator)
        drawn += 1
    if drawn == 0:
        raise ValueError("no nn.Linear parameters to initialise")
    return drawn


def zero_head(module: nn.Module) -> None:
    """Force a module's final nn.Linear weight and bias to exactly zero."""
    head = getattr(module, "head", None)
    if not isinstance(head, nn.Linear):
        raise TypeError("module has no nn.Linear head to zero")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()


class EnvironmentalPredictor(nn.Module):
    """ENV arm: per-node MLP over the 21 non-DOC environmental features.

    Two 64-wide ReLU layers with dropout, then a scalar head.  Nothing in the
    forward path reads edges, DOC channels, or other nodes, so an isolated node
    is predicted exactly like any other.
    """

    def __init__(
        self,
        in_features: int = N_ENV_FEATURES,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        dims = [in_features, *([hidden] * layers)]
        blocks: list[nn.Module] = []
        for a, b in itertools.pairwise(dims):
            blocks.extend((nn.Linear(a, b), nn.ReLU()))
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        if env.shape[-1] != self.trunk[0].in_features:
            raise ValueError("unexpected ENV width: " + str(tuple(env.shape)))
        h = env
        for layer in self.trunk:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)

    def reset_parameters(self, generator: torch.Generator) -> None:
        reset_linear_parameters(self, generator)


def make_h2x_trunk(
    in_channels: int,
    edge_dim: int,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.1,
    env_dim: int = 0,
    env_emb: int = 32,
    gate_mode: str = "static",
) -> nn.Module:
    """A static TransportGCNImputer, i.e. the frozen H2X trunk."""
    from river_graph.models.hydro import TransportGCNImputer

    return TransportGCNImputer(
        in_channels,
        edge_dim,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        env_dim=env_dim,
        env_emb=env_emb,
        gate_mode=gate_mode,
    )


class H3ResidualImputer(nn.Module):
    """H3A arm: base(env) + correction(graph), added in log1p space.

    base and correction are produced in log1p space and summed there.  They are
    never expm1-ed separately, so the mg/L prediction is expm1(total) exactly.

    edge_set="empty" gives the T18 no-message control: the correction branch
    keeps its self path (self_lin) and its gates are simply never exercised.
    """

    def __init__(
        self,
        base: EnvironmentalPredictor,
        correction: nn.Module,
        edge_set: str = "river",
    ):
        super().__init__()
        if edge_set not in ("river", "empty"):
            raise ValueError("edge_set must be 'river' or 'empty'")
        self.base = base
        self.correction = correction
        self.edge_set = edge_set

    def resolve_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.edge_set == "empty":
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return edge_index

    def forward_components(
        self,
        env: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        env_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(base, correction, total), all in log1p space."""
        base = self.base(env)
        correction = self.correction(
            x, self.resolve_edges(edge_index), edge_attr, env_raw
        )
        return base, correction, base + correction

    def forward(
        self,
        env: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        env_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_components(env, x, edge_index, edge_attr, env_raw)[2]

    def initialise(
        self,
        base_seed: int,
        correction_seed: int,
        zero_correction_head: bool = True,
    ) -> dict:
        """Deterministic init: identical base sub-seed, independent correction."""
        base_generator = torch.Generator().manual_seed(int(base_seed))
        correction_generator = torch.Generator().manual_seed(int(correction_seed))
        base_drawn = reset_linear_parameters(self.base, base_generator)
        correction_drawn = reset_linear_parameters(
            self.correction, correction_generator
        )
        if zero_correction_head:
            zero_head(self.correction)
        return {
            "base_seed": int(base_seed),
            "correction_seed": int(correction_seed),
            "base_linear_layers": base_drawn,
            "correction_linear_layers": correction_drawn,
            "correction_head_zeroed": bool(zero_correction_head),
        }
