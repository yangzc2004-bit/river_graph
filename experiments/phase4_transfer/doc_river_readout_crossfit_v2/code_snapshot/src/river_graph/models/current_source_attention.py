"""Select current source innovations inside the existing DOC residual head.

Candidate sources are ecological neighbours, not physical upstream edges.
Source values and hydrological keys are caller-frozen, station-cross-fitted
inputs. Receiving water-quality labels are never arguments to this operator.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import _state_copy


class _CurrentSourceHead(nn.Module):
    def __init__(self, linear, hidden, ecology, heads, dimensions, mode, seed):
        super().__init__()
        self.linear = linear
        self.heads, self.dimensions, self.mode = heads, dimensions, mode
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1907)
            self.query = nn.Linear(hidden + ecology + 8, heads * dimensions,
                                   bias=False, dtype=linear.weight.dtype)
            self.key = nn.Linear(ecology + 8, heads * dimensions,
                                 bias=False, dtype=linear.weight.dtype)
            self.output = nn.Linear(heads, 1, bias=False, dtype=linear.weight.dtype)
        nn.init.zeros_(self.output.weight)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def allocate(self, query, keys, values, valid, log_prior):
        q = self.query(query).reshape(len(query), self.heads, self.dimensions)
        k = self.key(keys).reshape(len(query), keys.shape[1], self.heads, self.dimensions)
        scores = torch.einsum("bhd,bchd->bhc", q, k) / math.sqrt(self.dimensions)
        if self.mode == "fixed_prior":
            # Keep identical allocated parameters and connected zero gradients.
            scores = scores * 0
        scores = scores + log_prior[:, None]
        weights = torch.softmax(scores.masked_fill(~valid[:, None], -torch.inf), -1)
        state = torch.einsum("bhc,bc->bh", weights, values)
        return state, weights


class CurrentSourceAttentionResidual(EncoderNativeResidual):
    """Two donor-selection heads added to the retained scalar residual readout."""

    def __init__(self, spatial, temporal, decay, *, attention_mode="learned",
                 attention_heads=2, attention_dimensions=32, daily_start=30, **kwargs):
        if attention_mode not in ("learned", "fixed_prior"):
            raise ValueError("attention_mode must be learned or fixed_prior")
        if any(type(value) is not int or value < 1
               for value in (attention_heads, attention_dimensions)):
            raise ValueError("positive attention heads and dimensions required")
        if type(daily_start) is not int or daily_start < 0:
            raise ValueError("nonnegative daily_start required")
        super().__init__(spatial, temporal, decay, **kwargs)
        if self.extra_dim < daily_start + 8:
            raise ValueError("extra inputs must contain the eight daily hydrology columns")
        self.attention_config = {"attention_mode": attention_mode, "attention_heads": attention_heads,
                                 "attention_dimensions": attention_dimensions, "daily_start": daily_start}
        self._install_head()

    def _install_head(self):
        self.head = _CurrentSourceHead(self.head, self.hidden_size,
            self.spatial_architecture["env_dim"], self.attention_config["attention_heads"],
            self.attention_config["attention_dimensions"], self.attention_config["attention_mode"], self.seed)
        self._initial_attention_head = _state_copy(self.head)
        self.attention_parameter_count_ = sum(p.numel() for module in
            (self.head.query, self.head.key, self.head.output) for p in module.parameters())
        self.trainable_parameter_count_ += self.attention_parameter_count_

    def _prepare_inputs(self, inputs):
        arrays = super()._prepare_inputs(inputs)
        required = ("donor_owner", "donor_ecology", "donor_values", "donor_valid",
                    "donor_log_prior", "donor_hydro_bank")
        if not all(name in inputs for name in required):
            raise ValueError("current source attention requires all donor arrays")
        owner = np.asarray(inputs["donor_owner"])
        valid = np.asarray(inputs["donor_valid"])
        if owner.dtype.kind not in "iu" or valid.dtype != np.bool_:
            raise ValueError("donor owners must be integers and donor validity boolean")
        arrays.update({name: torch.as_tensor(inputs[name], dtype=self.dtype).detach().cpu()
                       for name in required if name not in ("donor_owner", "donor_valid")})
        arrays["donor_owner"] = torch.as_tensor(owner, dtype=torch.long)
        arrays["donor_valid"] = torch.as_tensor(valid, dtype=torch.bool)
        n, t = arrays["age"].shape
        e = self.spatial_architecture["env_dim"]
        bank = arrays["donor_hydro_bank"]
        if owner.ndim != 2 or owner.shape[0] != n or owner.shape[1] < 2:
            raise ValueError("donor_owner must align with receiver stations and include a prior")
        c = owner.shape[1]
        if (bank.ndim != 3 or bank.shape[0] < 1 or bank.shape[1:] != (t, 8)
                or arrays["donor_ecology"].shape != (n, c, e)
                or arrays["donor_values"].shape != (n, t, c)
                or valid.shape != (n, t, c) or arrays["donor_log_prior"].shape != (n, c)
                or (owner < -1).any() or (owner >= bank.shape[0]).any()
                or not all(torch.isfinite(arrays[name]).all() for name in required
                           if name not in ("donor_owner", "donor_valid"))):
            raise ValueError("donor arrays must be finite and station/calendar aligned")
        if (not valid[:, :, -1].all() or (owner[:, -1] != -1).any()
                or torch.count_nonzero(arrays["donor_values"][..., -1])
                or torch.count_nonzero(arrays["donor_ecology"][:, -1])
                or torch.count_nonzero(arrays["donor_log_prior"][:, -1])
                or torch.count_nonzero(arrays["donor_values"][~arrays["donor_valid"]])
                or (valid[..., :-1] & (owner[:, None, :-1] == -1)).any()):
            raise ValueError("invalid donors must be hidden; last candidate must be the unit zero prior")
        return arrays

    def _attention_cells(self, inputs, cells, hidden):
        months = inputs["age"].shape[1]
        station, month = cells // months, cells % months
        owner = inputs["donor_owner"][station]
        donor_hydro = inputs["donor_hydro_bank"][owner.clamp_min(0), month[:, None]]
        donor_hydro = donor_hydro * (owner >= 0)[..., None]
        keys = torch.cat([inputs["donor_ecology"][station], donor_hydro], -1)
        start = self.attention_config["daily_start"]
        query = torch.cat([hidden, inputs["env"][station],
                           inputs["extra"][station, month, start:start + 8]], -1)
        return self.head.allocate(query, keys, inputs["donor_values"][station, month],
                                  inputs["donor_valid"][station, month], inputs["donor_log_prior"][station])

    def _delta_cells(self, inputs, cells):
        cells = torch.as_tensor(cells, dtype=torch.long)
        hidden = self._hidden_cells(inputs, cells)
        months = inputs["age"].shape[1]
        extra = inputs["extra"][cells // months, cells % months]
        pieces = [hidden, extra]
        if self.interaction_indices:
            interaction = hidden.unsqueeze(-1) * extra[:, self.interaction_indices].unsqueeze(1)
            pieces.append(interaction.flatten(1))
        state, _ = self._attention_cells(inputs, cells, hidden)
        delta = (self.head.linear(torch.cat(pieces, -1)) + self.head.output(state)).squeeze(-1).double()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite current source attention residual")
        return delta

    def fit(self, *args, **kwargs):
        self.head.load_state_dict(self._initial_attention_head)
        return super().fit(*args, **kwargs)

    def diagnostics(self, inputs, cells=None):
        arrays = self._prepare_inputs(inputs)
        n, t = arrays["age"].shape
        cells = np.arange(n * t) if cells is None else np.asarray(cells)
        if cells.ndim != 1 or cells.dtype.kind not in "iu" or (cells < 0).any() or (cells >= n*t).any():
            raise ValueError("diagnostic cells must be valid station/month indices")
        chunks = {name: [] for name in ("prior_mass", "entropy", "current_source_state")}
        with torch.no_grad():
            for offset in range(0, len(cells), self.batch_size):
                selected = torch.as_tensor(cells[offset:offset+self.batch_size], dtype=torch.long)
                hidden = self._hidden_cells(arrays, selected)
                state, weights = self._attention_cells(arrays, selected, hidden)
                chunks["prior_mass"].append(weights[..., -1].numpy())
                chunks["entropy"].append(-(weights * weights.clamp_min(1e-30).log()).sum(-1).numpy())
                chunks["current_source_state"].append(state.numpy())
        return {name: np.concatenate(values, axis=0) if values else np.zeros((0, self.head.heads))
                for name, values in chunks.items()}

    def to_dict(self):
        summary = super().to_dict()
        summary.update({"model_class": type(self).__name__, "attention_config": self.attention_config.copy(),
            "attention_trainable_parameters": self.attention_parameter_count_,
            "attention_output_norm": float(self.head.output.weight.detach().double().norm()),
            "query_parameter_change": float((self.head.query.weight.detach() -
                self._initial_attention_head["query.weight"]).double().norm()),
            "key_parameter_change": float((self.head.key.weight.detach() -
                self._initial_attention_head["key.weight"]).double().norm())})
        summary["protocol"].update({"source_attention": "ecological candidates and same-month hydrology",
            "source_values": "double-held-fold OOF native seasonal innovations; unit zero prior",
            "initialization": "original encoder/GRU/decay; zero scalar and attention-value outputs"})
        return summary

    def to_payload(self):
        payload = super().to_payload()
        payload.update({"model_class": type(self).__name__, "attention_config": self.attention_config.copy(),
                        "initial_attention_head": copy.deepcopy(self._initial_attention_head)})
        return payload

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1 or payload.get("model_class") != cls.__name__:
            raise ValueError("unsupported current source attention checkpoint")
        plain = copy.deepcopy(payload)
        plain["model_class"] = "EncoderNativeResidual"
        plain["head"] = {"weight": payload["head"]["linear.weight"], "bias": payload["head"]["linear.bias"]}
        base = EncoderNativeResidual.from_payload(plain)
        obj = cls(base.spatial, base.temporal, base.decay,
                  **base._config(), **payload["attention_config"])
        # Preserve the base loader's initial/fitted recurrent and encoder state.
        head, config, count = obj.head, obj.attention_config, obj.attention_parameter_count_
        obj.__dict__ = base.__dict__.copy()
        obj.head, obj.attention_config, obj.attention_parameter_count_ = head, config, count
        obj.trainable_parameter_count_ += count
        obj.head.load_state_dict(payload["head"])
        initial = copy.deepcopy(payload["initial_attention_head"])
        if (set(initial) != set(obj.head.state_dict())
                or not all(torch.isfinite(value).all() for value in initial.values())
                or torch.count_nonzero(initial["output.weight"])):
            raise ValueError("finite initial attention state with zero output required")
        if not all(torch.isfinite(p).all() for p in obj.head.parameters()):
            raise ValueError("checkpoint contains nonfinite attention weights")
        obj._initial_attention_head = initial
        return obj
