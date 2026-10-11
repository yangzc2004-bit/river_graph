"""Official SAITS/GRIN backbones adapted to label-free DOC deployment.

No published architecture is reimplemented here. Upstream Python files are
loaded at pinned revisions; only the DOC input/target interface is adapted.
These are explicitly named covariate-only adapters, not the papers' native
DOC-assisted masking experiments.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

from river_graph.experiments.provenance import sha256_file

THIRD_PARTY = Path(__file__).resolve().parents[3] / "cache/related_work"
OFFICIAL = {
    "saits": {"directory": "SAITS", "repository": "https://github.com/WenjieDu/SAITS",
        "commit": "660b87f19c1277065f314f24134f646229e89ca9",
        "files": ("modeling/saits.py", "modeling/layers.py", "modeling/utils.py", "LICENSE")},
    "grin": {"directory": "GRIN", "repository": "https://github.com/Graph-Machine-Learning-Group/grin",
        "commit": "4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e",
        "files": ("lib/__init__.py", "lib/nn/layers/gril.py", "lib/nn/layers/gcrnn.py",
                  "lib/nn/layers/spatial_conv.py", "lib/nn/layers/spatial_attention.py",
                  "lib/nn/utils/ops.py")},
}
RECIPES = {"saits": {"width64": {"width": 64}, "width128": {"width": 128}},
           "grin": {"width32": {"width": 32}, "width64": {"width": 64}}}


def official_file_hashes(root=THIRD_PARTY):
    root = Path(root)
    records = {method: {**{key: value for key, value in record.items() if key != "files"},
        "files": {name: sha256_file(root / record["directory"] / name) for name in record["files"]}}
        for method, record in OFFICIAL.items()}
    dependency = root / "python_deps"
    records["einops"] = {"version": "0.8.1", "files": {
        str(path.relative_to(dependency)): sha256_file(path)
        for path in sorted((dependency / "einops").rglob("*.py"))}}
    if not records["einops"]["files"]:
        raise FileNotFoundError("isolated einops dependency unavailable")
    return records


def _package(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__path__[0]).resolve() != Path(path).resolve():
            raise ValueError(f"official namespace {name} already bound to another source")
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def load_saits(root=THIRD_PARTY):
    root = Path(root) / "SAITS"
    _package("modeling", root / "modeling")
    return importlib.import_module("modeling.saits").SAITS


def load_grin(root=THIRD_PARTY):
    root = Path(root)
    dependency = str(root / "python_deps")
    if dependency not in sys.path:
        sys.path.insert(0, dependency)
    base = "_doc_official_grin"
    if base not in sys.modules:
        spec = importlib.util.spec_from_file_location(base, root / "GRIN/lib/__init__.py",
                                                     submodule_search_locations=[str(root / "GRIN/lib")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[base] = module
        spec.loader.exec_module(module)
    # Bypass only dependency-heavy package re-exports; load the actual core files.
    for suffix, path in (("nn", "nn"), ("nn.layers", "nn/layers"), ("nn.utils", "nn/utils")):
        _package(f"{base}.{suffix}", root / "GRIN/lib" / path)
    return importlib.import_module(f"{base}.nn.layers.gril").BiGRIL


def exogenous_sequence(windows, environment):
    if windows.shape[-2:] != (12, 8) or environment.shape != (windows.shape[1], 15):
        raise ValueError("declared 12-month/8-channel/15-environment input contract required")
    b, n, length, _ = windows.shape
    env = environment[None, :, None].expand(b, n, length, -1)
    return torch.cat([windows, env], -1)


def saits_inputs(windows, environment):
    exogenous = exogenous_sequence(windows, environment)
    b, n, length, _ = exogenous.shape
    zero_doc = torch.zeros(b, n, length, 1, device=exogenous.device, dtype=exogenous.dtype)
    values = torch.cat([exogenous, zero_doc], -1).reshape(b * n, length, 24)
    mask = torch.ones_like(values)
    mask[..., :2] = windows[..., 2:4].reshape(b * n, length, 2)
    mask[..., -1] = 0.  # No DOC, including source history, ever enters the backbone.
    return {"X": values, "missing_mask": mask}


class SAITSDOCAdapter(nn.Module):
    def __init__(self, width=64, *, root=THIRD_PARTY):
        super().__init__()
        self.core = load_saits(root)(n_groups=2, n_group_inner_layers=1, d_time=12,
            d_feature=24, d_model=width, d_inner=width * 2, n_head=4,
            d_k=width // 4, d_v=width // 4, dropout=.1,
            input_with_mask=True, param_sharing_strategy="inner_group", MIT=True,
            device="cpu", diagonal_attention_mask=True)

    def forward(self, windows, environment, season, graph):
        values = self.core(saits_inputs(windows, environment), stage="test")["imputed_data"]
        return values[:, -1, -1].reshape(windows.shape[:2])

    def objective(self, windows, environment, target, visible, graph):
        b, n = target.shape
        inputs = saits_inputs(windows, environment)
        holdout = torch.zeros_like(inputs["X"])
        indicating = torch.zeros_like(holdout)
        holdout[:, -1, -1] = torch.where(visible, target, 0.).reshape(b * n)
        indicating[:, -1, -1] = visible.reshape(b * n)
        result = self.core({**inputs, "X_holdout": holdout, "indicating_mask": indicating}, stage="train")
        return result["reconstruction_loss"] + result["imputation_loss"]


class GRINDOCAdapter(nn.Module):
    def __init__(self, width=32, *, root=THIRD_PARTY):
        super().__init__()
        self.core = load_grin(root)(input_size=1, hidden_size=width, ff_size=width,
            ff_dropout=.1, n_layers=1, n_nodes=None, embedding_size=0, u_size=23,
            kernel_size=2, decoder_order=1, global_att=False, layer_norm=False, merge="mlp")

    def outputs(self, windows, environment, graph):
        b, n, length, _ = windows.shape
        values = torch.zeros(b, 1, n, length, device=windows.device, dtype=windows.dtype)
        mask = torch.zeros_like(values, dtype=torch.bool)
        exogenous = exogenous_sequence(windows, environment).permute(0, 3, 1, 2)
        # Never cache graph support across source/validation/target graph changes.
        return self.core(values, graph["adjacency"], mask=mask, u=exogenous, cached_support=False)

    def forward(self, windows, environment, season, graph):
        output, _ = self.outputs(windows, environment, graph)
        return output[:, 0, :, -1]

    def objective(self, windows, environment, target, visible, graph):
        output, auxiliary = self.outputs(windows, environment, graph)
        prediction = output[:, 0, :, -1]
        primary = torch.abs(prediction[visible] - target[visible]).mean()
        # The official fitter adds the masked errors of its four prediction heads.
        extra = auxiliary[:, :, 0, :, -1][:, visible]
        return primary + torch.abs(extra - target[visible][None]).sum() / int(visible.sum())


def build_related_model(method, recipe, *, root=THIRD_PARTY):
    if method == "saits":
        return SAITSDOCAdapter(**recipe, root=root)
    if method == "grin":
        return GRINDOCAdapter(**recipe, root=root)
    raise ValueError("undeclared related-work model")
