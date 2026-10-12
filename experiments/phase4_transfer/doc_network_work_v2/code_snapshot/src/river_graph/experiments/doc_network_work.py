"""Common-data SSN covariance and official PhyST-HC architecture adaptations.

These are not native-input reproductions. SSN uses an environmental linear mean,
static network Gaussian random effects and independent measurement residuals.
Its exponential kernels follow USEPA SSN2 cov_matrix/cov_vector definitions.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import minimize
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

SSN_RECIPES = {
    "log_network": {"target": "log1p", "kernels": ["tailup", "taildown"]},
    "log_network_euclidean": {"target": "log1p", "kernels": ["tailup", "taildown", "euclidean"]},
    "native_network_euclidean": {"target": "native", "kernels": ["tailup", "taildown", "euclidean"]},
}
PHYST_RECIPES = {
    "compact": {"hidden_trans": 96, "hidden_gcn": 24, "dropout": .1},
    "wider": {"hidden_trans": 128, "hidden_gcn": 48, "dropout": .1},
}


def kernel(geometry, name, scale):
    if scale <= 0:
        raise ValueError("positive covariance range required")
    if name == "tailup":
        return geometry["tailup_weight"] * np.exp(-geometry["distance"] / scale)
    if name == "taildown":
        return np.exp(-geometry["distance"] / scale)
    if name == "euclidean":
        return np.exp(-geometry["euclidean"] / scale)
    raise ValueError("unknown SSN kernel")


class SpatialStreamNetwork:
    """Exact grouped REML for y = X beta + station-network effect + noise.

    Grouping repeated station measurements reduces the covariance solve to the
    number of source stations, without subsampling DOC or fitting station IDs as
    independent predictors. Prediction conditions on source supervision only.
    """

    def __init__(self, geometry, kernels, maxiter=100):
        self.geometry, self.kernels, self.maxiter = geometry, tuple(kernels), maxiter

    def covariance(self, theta):
        value = np.zeros_like(self.geometry["euclidean"])
        for i, name in enumerate(self.kernels):
            value += np.exp(theta[2*i]) * kernel(self.geometry, name, np.exp(theta[2*i+1]))
        return value

    def fit(self, x, y, nodes):
        x, y, nodes = np.asarray(x, float), np.asarray(y, float), np.asarray(nodes, int)
        if len(x) != len(y) or len(y) != len(nodes) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("finite aligned source training records required")
        full_x = np.column_stack([np.ones(len(x)), x])
        _, r, pivot = qr(full_x, mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(r)) > np.abs(r[0, 0])*1e-7))
        self.columns = np.sort(pivot[:rank])
        x = full_x[:, self.columns]
        self.nodes, inverse, counts = np.unique(nodes, return_inverse=True, return_counts=True)
        self.counts = counts.astype(float)
        sums_x = np.zeros((len(counts), x.shape[1]))
        sums_y = np.zeros(len(counts))
        np.add.at(sums_x, inverse, x)
        np.add.at(sums_y, inverse, y)
        self.mean_x, self.mean_y = sums_x / counts[:, None], sums_y / counts
        xx = x.T @ x - self.mean_x.T @ (counts[:, None]*self.mean_x)
        xy = x.T @ y - self.mean_x.T @ (counts*self.mean_y)
        yy = float(y @ y - (counts*self.mean_y) @ self.mean_y)
        observations, features = x.shape
        self.evaluations = 0

        def profile(theta, retain=False):
            self.evaluations += 1
            noise = np.exp(theta[-1])
            covariance = self.covariance(theta)[np.ix_(self.nodes, self.nodes)]
            matrix = covariance + np.diag(noise/counts)
            try:
                factor = cho_factor(matrix, lower=True, check_finite=False)
                bx = cho_solve(factor, self.mean_x, check_finite=False)
                by = cho_solve(factor, self.mean_y, check_finite=False)
                information = xx/noise + self.mean_x.T @ bx
                info_factor = cho_factor(information, lower=True, check_finite=False)
                rhs = xy/noise + self.mean_x.T @ by
                beta = cho_solve(info_factor, rhs, check_finite=False)
                rss = yy/noise + self.mean_y @ by - rhs @ beta
                logdet = ((observations-len(counts))*np.log(noise) + np.log(counts).sum()
                          + 2*np.log(np.diag(factor[0])).sum())
                restricted_logdet = 2*np.log(np.diag(info_factor[0])).sum()
                objective = .5*(logdet + restricted_logdet + max(float(rss), 0.)
                                 + (observations-features)*np.log(2*np.pi))
            except np.linalg.LinAlgError:
                return 1e100
            if retain:
                return beta, cho_solve(factor, self.mean_y-self.mean_x @ beta, check_finite=False)
            return float(objective)

        variance = max(float(np.var(y)), .01)
        initial, bounds = [], []
        for name in self.kernels:
            distance = self.geometry["euclidean" if name == "euclidean" else "distance"][np.ix_(self.nodes, self.nodes)]
            positive = distance[np.isfinite(distance) & (distance > 0)]
            initial.extend([np.log(variance/(2*len(self.kernels))), np.log(np.median(positive) if len(positive) else 100.)])
            bounds.extend([(np.log(variance)-12, np.log(variance)+5), (np.log(.1), np.log(10000.))])
        initial.append(np.log(variance/2))
        bounds.append((np.log(variance)-12, np.log(variance)+5))
        fits = []
        for multiplier in (1., .25):
            start = np.asarray(initial).copy()
            start[1:-1:2] += np.log(multiplier)
            fits.append(minimize(profile, start, method="L-BFGS-B", bounds=bounds,
                                 options={"maxiter": self.maxiter, "ftol": 1e-8}))
        best = min(fits, key=lambda item: item.fun)
        self.theta = best.x
        self.beta, self.alpha = profile(best.x, retain=True)
        self.info = {"objective": float(best.fun), "optimizer_success": bool(best.success),
                     "optimizer_message": str(best.message), "iterations": int(best.nit),
                     "objective_evaluations": self.evaluations, "mean_rank": rank,
                     "source_stations": len(self.nodes), "source_cells": len(y),
                     "theta": best.x.tolist(), "multistarts": 2}
        return self

    def predict(self, x, nodes):
        x = np.column_stack([np.ones(len(x)), np.asarray(x, float)])[:, self.columns]
        nodes = np.asarray(nodes, int)
        correction = self.covariance(self.theta)[:, self.nodes] @ self.alpha
        return x @ self.beta + correction[nodes]


def official_physt_class(source):
    """Load only the two reviewed architecture classes, without module side effects."""
    tree = ast.parse(Path(source).read_text())
    names = {"PositionalEncoding", "ImprovedTransformerGCN"}
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != names or len(selected) != 2:
        raise ValueError("official PhyST-HC architecture definitions changed")
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F, "GCNConv": GCNConv}
    # Only reviewed, hash-frozen architecture definitions; no module top-level I/O.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace["ImprovedTransformerGCN"]


class PhySTArchitecture(nn.Module):
    """Official Transformer->GCN core, wrapped for common monthly covariates.

    E15 is repeated at every step alongside the T8 window; a zero column makes
    24 features divisible by the original two attention heads. Independent
    monthly graph panels are disjoint; no batch edges cross months.
    """

    def __init__(self, source, recipe):
        super().__init__()
        self.core = official_physt_class(source)(n_features=24, seq_len=12,
            num_heads=2, trans_layers=2, **recipe)

    def forward(self, windows, environment, season, graph):
        batch, nodes, length, _ = windows.shape
        env = environment[None, :, None].expand(batch, -1, length, -1)
        x = torch.cat([env, windows, windows.new_zeros(batch, nodes, length, 1)], dim=-1)
        downstream, upstream = torch.where(graph["relation"] == 1)
        edges = torch.stack([upstream, downstream])
        offsets = torch.arange(batch, device=x.device)*nodes
        edge_index = (edges[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        data = SimpleNamespace(x=x.reshape(batch*nodes, length*24), edge_index=edge_index)
        return self.core(data).reshape(batch, nodes)
