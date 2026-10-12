"""DOC literature motivated baselines, not complete paper reproductions.

The satellite optical water type pipeline of Aqua-OC is not present here.
The GAM uses local monthly discharge rather than river-mouth campaign flow,
and pyGAM penalized splines rather than the paper's R mgcv implementation.
"""
from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from river_graph.experiments.doc_fusion_benchmark import tabular_features

FAMILIES = ("doc_ecological_rf", "doc_hydro_gam", "doc_xgboost", "doc_svr")
MODELS = (*FAMILIES, "rf_history_control", "fixed_fusion")
DETERMINISTIC = ("doc_hydro_gam", "doc_svr")
CANDIDATES = {
    "doc_ecological_rf": {
        "native_leaf3": {"target": "native", "min_samples_leaf": 3},
        "log_leaf3": {"target": "log1p", "min_samples_leaf": 3},
        "log_leaf10": {"target": "log1p", "min_samples_leaf": 10},
    },
    "doc_hydro_gam": {
        "native_lam10": {"target": "native", "lam": 10.},
        "log_lam1": {"target": "log1p", "lam": 1.},
        "log_lam10": {"target": "log1p", "lam": 10.},
    },
    "doc_xgboost": {
        "native_l1_depth3": {"target": "native", "objective": "reg:absoluteerror", "max_depth": 3},
        "log_l2_depth3": {"target": "log1p", "objective": "reg:squarederror", "max_depth": 3},
        "log_l2_depth6": {"target": "log1p", "objective": "reg:squarederror", "max_depth": 6},
    },
    "doc_svr": {
        "native_c10": {"target": "native", "C": 10., "epsilon": .2},
        "log_c1": {"target": "log1p", "C": 1., "epsilon": .1},
        "log_c10": {"target": "log1p", "C": 10., "epsilon": .1},
    },
}


def optional_dependencies():
    """Use a cache-local pinned package set, without changing the project venv."""
    root = Path(__file__).resolve().parents[3]
    path = root / "cache/doc_domain_work_v1/python_deps"
    if not path.is_dir():
        raise ImportError("isolated DOC baseline dependencies are unavailable")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    # Reuse the existing torch OpenMP runtime; no system installation is needed.
    if sys.platform == "darwin":
        library = Path(torch.__file__).parent / "lib/libomp.dylib"
        if library.exists():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return path


def features(inputs, cells, months, family):
    if family not in FAMILIES:
        raise ValueError("undeclared DOC baseline family")
    if family in ("doc_xgboost", "doc_svr"):
        return tabular_features(inputs, cells, months)
    cells = np.asarray(cells, dtype=np.int64)
    node, month = cells // months, cells % months
    result = np.column_stack([inputs["environment"].numpy()[node],
                              inputs["windows"].numpy()[month, node, -1]])
    if result.shape != (len(cells), 23) or not np.isfinite(result).all():
        raise ValueError("finite 23-column ecological features required")
    return result


def make_estimator(family, recipe, seed, threads):
    settings = {key: value for key, value in recipe.items() if key != "target"}
    if family == "doc_ecological_rf":
        return RandomForestRegressor(n_estimators=300, max_features=1.,
            criterion="squared_error", random_state=seed, n_jobs=threads, **settings)
    if family == "doc_svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", gamma="scale",
            cache_size=512, tol=.001, max_iter=-1, **settings))
    optional_dependencies()
    if family == "doc_xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=200, learning_rate=.05, tree_method="hist",
            subsample=.8, colsample_bytree=1., min_child_weight=5, reg_lambda=1.,
            reg_alpha=0., random_state=seed, n_jobs=threads, **settings)
    if family == "doc_hydro_gam":
        from pygam import LinearGAM, l, s, te

        # 15 environmental columns, T/Q, masks/ages and seasonal sin/cos.
        terms = l(3)
        for index in (*range(3), *range(4, 17)):
            terms += s(index, n_splines=6, spline_order=3)
        for index in range(17, 23):
            terms += l(index)
        terms += te(1, 16, n_splines=[5, 5], spline_order=[3, 3])
        return LinearGAM(terms, max_iter=100, tol=1e-4, **settings)
    raise ValueError("undeclared DOC baseline family")


def dependency_versions():
    optional_dependencies()
    return {name: importlib.import_module(name).__version__
            for name in ("pygam", "xgboost", "scipy")}
