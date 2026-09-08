"""Baselines B0-B3 for the DOC reconstruction benchmark (design.md).

All models implement fit_predict(dataset, split) -> (N, T) ndarray of
predictions in mg/L space (NaN where not predicted). `split` carries flat
C-order cell indices: train / val / test.

- B0 StationMean: per-station mean of train cells (global train mean fallback)
- B1 Kriging: per-month ordinary kriging over stations observed that month
- B2 RandomForest: temp, discharge, month-of-year, lat, lon
- B3 MLP: same features, standardized
"""

from __future__ import annotations

import warnings

import numpy as np


def _ctx(dataset: dict, split: dict[str, np.ndarray]):
    """Shared context: (N, T) arrays, flat-index helpers.

    The dataset stores y in raw mg/L; models work in log1p space per
    design.md, and their fit_predict must return mg/L (expm1 applied).

    train_mask marks cells usable for *fitting*; obs_mask additionally
    includes val and (E2-b) context cells — everything a model may *see*
    as observations at inference time.
    """
    y = np.log1p(dataset["y"].numpy())  # log1p mg/L
    x = dataset["x"].numpy()  # (N, T, F)
    months = np.array(dataset["months"], dtype="datetime64[M]")
    latlon = dataset["static"].numpy()  # (N, 2)
    n, t = y.shape
    train_mask = np.zeros(n * t, dtype=bool)
    train_mask[split["train"]] = True
    train_mask = train_mask.reshape(n, t)
    obs_mask = train_mask.copy()
    for key in ("val", "context"):
        if key in split and len(split[key]):
            obs_mask.ravel()[split[key]] = True
    return y, x, months, latlon, n, t, train_mask, obs_mask


class StationMean:
    """B0: per-station mean of train cells; global train mean as fallback."""

    def fit_predict(self, dataset, split) -> np.ndarray:
        y, _x, _m, _ll, n, t, train_mask, _obs = _ctx(dataset, split)
        pred = np.full((n, t), np.nan)
        ytr = np.where(train_mask, y, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            station_mean = np.nanmean(ytr, axis=1)
            global_mean = np.nanmean(ytr)
        fill = np.where(np.isnan(station_mean), global_mean, station_mean)
        pred[:] = fill[:, None]
        return np.expm1(pred)


class Kriging:
    """B1: ordinary kriging over stations observed (in train) each month."""

    def __init__(self, min_obs: int = 5):
        self.min_obs = min_obs

    def fit_predict(self, dataset, split) -> np.ndarray:
        from pykrige.ok import OrdinaryKriging

        y, _x, _m, latlon, n, t, train_mask, obs_mask = _ctx(dataset, split)
        pred = np.full((n, t), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            train_station_mean = np.nanmean(np.where(train_mask, y, np.nan), axis=1)
            global_mean = np.nanmean(np.where(train_mask, y, np.nan))
        # deterministic jitter so co-located stations cannot produce a
        # singular kriging system
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 1e-4, latlon.shape)
        ll = latlon + jitter
        for j in range(t):
            obs = obs_mask[:, j]
            if obs.sum() < self.min_obs:
                continue
            try:
                okr = OrdinaryKriging(
                    ll[obs, 1], ll[obs, 0], y[obs, j],
                    variogram_model="linear", verbose=False, enable_plotting=False,
                )
                z, _ = okr.execute("points", ll[:, 1], ll[:, 0])
                col = np.asarray(z.filled(np.nan) if hasattr(z, "filled") else z)
            except Exception:  # noqa: BLE001 - singular systems, convergence, ...
                col = np.full(n, np.nan)
            # fallback for stations kriging could not predict
            fb = np.where(np.isnan(train_station_mean), global_mean, train_station_mean)
            pred[:, j] = np.where(np.isnan(col), fb, col)
        return np.expm1(pred)


def _tabular_features(x, months, latlon, n, t):
    """(N*T, 6): temp, discharge, month sin/cos, lat, lon."""
    month_num = months.astype("datetime64[M]").astype(int) % 12  # 0..11
    sincos = np.stack(
        [np.sin(2 * np.pi * month_num / 12), np.cos(2 * np.pi * month_num / 12)], axis=1
    )  # (T, 2)
    idx = np.arange(n * t)
    i, j = idx // t, idx % t
    feats = np.column_stack(
        [x.reshape(n * t, -1), sincos[j], latlon[i]]
    )
    return feats


class RandomForest:
    """B2: RF on tabular features."""

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit_predict(self, dataset, split) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer

        y, x, months, latlon, n, t, _tm, _obs = _ctx(dataset, split)
        feats = _tabular_features(x, months, latlon, n, t)
        tr = split["train"]
        imp = SimpleImputer(strategy="median").fit(feats[tr])
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=self.seed
        )
        rf.fit(imp.transform(feats[tr]), y.ravel()[tr])
        pred = rf.predict(imp.transform(feats[split["test"]]))
        out = np.full(n * t, np.nan)
        out[split["test"]] = pred
        return np.expm1(out.reshape(n, t))


class MLP:
    """B3: sklearn MLP on standardized tabular features."""

    def __init__(self, hidden=(128, 64), max_iter: int = 300, seed: int = 0):
        self.hidden = hidden
        self.max_iter = max_iter
        self.seed = seed

    def fit_predict(self, dataset, split) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y, x, months, latlon, n, t, _tm, _obs = _ctx(dataset, split)
        feats = _tabular_features(x, months, latlon, n, t)
        tr = split["train"]
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self.hidden,
                max_iter=self.max_iter,
                early_stopping=True,
                random_state=self.seed,
            ),
        )
        pipe.fit(feats[tr], y.ravel()[tr])
        pred = pipe.predict(feats[split["test"]])
        out = np.full(n * t, np.nan)
        out[split["test"]] = pred
        return np.expm1(out.reshape(n, t))
