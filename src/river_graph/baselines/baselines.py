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


def ecological_tabular_features(
    dataset: dict, split: dict[str, np.ndarray],
    visibility: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """(N*T, 23) ecological feature set for the Phase-2 tabular arms.

    Frozen definition (docs/paper/phase2_ablation_spec.md §4): standardized
    dynamic hydro (temp, temp availability, discharge, discharge availability),
    month sin/cos, lat/lon, the 13 regime columns, and two graph-free
    aggregates of the month-t visible-DOC field (self-excluded mean and count
    across stations). A cell's own DOC is never a feature of its own row, and
    no edge, neighbour identity or distance information is included.

    ``visibility`` selects which split cells' labels may enter the DOC
    aggregates (keys of ``split`` among ``train``/``val``/``context``). This
    stages label exposure: during fitting and early stopping use
    ``{"train", "context"}`` so no validation label is visible; only the final
    test-time view uses ``{"train", "val", "context"}`` per protocol. Passing
    ``None`` gives the test-time view.
    """
    y, x, months, latlon, n, t, _tm, _obs = _ctx(dataset, split)
    x_mask = dataset["x_mask"].numpy()
    regime = dataset["regime"].numpy()
    if regime.shape[1] != 13:
        raise ValueError(f"expected 13 regime columns, got {regime.shape[1]}")

    vis = {"train", "val", "context"} if visibility is None else set(visibility)
    bad = vis - {"train", "val", "context"}
    if bad:
        raise ValueError(f"unknown visibility keys: {sorted(bad)}")
    obs_mask = np.zeros(n * t, dtype=bool)
    for key in ("train", "val", "context"):
        if key in vis and key in split and len(split[key]):
            obs_mask[np.asarray(split[key], dtype=np.int64)] = True
    obs_mask = obs_mask.reshape(n, t)

    month_num = months.astype("datetime64[M]").astype(int) % 12
    sincos = np.stack(
        [np.sin(2 * np.pi * month_num / 12), np.cos(2 * np.pi * month_num / 12)],
        axis=1,
    )  # (T, 2)
    idx = np.arange(n * t)
    i, j = idx // t, idx % t

    # visible-DOC field in log1p mg/L (_ctx already log1p-transformed y)
    doc_field = np.where(obs_mask, y, 0.0)  # (N, T)
    month_sum = doc_field.sum(axis=0)  # (T,)
    month_cnt = obs_mask.sum(axis=0).astype(float)  # (T,)
    own_doc = doc_field[i, j]
    own_vis = obs_mask[i, j].astype(float)
    others_cnt = month_cnt[j] - own_vis
    others_mean = np.where(
        others_cnt > 0, (month_sum[j] - own_doc) / np.maximum(others_cnt, 1.0), 0.0
    )

    names = (
        ["temp", "temp_avail", "discharge", "discharge_avail",
         "month_sin", "month_cos", "lat", "lon"]
        + [f"regime_{k}" for k in range(13)]
        + ["month_visdoc_mean_excl_self", "month_visdoc_count_excl_self"]
    )
    feats = np.column_stack(
        [
            x[i, j, 0], x_mask[i, j, 0].astype(float),
            x[i, j, 1], x_mask[i, j, 1].astype(float),
            sincos[j], latlon[i],
            regime[i],
            others_mean, others_cnt,
        ]
    )
    return feats, names


FIT_VISIBILITY = frozenset({"train", "context"})
TEST_VISIBILITY = frozenset({"train", "val", "context"})


def _staged_features(dataset: dict, split: dict[str, np.ndarray]):
    """(fit-view, test-view) feature matrices; val labels hidden in fit-view."""
    fit_feats, names = ecological_tabular_features(
        dataset, split, visibility=FIT_VISIBILITY
    )
    test_feats, _ = ecological_tabular_features(
        dataset, split, visibility=TEST_VISIBILITY
    )
    return fit_feats, test_feats, names


class EcoRandomForest:
    """Phase-2 ecological RF arm (spec §3-4): fit on train cells only.

    Fit rows see only train+context labels in the DOC aggregates (validation
    labels are hidden); test rows are predicted under the protocol's test-time
    visibility (train+val+context).
    """

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit_predict(self, dataset: dict, split: dict[str, np.ndarray]) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor

        y, _x, _m, _ll, n, t, _tm, _obs = _ctx(dataset, split)
        fit_feats, test_feats, names = _staged_features(dataset, split)
        self.feature_names_ = names
        tr, te = np.asarray(split["train"]), np.asarray(split["test"])
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=self.seed
        )
        rf.fit(fit_feats[tr], y.ravel()[tr])
        self.early_stop_ = None  # RF has no early stopping
        out = np.full(n * t, np.nan)
        out[te] = rf.predict(test_feats[te])
        return np.expm1(out.reshape(n, t))


class EcoMLP:
    """Phase-2 ecological MLP arm (spec §3-4).

    Same train/val logic as the GNN arms: fit rows are the train cells, and
    early stopping watches the frozen val cells with the same patience rule
    (min_delta=1e-6) — never a random carve of train. Validation labels are
    hidden from every input during fitting and early stopping (fit-view
    features); only the final test-time prediction uses the test-view.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        max_epochs: int = 300,
        patience: int = 20,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit_predict(self, dataset: dict, split: dict[str, np.ndarray]) -> np.ndarray:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        y, _x, _m, _ll, n, t, _tm, _obs = _ctx(dataset, split)
        fit_feats, test_feats, names = _staged_features(dataset, split)
        self.feature_names_ = names
        tr = np.asarray(split["train"])
        va = np.asarray(split.get("val", []), dtype=np.int64)
        te = np.asarray(split["test"])
        y_flat = y.ravel()

        scaler = StandardScaler().fit(fit_feats[tr])
        x_tr = scaler.transform(fit_feats[tr])
        x_va = scaler.transform(fit_feats[va]) if len(va) else None
        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden, random_state=self.seed
        )
        best_state, best_loss, bad, best_epoch = None, float("inf"), 0, -1
        val_losses: list[float] = []
        epochs_run = 0
        for epoch in range(self.max_epochs):
            mlp.partial_fit(x_tr, y_flat[tr])  # one iteration
            epochs_run = epoch + 1
            if x_va is None:
                continue
            vloss = float(np.mean((mlp.predict(x_va) - y_flat[va]) ** 2))
            val_losses.append(vloss)
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                best_epoch = epoch
                best_state = (
                    [c.copy() for c in mlp.coefs_],
                    [b.copy() for b in mlp.intercepts_],
                )
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            mlp.coefs_, mlp.intercepts_ = best_state
        self.early_stop_ = {
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss if np.isfinite(best_loss) else None,
            "val_losses": val_losses,
            "visibility_fit": sorted(FIT_VISIBILITY),
            "visibility_test": sorted(TEST_VISIBILITY),
        }
        out = np.full(n * t, np.nan)
        out[te] = mlp.predict(scaler.transform(test_feats[te]))
        return np.expm1(out.reshape(n, t))
