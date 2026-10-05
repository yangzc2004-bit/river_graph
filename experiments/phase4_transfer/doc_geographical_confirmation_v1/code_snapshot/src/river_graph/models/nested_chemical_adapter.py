"""Chemical support shape after a completed legacy station calibration.

The parent prediction includes its already selected support correction and
ecological mix. This operator does not select those choices again. It fits two
chemical coordinates to the residual left on the support rows, without adding
another station intercept. One source-validation rule is shared by K=3 and 5.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

K_VALUES = (0, 1, 3, 5)
SELECTION_K_VALUES = (3, 5)
RIDGE_VALUES = (.1, 1., 10., 100.)
STRENGTH_VALUES = (0., .25, .5, 1.)


@dataclass(frozen=True)
class NestedChemicalEpisode:
    """Source-validation task with completed legacy support/query predictions.

Cells are flattened station-major indices. Chemical arrays have [rows, 2]
shape and come from a source-fitted ChemicalSupportBasis. Query values belong
only to source validation; target adaptation never accepts query values.
"""

    k: int
    query_cells: np.ndarray
    query_values: np.ndarray
    query_prediction: np.ndarray
    support_cells: np.ndarray
    support_values: np.ndarray
    support_prediction: np.ndarray
    query_chemical: np.ndarray
    support_chemical: np.ndarray
    query_active: np.ndarray
    support_active: np.ndarray


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _values(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite nonnegative one-dimensional values")
    return array


def _active(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isin(array, (False, True)).all():
        raise ValueError(f"{name} must be aligned binary availability flags")
    return array.astype(bool)


def _coordinates(value, active, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (*active.shape, 2) or not np.isfinite(array[active]).all():
        raise ValueError(f"{name} must be aligned chemical2 with finite active rows")
    # Unavailable placeholders have no role in normalization or adaptation.
    return np.where(active[..., None], array, 0.)


class NestedChemicalAdapter:
    """Separately regularized chemical shape correction in log1p DOC space.

    API order:

    1. ``fit_coordinates(source_chemical, source_active,
       source_role='source_training')`` freezes source-global center/scale.
    2. ``fit(validation_episodes, selection_role='source_validation')`` chooses
       a single ridge/strength using K-equal, then station-equal native MAE.
    3. ``adapt`` receives only target support DOC and completed legacy
       predictions. Its signature deliberately contains no query DOC truth.

    Within each station, active support chemical coordinates and remaining
    log1p residuals are centered. Ridge minimizes mean squared centered
    residual plus ``ridge_strength * ||coefficient||²``. Strength scales only
    the chemical increment. No parent intercept, ridge, or ecological gamma
    changes. K0/K1, zero strength, fewer than two active support cells, and
    unavailable query chemistry preserve the parent exactly.

    Leave-station-fold-out selection summaries assess this adapter conditional
    on the frozen parent. They are not fully OOF performance estimates because
    the parent may already have used all source validation for selection.
    """

    def __init__(self, n_months, *, ridge_values=RIDGE_VALUES,
                 strength_values=STRENGTH_VALUES, scale_floor=1e-8, selection_folds=5, fold_seed=42):
        self.n_months = _positive_integer(n_months, "n_months")
        self.selection_folds = _positive_integer(selection_folds, "selection_folds")
        if self.selection_folds < 2:
            raise ValueError("selection_folds must be at least two")
        if (isinstance(fold_seed, (bool, np.bool_)) or not isinstance(fold_seed, (int, np.integer))
                or fold_seed < 0):
            raise ValueError("fold_seed must be a nonnegative integer")
        self.fold_seed = int(fold_seed)
        ridges = np.asarray(tuple(ridge_values), dtype=np.float64)
        strengths = np.asarray(tuple(strength_values), dtype=np.float64)
        if (ridges.ndim != 1 or not len(ridges) or not np.isfinite(ridges).all()
                or (ridges <= 0).any()):
            raise ValueError("ridge_values must be finite positive candidates")
        if (strengths.ndim != 1 or not len(strengths) or not np.isfinite(strengths).all()
                or ((strengths < 0) | (strengths > 1)).any() or not (strengths == 0).any()):
            raise ValueError("strength_values must lie in [0,1] and include exact zero")
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")
        self.ridge_values = tuple(float(x) for x in np.unique(ridges))
        self.strength_values = tuple(float(x) for x in np.unique(strengths))
        self.scale_floor = float(scale_floor)
        self.coordinates_fitted_ = False
        self.fitted_ = False

    def fit_coordinates(self, source_chemical, source_active, *, source_role):
        """Freeze global coordinate moments from explicit source-training rows.

        Arrays may be [row,2]/[row] or [station,month,2]/[station,month]. No DOC
        values are accepted. Target/source-validation rows must be sliced out
        by the caller before invoking this method.
        """
        if source_role != "source_training":
            raise ValueError("coordinate fitting requires source_training rows")
        chemical = np.asarray(source_chemical, dtype=np.float64)
        if chemical.ndim not in (2, 3) or chemical.shape[-1] != 2 or min(chemical.shape[:-1]) < 1:
            raise ValueError("source_chemical must be nonempty [row,2] or [station,month,2]")
        active = _active(source_active, chemical.shape[:-1], "source_active")
        chemical = _coordinates(chemical, active, "source_chemical")
        rows = chemical[active]
        self.n_source_rows_ = int(active.size)
        self.n_source_active_rows_ = int(active.sum())
        self.coordinate_mean_ = rows.mean(0) if len(rows) else np.zeros(2)
        self.coordinate_std_ = rows.std(0) if len(rows) else np.zeros(2)
        if not np.isfinite(self.coordinate_mean_).all() or not np.isfinite(self.coordinate_std_).all():
            raise FloatingPointError("source coordinate moments are not finite")
        self.identified_ = self.coordinate_std_ > self.scale_floor
        self.coordinate_scale_ = np.maximum(self.coordinate_std_, self.scale_floor)
        self.source_role_ = source_role
        self.coordinates_fitted_, self.fitted_ = True, False
        return self

    def _standardize(self, chemical):
        if not self.coordinates_fitted_:
            raise RuntimeError("fit source coordinates before chemical adaptation")
        result = (chemical - self.coordinate_mean_) / self.coordinate_scale_
        result[..., ~self.identified_] = 0.
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite source-standardized chemical coordinates")
        return result

    def _prepare(self, query_prediction, query_cells, support_prediction, support_cells,
                 support_values, query_chemical, support_chemical, query_active, support_active, k):
        if isinstance(k, (bool, np.bool_)) or k not in K_VALUES:
            raise ValueError("K must be 0, 1, 3 or 5")
        raw_prediction = np.asarray(query_prediction)
        prediction = _values(query_prediction, "query_prediction")
        sp = _values(support_prediction, "support_prediction")
        sv = _values(support_values, "support_values")
        indices = []
        for value in (query_cells, support_cells):
            raw = np.asarray(value)
            if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
                raise ValueError("cell identities must be one-dimensional integers")
            cells = raw.astype(np.int64)
            if (cells < 0).any() or len(np.unique(cells)) != len(cells):
                raise ValueError("cell identities must be nonnegative and unique")
            indices.append(cells)
        query, support = indices
        if prediction.shape != query.shape or sp.shape != support.shape or sv.shape != support.shape:
            raise ValueError("predictions and support values must align with cell identities")
        if np.intersect1d(query, support).size:
            raise ValueError("support and query cells must be disjoint")
        qa = _active(query_active, query.shape, "query_active")
        sa = _active(support_active, support.shape, "support_active")
        qc = self._standardize(_coordinates(query_chemical, qa, "query_chemical"))
        sc = self._standardize(_coordinates(support_chemical, sa, "support_chemical"))
        stations, counts = np.unique(support // self.n_months, return_counts=True)
        if k == 0:
            if len(support):
                raise ValueError("K=0 requires empty support")
        elif (not np.array_equal(stations, np.unique(query // self.n_months)) or not (counts == k).all()):
            raise ValueError("every query station must have exactly K support cells")
        return raw_prediction, prediction, query, sp, support, sv, qc, sc, qa, sa, int(k)

    def _apply(self, prepared, *, ridge_strength, strength):
        original, prediction, query, sp, support, sv, qc, sc, qa, sa, k = prepared
        increment = np.zeros(len(query), dtype=np.float64)
        count = np.zeros(len(query), dtype=np.int64)
        coefficients = np.zeros((len(query), 2), dtype=np.float64)
        for station in np.unique(query // self.n_months):
            q = query // self.n_months == station
            count[q] = int(((support // self.n_months == station) & sa).sum())
        if k in (0, 1) or strength == 0:
            return {"y_pred": original.copy(), "chemical_delta": increment,
                    "chemical_support_count": count, "chemical_coefficients": coefficients}
        residual = np.log1p(sv) - np.log1p(sp)
        for station in np.unique(query // self.n_months):
            q = query // self.n_months == station
            s = (support // self.n_months == station) & sa
            if s.sum() < 2 or not qa[q].any():
                continue
            center = sc[s].mean(0)
            centered = sc[s] - center
            response = residual[s] - residual[s].mean()
            gram = centered.T @ centered / s.sum() + ridge_strength * np.eye(2)
            rhs = centered.T @ response / s.sum()
            coefficient = np.linalg.solve(gram, rhs)
            selected = q & qa
            coefficients[selected] = coefficient
            increment[selected] = strength * ((qc[selected] - center) @ coefficient)
        if not np.isfinite(increment).all():
            raise FloatingPointError("nonfinite nested chemical increment")
        # Only genuinely modified rows take a log/inverse round trip. Missing
        # chemistry and zero-shape rows retain exact parent values and dtype.
        modified = increment != 0.
        output = original.copy() if original.dtype.kind == "f" else prediction.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            output[modified] = np.maximum(np.expm1(np.log1p(prediction[modified]) + increment[modified]), 0.)
        if not np.isfinite(output).all():
            raise FloatingPointError("nested chemical correction produced nonfinite DOC")
        return {"y_pred": output, "chemical_delta": increment,
                "chemical_support_count": count, "chemical_coefficients": coefficients}

    def fit(self, episodes, *, selection_role):
        """Select one chemical rule across K3/K5; retain all candidate/fold scores."""
        if selection_role != "source_validation":
            raise ValueError("selection requires source_validation episodes")
        if not self.coordinates_fitted_:
            raise RuntimeError("fit source coordinates before source-validation selection")
        tasks = []
        for episode in episodes:
            if not isinstance(episode, NestedChemicalEpisode) or episode.k not in SELECTION_K_VALUES:
                raise ValueError("selection requires NestedChemicalEpisode tasks for K3 and K5")
            prepared = self._prepare(episode.query_prediction, episode.query_cells,
                episode.support_prediction, episode.support_cells, episode.support_values,
                episode.query_chemical, episode.support_chemical, episode.query_active,
                episode.support_active, episode.k)
            truth = _values(episode.query_values, "source-validation query values")
            if not len(truth) or truth.shape != prepared[1].shape:
                raise ValueError("source-validation query values must be aligned and nonempty")
            tasks.append((prepared, truth))
        station_sets = {k: set(np.concatenate([task[0][2] // self.n_months for task in tasks
                       if task[0][-1] == k]).tolist()) for k in SELECTION_K_VALUES
                        if any(task[0][-1] == k for task in tasks)}
        if (set(station_sets) != set(SELECTION_K_VALUES)
                or station_sets[3] != station_sets[5]):
            raise ValueError("selection requires the same validation stations at both K3 and K5")
        # Nested K comparisons use a fixed query set. Repeated task seeds may
        # repeat rows, but every budget must expose the same union of queries.
        cells_by_k = {k: np.unique(np.concatenate([task[0][2] for task in tasks if task[0][-1] == k]))
                      for k in SELECTION_K_VALUES}
        if not np.array_equal(cells_by_k[3], cells_by_k[5]):
            raise ValueError("K3/K5 validation query cells must be fixed")
        stations = sorted(station_sets[3])
        candidates, all_losses = [], []
        for strength in self.strength_values:
            for ridge in self.ridge_values:
                errors = {(k, station): [0., 0] for k in SELECTION_K_VALUES for station in stations}
                valid = True
                for prepared, truth in tasks:
                    try:
                        output = self._apply(prepared, ridge_strength=ridge, strength=strength)["y_pred"]
                    except (FloatingPointError, np.linalg.LinAlgError):
                        valid = False
                        break
                    error = np.abs(output - truth)
                    for station in np.unique(prepared[2] // self.n_months):
                        selected = prepared[2] // self.n_months == station
                        target = errors[(prepared[-1], int(station))]
                        target[0] += float(error[selected].sum())
                        target[1] += int(selected.sum())
                losses = {key: total / count for key, (total, count) in errors.items()} if valid else None
                by_k = ({str(k): float(np.mean([losses[(k, station)] for station in stations]))
                         for k in SELECTION_K_VALUES} if valid else None)
                mae = float(np.mean(list(by_k.values()))) if valid else None
                if valid and not np.isfinite(mae):
                    valid, mae, by_k, losses = False, None, None, None
                candidates.append({"strength": strength, "ridge_strength": ridge, "mae": mae,
                    "mae_by_k": by_k, "valid": valid, "n_stations": len(stations),
                    "n_query_occurrences_by_k": {str(k): sum(len(truth) for prepared, truth in tasks
                                                           if prepared[-1] == k) for k in SELECTION_K_VALUES}})
                all_losses.append(losses)
        viable = [index for index, row in enumerate(candidates) if row["valid"]]
        if not viable:
            raise FloatingPointError("no finite source-validation chemical candidate")

        def best(indices, selected_stations):
            return min(indices, key=lambda index: (
                np.mean([all_losses[index][(k, station)] for k in SELECTION_K_VALUES
                         for station in selected_stations]),
                candidates[index]["strength"], -candidates[index]["ridge_strength"]))

        selected = min(viable, key=lambda index: (candidates[index]["mae"],
                       candidates[index]["strength"], -candidates[index]["ridge_strength"]))
        folds = []
        n_folds = min(self.selection_folds, len(stations))
        shuffled = np.random.default_rng(self.fold_seed).permutation(stations).tolist()
        if n_folds >= 2:
            for fold in range(n_folds):
                held = sorted(shuffled[fold::n_folds])
                training = [station for station in stations if station not in held]
                choice = best(viable, training)
                fold_candidates = []
                for index in viable:
                    fold_candidates.append({"strength": candidates[index]["strength"],
                        "ridge_strength": candidates[index]["ridge_strength"],
                        "selection_mae": float(np.mean([all_losses[index][(k, station)]
                            for k in SELECTION_K_VALUES for station in training])),
                        "held_mae_by_k": {str(k): float(np.mean([all_losses[index][(k, station)]
                            for station in held])) for k in SELECTION_K_VALUES}})
                held_by_k = {str(k): float(np.mean([all_losses[choice][(k, station)]
                            for station in held])) for k in SELECTION_K_VALUES}
                selection_by_k = {str(k): float(np.mean([all_losses[choice][(k, station)]
                            for station in training])) for k in SELECTION_K_VALUES}
                folds.append({"fold": fold, "selection_stations": training, "held_stations": held,
                    "strength": candidates[choice]["strength"],
                    "ridge_strength": candidates[choice]["ridge_strength"],
                    "choice": {"strength": candidates[choice]["strength"],
                               "ridge_strength": candidates[choice]["ridge_strength"]},
                    "selected_zero": candidates[choice]["strength"] == 0,
                    "selection_mae": float(np.mean(list(selection_by_k.values()))),
                    "selection_mae_by_k": selection_by_k,
                    "held_mae": float(np.mean(list(held_by_k.values()))),
                    "held_mae_by_k": held_by_k,
                    "candidate_scores": fold_candidates})
        self.selection_ = {"strength": candidates[selected]["strength"],
                           "ridge_strength": candidates[selected]["ridge_strength"]}
        self.selection_scores_ = candidates
        self.fold_diagnostics_ = folds
        self.validation_stations_ = stations
        self.selection_role_, self.fitted_ = selection_role, True
        return self

    def adapt_components(self, query_prediction, query_cells, support_prediction, support_cells,
                         support_values, *, query_chemical, support_chemical,
                         query_active, support_active, k, ridge_strength=None, strength=None):
        """Return predictions and chemical-only diagnostics, without query truth.

        An explicit candidate ``ridge_strength``/``strength`` pair allows saved
        validation-fold choices to be replayed without another fit. Omitting
        both uses the shared selected rule. Overrides cannot introduce a new
        candidate or per-row parameter.
        """
        if not self.fitted_:
            raise RuntimeError("fit or restore the nested chemical rule before prediction")
        prepared = self._prepare(query_prediction, query_cells, support_prediction, support_cells,
            support_values, query_chemical, support_chemical, query_active, support_active, k)
        if (ridge_strength is None) != (strength is None):
            raise ValueError("supply both ridge_strength and strength for a saved candidate override")
        choice = self.selection_ if ridge_strength is None else {
            "ridge_strength": ridge_strength, "strength": strength}
        if choice["ridge_strength"] not in self.ridge_values or choice["strength"] not in self.strength_values:
            raise ValueError("override must be a fixed configured chemical candidate")
        return self._apply(prepared, **choice)

    def adapt(self, query_prediction, query_cells, support_prediction, support_cells,
              support_values, *, query_chemical, support_chemical, query_active, support_active, k):
        """Apply the frozen shared K3/K5 rule to support labels only."""
        return self.adapt_components(query_prediction, query_cells, support_prediction, support_cells,
            support_values, query_chemical=query_chemical, support_chemical=support_chemical,
            query_active=query_active, support_active=support_active, k=k)["y_pred"]

    def to_dict(self):
        """JSON-safe complete rule; support-specific coefficients are recomputed."""
        if not self.fitted_:
            raise RuntimeError("fit before nested chemical state serialization")
        return deepcopy({"model_class": "NestedChemicalAdapter", "schema_version": 1,
            "n_months": self.n_months, "ridge_values": list(self.ridge_values),
            "strength_values": list(self.strength_values), "scale_floor": self.scale_floor,
            "selection_folds": self.selection_folds, "fold_seed": self.fold_seed, "source_role": self.source_role_,
            "n_source_rows": self.n_source_rows_, "n_source_active_rows": self.n_source_active_rows_,
            "coordinate_mean": self.coordinate_mean_.tolist(), "coordinate_std": self.coordinate_std_.tolist(),
            "coordinate_scale": self.coordinate_scale_.tolist(), "identified": self.identified_.tolist(),
            "selection_role": self.selection_role_, "selection": self.selection_,
            "selection_scores": self.selection_scores_, "validation_stations": self.validation_stations_,
            "fold_diagnostics": self.fold_diagnostics_, "definition": {
                "input_prediction": "completed frozen legacy support calibration and ecological mixing",
                "response": "centered log1p support truth minus log1p completed legacy support prediction",
                "coordinate_scaling": "global mean/std from active source-training chemical2 rows only",
                "fit": "mean centered support squared error plus separate ridge times coefficient norm squared",
                "increment": "shared strength times support-centered chemical2 dot coefficient; no intercept",
                "selection": "one native MAE rule shared across K3/K5; K equal, then station equal",
                "tie_rule": "lower strength (exact zero first), then stronger chemical ridge",
                "locked_parent": "parent decoder, alpha, ridge and ecological gamma are never reselected",
                "fallback": "exact parent at K0/K1, zero strength, fewer than two active supports or absent query chemistry",
                "fold_interpretation": "conditional adapter selection diagnostics; not fully OOF parent generalization",
                "fold_assignment": "seeded shuffle of sorted validation station IDs then round-robin; all K held together",
                "target_query_truth": "not accepted by adaptation"}})

    @classmethod
    def from_dict(cls, state):
        """Restore normalization, the shared rule, and selection diagnostics."""
        state = deepcopy(state)
        if (state.get("model_class") != "NestedChemicalAdapter" or state.get("schema_version") != 1
                or state.get("source_role") != "source_training"
                or state.get("selection_role") != "source_validation"):
            raise ValueError("unsupported nested chemical state or source roles")
        obj = cls(state["n_months"], ridge_values=state["ridge_values"],
                  strength_values=state["strength_values"], scale_floor=state["scale_floor"],
                  selection_folds=state["selection_folds"], fold_seed=state["fold_seed"])
        for attribute, key in (("coordinate_mean_", "coordinate_mean"),
                               ("coordinate_std_", "coordinate_std"),
                               ("coordinate_scale_", "coordinate_scale")):
            value = np.asarray(state[key], dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"invalid saved {key}")
            setattr(obj, attribute, value)
        obj.n_source_rows_ = _positive_integer(state["n_source_rows"], "n_source_rows")
        count = state["n_source_active_rows"]
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not 0 <= count <= obj.n_source_rows_ or (obj.coordinate_std_ < 0).any()
                or not np.array_equal(obj.coordinate_scale_, np.maximum(obj.coordinate_std_, obj.scale_floor))):
            raise ValueError("invalid saved source coordinate scale/counts")
        obj.n_source_active_rows_ = int(count)
        obj.identified_ = obj.coordinate_std_ > obj.scale_floor
        if not np.array_equal(np.asarray(state["identified"]), obj.identified_):
            raise ValueError("invalid saved identified coordinates")
        scores = state["selection_scores"]
        if (not isinstance(scores, list) or len(scores) != len(obj.ridge_values) * len(obj.strength_values)
                or {(row["strength"], row["ridge_strength"]) for row in scores}
                != {(strength, ridge) for strength in obj.strength_values for ridge in obj.ridge_values}):
            raise ValueError("invalid saved candidate scores")
        for row in scores:
            if row["valid"]:
                by_k = row["mae_by_k"]
                if (set(by_k) != {"3", "5"} or not np.isfinite(list(by_k.values())).all()
                        or min(by_k.values()) < 0 or not np.isfinite(row["mae"])
                        or row["mae"] != float(np.mean(list(by_k.values())))):
                    raise ValueError("invalid saved shared candidate score")
            elif row["mae"] is not None or row["mae_by_k"] is not None:
                raise ValueError("invalid failed candidate score")
        valid = [row for row in scores if row["valid"]]
        if not valid:
            raise ValueError("saved rule has no finite candidate")
        winner = min(valid, key=lambda row: (row["mae"], row["strength"], -row["ridge_strength"]))
        choice = {key: winner[key] for key in ("strength", "ridge_strength")}
        if state["selection"] != choice:
            raise ValueError("saved selection does not match the shared candidate winner")
        stations = np.asarray(state["validation_stations"])
        if (stations.ndim != 1 or not len(stations) or stations.dtype.kind not in "iu"
                or (stations < 0).any() or np.any(np.diff(stations) <= 0)):
            raise ValueError("invalid saved validation stations")
        obj.validation_stations_ = stations.astype(np.int64).tolist()
        obj.source_role_, obj.selection_role_ = state["source_role"], state["selection_role"]
        obj.selection_, obj.selection_scores_ = choice, scores
        obj.fold_diagnostics_ = state["fold_diagnostics"]
        obj.coordinates_fitted_, obj.fitted_ = True, True
        return obj
