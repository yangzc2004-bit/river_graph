"""Synthetic checks for conditional chemical calibration analysis."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def analyzer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("nested_analysis", scripts / "analyze_doc_nested_chemistry_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(module):
    rows = []
    for split in module.SPLITS:
        cells = range(3 + 2*(split-module.SPLITS[0]))
        for seed in module.SEEDS:
            for model in module.MODELS:
                for k in module.KS:
                    for cell in cells:
                        truth = float(cell+1)
                        ph, ec = cell % 2 == 0, cell % 3 == 0
                        parent = truth + split-module.SPLITS[0]+1 + (seed-module.SEEDS[0])/10
                        delta = (-.01 if model == module.NESTED else -.005) if (
                            model in (module.NESTED, module.MASKS) and k >= 3 and (ph or ec)) else 0.
                        prediction = parent if delta == 0 else float(np.expm1(np.log1p(parent)+delta))
                        if model == module.GENERAL:
                            prediction += .2
                        elif model == module.JOINT:
                            prediction += .1
                        elif model == module.TREE:
                            prediction -= .1
                        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                            "cell": cell, "station": f"S{cell % 3}", "month": f"2000-{cell+1:02d}",
                            "y_true": truth, "y_pred": prediction, "ph_available": ph,
                            "ec_available": ec, "aux_available": ph or ec, "chemical_delta": delta,
                            "chemical_support_count": min(k, 2) if model in (module.NESTED, module.MASKS) and (ph or ec) else 0})
    return pd.DataFrame(rows)


def thresholds(module, value=3.):
    return {(split, seed): value for split in module.SPLITS for seed in module.SEEDS}


def test_full_queries_and_increment_fallback_contracts():
    module = analyzer()
    data = panel(module)
    module.validate_panel(data)
    for change, message in (
        (lambda frame: pd.concat((frame, frame.iloc[:1])), "duplicate"),
        (lambda frame: frame.iloc[1:].copy(), "Fixed query"),
        (lambda frame: frame.assign(split_seed=frame.split_seed-100), "nine"),
        (lambda frame: frame[~frame.model_name.eq(module.TREE)].copy(), "Incomplete"),
        (lambda frame: frame.assign(y_pred=np.nan), "Invalid validation"),
        (lambda frame: frame.assign(aux_available=False), "availability"),
        (lambda frame: frame.assign(chemical_support_count=99), "support count"),
    ):
        with pytest.raises(ValueError, match=message):
            module.validate_panel(change(data))
    changed = data.copy()
    mask = changed.model_name.eq(module.NESTED) & changed.k.eq(0)
    changed.loc[mask, "y_pred"] += .1
    with pytest.raises(ValueError, match="Zero chemical increment"):
        module.validate_panel(changed)
    changed = data.copy()
    mask = changed.model_name.eq(module.NESTED) & changed.k.eq(3) & changed.chemical_delta.ne(0)
    changed.loc[mask, "y_pred"] += .1
    with pytest.raises(ValueError, match="reproduce"):
        module.validate_panel(changed)


def test_report_estimator_uses_seed_mean_and_equal_partitions():
    module = analyzer()
    data = panel(module)
    runs, parts, curves = module.summarize(data, thresholds(module))
    row = curves[curves.model_name.eq(module.LEGACY) & curves.k.eq(0)].iloc[0]
    assert row.mae == pytest.approx(2.1)  # mean seed deviations=.1; mean split errors=2.
    assert not np.isclose(row.mae, np.average([1.1, 2.1, 3.1], weights=[3, 5, 7]))
    assert set(parts.n_seeds) == {3}
    assert parts.n_query_cells.max() == 7 and runs.n_query_cells.max() == 7
    assert row.q90_signed_bias > 0
    assert curves.q90_unstable_any.all()
    missing = module.summarize(data, thresholds(module, 999.))[2]
    assert missing.q90_mae.isna().all() and missing.q90_signed_bias.isna().all()


def test_fixed_effects_and_station_weights_reproduce_reported_delta():
    module = analyzer()
    data = panel(module)
    effects, _, parts, _, stations, concentration = module.compare(data, thresholds(module))
    assert len(module.comparison_definitions()) == 8
    assert set(effects.k) == {3, 5}
    assert set(effects.reference) == {module.LEGACY, module.JOINT, module.GENERAL, module.MASKS}
    for row in effects[effects.region.eq("overall")].itertuples():
        summed = stations[stations.comparison.eq(row.comparison)].weighted_gain.sum()
        assert summed == pytest.approx(-row.delta_mae, abs=1e-14)
        weighted = concentration[concentration.comparison.eq(row.comparison)].iloc[0]
        assert weighted.net_mae_reduction == pytest.approx(-row.delta_mae, abs=1e-14)
        expected = parts[parts.comparison.eq(row.comparison) & parts.region.eq("overall")].delta_mae.mean()
        assert row.delta_mae == pytest.approx(expected)
        assert row.n_seed_fits == 9
    # Repeated seeds change neither the weighting nor the number of unique stations.
    repeated = pd.concat((data, data.assign(seed=data.seed+10)), ignore_index=True)
    duplicate_effects = module.compare(repeated, thresholds(module) | {
        (split, seed+10): 3. for split in module.SPLITS for seed in module.SEEDS})[0]
    np.testing.assert_allclose(effects.delta_mae, duplicate_effects.delta_mae, rtol=0, atol=1e-14)
    assert set(concentration.n_stations_unique) == {3}


def test_absent_tail_partitions_are_not_silently_dropped():
    module = analyzer()
    data = panel(module)
    cutoffs = thresholds(module)
    cutoffs.update({(module.SPLITS[0], seed): 999. for seed in module.SEEDS})
    effects = module.compare(data, cutoffs)[0]
    tail = effects[effects.region.eq("q90")]
    assert set(tail.status) == {"missing_partition"}
    assert tail.delta_mae.isna().all() and set(tail.n_nonempty_partitions) == {2}
    tail = module.compare(data, thresholds(module, 999.))[0]
    assert set(tail[tail.region.eq("q90")].status) == {"empty_region"}


def small_fitted_adapter():
    analyzer()
    from river_graph.models.nested_chemical_adapter import (
        NestedChemicalAdapter,
        NestedChemicalEpisode,
    )
    months, stations = 12, 6
    grid = np.stack((np.sin(np.arange(months)), np.cos(np.arange(months))), axis=1)
    source = np.tile(grid, (stations, 1, 1))
    adapter = NestedChemicalAdapter(months, ridge_values=(1., 10.), strength_values=(0., .5),
                                    selection_folds=3).fit_coordinates(
        source, np.ones((stations, months), bool), source_role="source_training")
    episodes = []
    for k in (3, 5):
        support = np.asarray([station*months+month for station in range(stations) for month in range(k)])
        query = np.asarray([station*months+month for station in range(stations) for month in (8, 9)])
        qchem, schem = source.reshape(-1, 2)[query], source.reshape(-1, 2)[support]
        qy = np.expm1(np.log1p(5.)+.1*qchem[:, 0])
        sy = np.expm1(np.log1p(5.)+.1*schem[:, 0])
        episodes.append(NestedChemicalEpisode(k, query, qy, np.full(len(query), 5.),
            support, sy, np.full(len(support), 5.), qchem, schem,
            np.ones(len(query), bool), np.ones(len(support), bool)))
    return adapter.fit(episodes, selection_role="source_validation"), episodes


def small_state():
    return small_fitted_adapter()[0].to_dict()


def test_fold_choices_retain_shared_k_rule_and_reject_changed_choice():
    module = analyzer()
    state = small_state()
    choice, scores, folds = module.state_records(state, {"split_seed": 242, "seed": 42}, "chemistry")
    assert choice["strength"] == state["selection"]["strength"]
    assert len(scores) == 4 and len(folds) == 3
    assert all({"selection_mae_station_equal_k3", "held_mae_station_equal_k5"} <= row.keys() for row in folds)
    changed = copy.deepcopy(state)
    changed["fold_diagnostics"][0]["selected_zero"] = not changed["fold_diagnostics"][0]["selected_zero"]
    with pytest.raises(ValueError, match="shared-choice"):
        module.state_records(changed, {"split_seed": 242, "seed": 42}, "chemistry")


def test_saved_fold_scores_and_cv_assignments_match_predictions():
    module = analyzer()
    adapter, episodes = small_fitted_adapter()
    validation, cv = [], []
    for episode in episodes:
        arguments = (episode.query_prediction, episode.query_cells, episode.support_prediction,
                     episode.support_cells, episode.support_values)
        keywords = {"query_chemical": episode.query_chemical, "support_chemical": episode.support_chemical,
                    "query_active": episode.query_active, "support_active": episode.support_active, "k": episode.k}
        prediction = adapter.adapt(*arguments, **keywords)
        rows = pd.DataFrame({"cell": episode.query_cells, "k": episode.k, "model_name": module.NESTED,
                             "y_true": episode.query_values, "y_pred": prediction, "conditional_fold": -1})
        validation.append(rows)
        rows = rows.copy()
        for fold in adapter.fold_diagnostics_:
            selected = np.isin(episode.query_cells//adapter.n_months, fold["held_stations"])
            prediction = adapter.adapt_components(*arguments, **keywords, **fold["choice"])["y_pred"]
            rows.loc[selected, "y_pred"] = prediction[selected]
            rows.loc[selected, "conditional_fold"] = fold["fold"]
        cv.append(rows)
    validation, cv = pd.concat(validation, ignore_index=True), pd.concat(cv, ignore_index=True)
    module.check_state_predictions(adapter.to_dict(), validation, cv, module.NESTED)
    changed = cv.copy()
    changed.loc[0, "conditional_fold"] = 99
    with pytest.raises(ValueError, match="wrong held-station fold"):
        module.check_state_predictions(adapter.to_dict(), validation, changed, module.NESTED)
    changed = cv.copy()
    changed.loc[0, "y_pred"] += .2
    with pytest.raises(ValueError, match="CV scores"):
        module.check_state_predictions(adapter.to_dict(), validation, changed, module.NESTED)


def test_lightweight_product_identity_does_not_need_parent_forests(tmp_path):
    module = analyzer()
    frame = panel(module)
    frame = frame[frame.split_seed.eq(242) & frame.seed.eq(42)]
    prediction = tmp_path / "validation.parquet"
    frame.to_parquet(prediction, index=False)
    config = {"split_seed": 242, "seed": 42, "runtime_snapshot_hash": "runtime",
        "dataset_hash": "dataset", "mask_hash": "mask", "started_at": "2026-10-04T00:00:00+00:00"}
    digest = module.config_digest(config)
    state = tmp_path / "nested_chemistry.json"
    state.write_text("{}")
    metadata = {"config": config, "config_hash": digest, "rows": len(frame),
        "runtime_snapshot_hash": "runtime", "dataset_sha256": "dataset", "mask_sha256": "mask",
        "prediction_sha256": module.sha256(prediction), "selection_role": "source_validation",
        "run_identity_sha256": module.run_identity_sha256(digest, config["started_at"], "runtime"),
        "model_files": {state.name: module.sha256(state)}}
    meta_path = tmp_path / "validation.meta.json"
    meta_path.write_text(json.dumps(metadata))
    completion = {"files": {path.name: module.sha256(path) for path in (prediction, state, meta_path)}}
    result = module.read_product(tmp_path, prediction.name, config, completion, [])
    pd.testing.assert_frame_equal(result, frame.reset_index(drop=True))
    metadata["mask_sha256"] = "changed"
    meta_path.write_text(json.dumps(metadata))
    completion["files"][meta_path.name] = module.sha256(meta_path)
    with pytest.raises(ValueError, match="sidecar"):
        module.read_product(tmp_path, prediction.name, config, completion, [])


def test_report_separates_tuning_from_conditional_cv_and_no_promotion(tmp_path):
    module = analyzer()
    data = panel(module)
    curves = module.summarize(data, thresholds(module))[2]
    effects = module.compare(data, thresholds(module))[0]
    tables = {f"{scope}_kcurves": curves for scope in ("validation_tuning", "conditional_cv")}
    tables |= {f"{scope}_comparisons": effects for scope in ("validation_tuning", "conditional_cv")}
    choices = pd.DataFrame({"mode": ["chemistry", "masks"], "strength": [.5, 0.]})
    module.write_findings(tmp_path, tables, {scope: data for scope in ("validation_tuning", "conditional_cv")}, choices)
    text = (tmp_path / "findings.md").read_text()
    assert "Conditional station CV" in text and "Full-validation tuning" in text
    assert "not a fully OOF or independent confirmation" in text
    assert "No model promotion" in text and "stations equally" in text
