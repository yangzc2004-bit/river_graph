"""Fresh-role query matching and inferential estimator on synthetic panels."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def analyzer():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("confirmation_analysis", scripts / "analyze_doc_chemistry_confirmation_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(module, models=("a", "b")):
    rows = []
    for split in module.SPLITS:
        # Different cell counts test equal partition weighting; cells/stations
        # repeat across roles to test global identity and shared resampling.
        cells = range(3 + 2*(split-module.SPLITS[0]))
        for seed in module.SEEDS:
            for model in models:
                for k in module.KS:
                    for cell in cells:
                        truth = float(1+cell)
                        error = (split-module.SPLITS[0]+1)*(1 if model == models[0] else 2)
                        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                            "cell": cell, "station": f"S{cell % 3}", "month": f"2000-{cell+1:02d}",
                            "y_true": truth, "y_pred": truth+error, "ecological_novelty": .2,
                            "upstream_support": .5, "ph_available": cell % 2 == 0,
                            "ec_available": cell % 3 == 0})
    return pd.DataFrame(rows)


def test_fixed_new_roles_complete_queries_and_bad_panels():
    module = analyzer()
    data = panel(module)
    module.validate_panel(data, expected_models=("a", "b"))
    for change, message in (
        (lambda f: pd.concat((f, f.iloc[:1])), "duplicate"),
        (lambda f: f.iloc[1:].copy(), "Fixed query"),
        (lambda f: f.assign(split_seed=f.split_seed-100), "fresh"),
        (lambda f: f[~f.model_name.eq("b")].copy(), "Incomplete"),
        (lambda f: f.assign(y_pred=np.nan), "Nonfinite"),
    ):
        with pytest.raises(ValueError, match=message):
            module.validate_panel(change(data), expected_models=("a", "b"))
    changed = data.copy()
    changed.loc[0, "ph_available"] = not changed.loc[0, "ph_available"]
    with pytest.raises(ValueError, match="availability"):
        module.validate_panel(changed, expected_models=("a", "b"))


def test_metrics_weight_partitions_equally_and_seeds_are_not_new_cells():
    module = analyzer()
    data = panel(module)
    thresholds = {(s, r): 3. for s in module.SPLITS for r in module.SEEDS}
    runs, parts, curves = module.metric_summary(data, thresholds)
    first = curves[curves.model_name.eq("a") & curves.k.eq(0)].iloc[0]
    assert first.mae == 2  # mean(1,2,3); pooled-cell MAE would differ.
    assert not np.isclose(first.mae, np.average([1, 2, 3], weights=[3, 5, 7]))
    assert parts.n_query_cells.max() == 7 and runs.n_query_cells.max() == 7
    assert set(parts.n_seeds) == {3}
    assert np.isfinite(first[["log_mae", "log_rmse", "log_r2", "signed_bias"]].to_numpy(float)).all()
    missing_tail = {(s, r): 999. for s in module.SPLITS for r in module.SEEDS}
    assert module.metric_summary(data, missing_tail)[2].q90_mae.isna().all()


def test_joint_station_bootstrap_matches_explicit_shared_multiplicities():
    analyzer()  # Put shared script helpers on sys.path.
    from analyze_unified_doc_spatial import joint_station_bootstrap
    rows = []
    for split in (242, 243, 244):
        for station in ("s0", "s1", "s2"):
            for seed in (42, 43, 44):
                rows.append({"split_seed": split, "station": station, "seed": seed,
                    "cell": int(station[-1]), "candidate_error": (split-241)*(int(station[-1])+1),
                    "reference_error": (split-240)*(4-int(station[-1]))})
    pair = pd.DataFrame(rows)
    result = joint_station_bootstrap(pair, draws=101, seed=42)
    weight = np.random.default_rng(42).multinomial(3, [1/3]*3, size=101)
    c = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9.]])
    r = np.array([[8, 12, 16], [6, 9, 12], [4, 6, 8.]])
    candidate = ((weight@c)/3).mean(1)
    reference = ((weight@r)/3).mean(1)
    np.testing.assert_allclose([result["delta_ci_low"], result["delta_ci_high"]],
                               np.quantile(candidate-reference, [.025, .975]), rtol=0, atol=1e-14)
    assert result["n_station_months_unique"] == 3
    assert result["n_split_cell_occurrences"] == 9
    assert result["n_stations_unique"] == 3
    # Duplicating a training seed must not change ecological sample size or CI.
    copy = pd.concat((pair, pair.assign(seed=45)))
    assert joint_station_bootstrap(copy, draws=101, seed=42) == result


def test_fixed_contrasts_keep_decoder_and_support_controls_separate():
    module = analyzer()
    definitions = module.comparison_definitions()
    assert len(definitions) == 30 and len({row[0] for row in definitions}) == 30
    for _, candidate, ck, reference, rk, role in definitions:
        assert candidate in module.MODELS and reference in module.MODELS and ck == rk
        if role == "matched_decoder_information":
            assert candidate.endswith("_legacy") and reference.endswith("_legacy")
    assert {row[2] for row in definitions if row[-1] == "accepted_vs_general"} == {0, 3, 5}


def test_selected_curve_must_equal_source_choice_not_best_target_model():
    module = analyzer()
    models = tuple(f"{p}_{b}" for p in module.PIPELINES for b in module.VARIANTS)
    data = panel(module, models)
    data = data[data.split_seed.eq(242) & data.seed.eq(42)].copy()
    data["y_pred"] = data.y_true + 2
    selection = {"choices": {p: {str(k): {"basis": "legacy"} for k in module.KS} for p in module.PIPELINES}}
    checks = module.check_selected_curves(data, selection)
    assert len(checks) == 12
    changed = data.copy()
    mask = changed.model_name.eq("neural_chemistry_selected") & changed.k.eq(5)
    changed.loc[mask, "y_pred"] -= 1  # Better is still invalid if not source selected.
    with pytest.raises(AssertionError):
        module.check_selected_curves(changed, selection)


def test_region_contrasts_accept_fresh_ids_and_flag_missing_tail_partition():
    module = analyzer()
    data = panel(module)
    thresholds = {(s, r): 3. for s in module.SPLITS for r in module.SEEDS}
    definitions = [("a_vs_b", "a", 0, "b", 0, "synthetic")]
    effects, seeds, parts, *_ = module.compare(data, thresholds, 31, definitions)
    effect = effects[effects.metric.eq("mae") & effects.region.eq("overall")].iloc[0]
    assert effect.delta_value == -2 and effect.improved_splits == 3
    assert len(seeds[seeds.region.eq("overall")]) == 9
    assert set(parts.split_seed) == {242, 243, 244}
    thresholds.update({(242, seed): 999. for seed in module.SEEDS})
    effects = module.compare(data, thresholds, 31, definitions)[0]
    assert set(effects[effects.region.eq("q90")].status) == {"missing_partition"}
