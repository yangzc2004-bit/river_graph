"""Synthetic fixed-query, estimator and figure contracts for fresh confirmation."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def modules():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return (importlib.import_module("analyze_doc_nested_confirmation_v1"),
            importlib.import_module("plot_doc_nested_confirmation_v1"))


def toy_panel(module):
    """Unequal partition sizes, repeated stations, fixed queries and real fallback."""
    rows = []
    for split in module.SPLITS:
        size = 3 + 2 * (split - module.SPLITS[0])
        error = float(split - module.SPLITS[0] + 1)
        for seed in module.SEEDS:
            for model in module.MODELS:
                for k in module.KS:
                    for cell in range(size):
                        truth = float(1 + cell)
                        ph, ec = cell % 2 == 0, cell % 3 == 0
                        active = ph or ec
                        parent = truth + error
                        offset = {module.GENERAL: .2, module.JOINT: .1, module.TREE: -.1}.get(model, 0.)
                        delta = (-.03 if model == module.NESTED else -.01) if (
                            model in (module.NESTED, module.MASKS) and k >= 3 and active) else 0.
                        prediction = np.expm1(np.log1p(parent) + delta) if delta else parent + offset
                        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                            "cell": cell, "station": f"S{cell % 3}", "month": f"2000-{cell + 1:02d}",
                            "y_true": truth, "y_pred": prediction, "ecological_novelty": .2,
                            "upstream_support": .5, "ph_available": ph, "ec_available": ec,
                            "aux_available": active, "chemical_delta": delta,
                            "chemical_support_count": min(k, 3) if active else 0,
                            "visibility_role": "test"})
    return pd.DataFrame(rows)


def test_fixed_six_models_fresh_roles_and_identical_queries():
    module, _ = modules()
    frame = toy_panel(module)
    module.validate_panel(frame)
    for change, match in (
        (lambda f: pd.concat((f, f.iloc[:1])), "duplicate"),
        (lambda f: f.iloc[1:], "Fixed query"),
        (lambda f: f.assign(split_seed=f.split_seed-100), "fresh confirmation"),
        (lambda f: f[~f.model_name.eq(module.TREE)], "Incomplete"),
        (lambda f: f.assign(visibility_role="val"), "held-target"),
        (lambda f: f.assign(y_pred=np.nan), "Invalid"),
        (lambda f: f.assign(upstream_support=2), "descriptors"),
    ):
        with pytest.raises(ValueError, match=match):
            module.validate_panel(change(frame))


def test_increment_invariants_reject_low_k_unavailable_and_wrong_formula():
    module, _ = modules()
    frame = toy_panel(module)
    for mask in (frame.model_name.eq(module.NESTED) & frame.k.eq(1),
                 frame.model_name.eq(module.NESTED) & frame.k.eq(3) & ~frame.aux_available):
        bad = frame.copy()
        bad.loc[mask, "chemical_delta"] = .2
        with pytest.raises(ValueError, match="zero without"):
            module.validate_panel(bad)
    bad = frame.copy()
    mask = bad.model_name.eq(module.NESTED) & bad.k.eq(5) & bad.aux_available
    bad.loc[mask, "y_pred"] += .01
    with pytest.raises(ValueError, match="reproduce"):
        module.validate_panel(bad)
    bad = frame.copy()
    bad.loc[frame.model_name.eq(module.NESTED) & frame.k.eq(0), "y_pred"] += .01
    with pytest.raises(ValueError, match="Zero chemical increment"):
        module.validate_panel(bad)


def test_exact_fixed_comparisons_and_equal_partition_estimand():
    module, _ = modules()
    definitions = module.comparison_definitions()
    assert len(definitions) == len({row[0] for row in definitions}) == 10
    primary = [row for row in definitions if row[-1] == "primary_nested_vs_joint"]
    assert {(row[1], row[2], row[3], row[4]) for row in primary} == {
        (module.NESTED, k, module.JOINT, k) for k in (3, 5)}
    assert {row[3] for row in definitions} == set(module.MODELS) - {module.NESTED}
    panel = toy_panel(module)
    thresholds = {(s, r): 3. for s in module.SPLITS for r in module.SEEDS}
    runs, parts, curves = module.metric_summary(panel, thresholds)
    row = curves[curves.model_name.eq(module.LEGACY) & curves.k.eq(0)].iloc[0]
    assert row.mae == 2.  # partition means 1,2,3, despite query counts3,5,7
    assert not np.isclose(row.mae, np.average([1, 2, 3], weights=[3, 5, 7]))
    assert set(parts.n_seeds) == {3} and runs.n_query_cells.max() == 7
    assert np.isfinite(row[["log_mae", "log_rmse", "log_r2"]].to_numpy(float)).all()
    no_tail = {(s, r): 999. for s in module.SPLITS for r in module.SEEDS}
    assert module.metric_summary(panel, no_tail)[2].q90_mae.isna().all()


def test_paired_comparison_is_seed_invariant_with_joint_station_resampling():
    module, _ = modules()
    from analyze_unified_doc_spatial import joint_station_bootstrap, paired_cells
    frame = toy_panel(module)
    pair = paired_cells(frame, module.NESTED, 3, module.JOINT, 3)
    result = joint_station_bootstrap(pair, draws=101, seed=42)
    repeated = pd.concat((pair, pair[pair.seed.eq(42)].assign(seed=45)))
    repeat_result = joint_station_bootstrap(repeated, draws=101, seed=42)
    for key, value in result.items():
        if isinstance(value, (int, np.integer, str, bool)):
            assert repeat_result[key] == value
        else:
            np.testing.assert_allclose(repeat_result[key], value, rtol=0, atol=1e-12)
    assert result["n_station_months_unique"] == 7
    assert result["n_split_cell_occurrences"] == 15
    # Exact station multiplicities are shared across all role partitions.
    stations = sorted(pair.station.unique())
    weights = np.random.default_rng(42).multinomial(len(stations), np.full(len(stations), 1/len(stations)), size=101)
    values = []
    for column in ("candidate_error", "reference_error"):
        means = []
        for _, group in pair.groupby("split_seed"):
            single = group.groupby(["station", "cell"])[column].mean().reset_index()
            sums = single.groupby("station")[column].sum().reindex(stations, fill_value=0).to_numpy()
            counts = single.groupby("station").size().reindex(stations, fill_value=0).to_numpy()
            means.append((weights @ sums)/(weights @ counts))
        values.append(np.mean(means, axis=0))
    np.testing.assert_allclose([result["delta_ci_low"], result["delta_ci_high"]],
                               np.quantile(values[0]-values[1], [.025, .975]), rtol=0, atol=1e-14)


def synthetic_figure_tables(tmp_path, module):
    """A fabricated test fixture only, never saved in an experiment directory."""
    from analyze_doc_chemistry_support_v1 import (
        attach_availability,
        availability_profiles,
        full_grid_availability,
    )
    panel = toy_panel(module)
    thresholds = {(s, r): 3. for s in module.SPLITS for r in module.SEEDS}
    runs, parts, curves = module.metric_summary(panel, thresholds)
    definitions = [row for row in module.comparison_definitions() if row[-1] == "primary_nested_vs_joint"]
    effects, _, directions, *_ = module.compare(panel, thresholds, 5000, definitions)
    attached = attach_availability(panel)
    population = availability_profiles(attached, thresholds)[-1]
    grid = panel.drop_duplicates("cell").copy().assign(doc_observed=True)
    deployment = full_grid_availability(attach_availability(grid))
    tables = {"k_curves": curves, "metrics_by_run": runs, "metrics_by_partition": parts,
                  "paired_effects": effects, "directions_by_partition": directions,
                  "query_population": population, "deployment_availability": deployment}
    out = tmp_path / "analysis"
    out.mkdir()
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    manifest = {"complete": True, "packages": 9, "partitions": list(module.SPLITS),
        "training_seeds": list(module.SEEDS), "models": list(module.MODELS), "bootstrap_draws": 5000,
        "outputs": {path.name: module.sha256(path) for path in out.glob("*.csv")}}
    (out / "sources.json").write_text(json.dumps(manifest))
    return out


def test_figure_tables_reconcile_render_and_reject_tampering(tmp_path):
    module, plot = modules()
    out = synthetic_figure_tables(tmp_path, module)
    data = plot.load_analysis(tmp_path)
    assert set(data["curves"].model_name) == set(module.MODELS)
    fig = plot.draw_figure(data)
    assert len(fig.axes) == 3 and len(fig.axes[0].lines) == 6 and len(fig.axes[2].lines) == 6
    assert [fig.axes[1].get_yticklabels()[i].get_text() for i in range(2)] == ["K = 3", "K = 5"]
    for extension in ("pdf", "svg", "png"):
        fig.savefig(tmp_path / f"synthetic_only.{extension}", dpi=100)
        assert (tmp_path / f"synthetic_only.{extension}").stat().st_size > 1000
    plot.plt.close(fig)
    path = out / "k_curves.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Changed or unbound"):
        plot.load_analysis(tmp_path)


def test_figure_rejects_bound_but_inconsistent_summary(tmp_path):
    module, plot = modules()
    out = synthetic_figure_tables(tmp_path, module)
    path = out / "k_curves.csv"
    curves = pd.read_csv(path)
    curves.loc[0, "mae"] += .1
    curves.to_csv(path, index=False)
    manifest_path = out / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][path.name] = module.sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        plot.load_analysis(tmp_path)
