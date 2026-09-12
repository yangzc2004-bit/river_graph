"""Figure regression tests: the figure scripts must fail loudly on bad input.

The R2 defect in the previous revision was a silent coverage change: the script
averaged whatever rows it happened to find. These tests pin the guard rails --
missing rows, duplicated rows and non-finite metrics must all raise instead of
quietly changing what the figure covers.

No real training or rendering of report figures is involved beyond building the
matplotlib objects in-process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fig1 = _load("figure1_under_test", "figure1_evolution.py")
fig2 = _load("figure2_under_test", "figure2_multiseed_variation.py")


@pytest.fixture
def tables():
    return (pd.read_csv(ROOT / fig1.FROZEN),
            pd.read_csv(ROOT / fig1.MULTISEED))


def test_source_data_shape_and_values(tables):
    frozen, multi = tables
    per_mask, grouped = fig1.build_source_data(frozen, multi)

    # 4 models x 7 masks (E1 three, E2b one, E3 three) = 28 per-mask rows
    assert len(per_mask) == 4 * 7
    assert len(grouped) == 4 * 3
    assert set(grouped["n_masks"]) == {1, 3}

    g0 = grouped[grouped["model"] == "G0"]
    assert (g0["training_seeds"] == 1).all()
    assert g0["source"].str.contains("benchmark.csv").all()

    h = grouped[grouped["model"] != "G0"]
    assert (h["training_seeds"] == 5).all()

    e1 = grouped[(grouped["model"] == "H2") &
                 (grouped["series"] == "E1 random mask (r20)")]["value"].iloc[0]
    assert e1 == pytest.approx(0.439613, abs=1e-6)
    e1x = grouped[(grouped["model"] == "H2X") &
                  (grouped["series"] == "E1 random mask (r20)")]["value"].iloc[0]
    assert e1x == pytest.approx(0.437845, abs=1e-6)
    assert e1x < e1, "E1 must not be presented as an H2X R2 improvement"


def test_missing_row_raises(tables):
    frozen, multi = tables
    broken = frozen[~((frozen["model"] == fig1.G0_MODEL_NAME) &
                      (frozen["mask"] == "e3_spatial_seed43"))]
    with pytest.raises(fig1.Figure1Error, match="no row for"):
        fig1.build_source_data(broken, multi)


def test_duplicate_row_raises(tables):
    frozen, multi = tables
    # duplicate a G0 row, i.e. a row the figure actually consumes
    target = frozen[(frozen["model"] == fig1.G0_MODEL_NAME)].iloc[[0]]
    dup = pd.concat([frozen, target], ignore_index=True)
    with pytest.raises(fig1.Figure1Error, match="refusing to average silently"):
        fig1.build_source_data(dup, multi)


def test_non_finite_metric_raises(tables):
    frozen, multi = tables
    broken = multi.copy()
    idx = broken.index[(broken["model"] == "H2") &
                       (broken["mask"] == "e2b_partial")][0]
    broken.loc[idx, "r2"] = float("nan")
    with pytest.raises(fig1.Figure1Error, match="non-finite"):
        fig1.build_source_data(frozen, broken)


def test_multiseed_missing_row_raises(tables):
    frozen, multi = tables
    assert (multi["model"] == "H2").any(), "fixture assumption"
    broken = multi[~((multi["model"] == "H2") &
                     (multi["mask"] == "e1_r20_seed42"))]
    assert len(broken) == len(multi) - 1
    with pytest.raises(fig1.Figure1Error, match="no row for"):
        fig1.build_source_data(frozen, broken)


def test_layout_check_detects_overlap(tables, tmp_path):
    """Force two labels onto each other and require a failure."""
    frozen, multi = tables
    _per_mask, grouped = fig1.build_source_data(frozen, multi)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0.2, 0.5], "o-")
    # identical anchors: the second label can only escape by moving, and with a
    # tight axis range it cannot, so the check must report the collision
    ax.annotate("0.20", (0, 0.2), textcoords="offset points", xytext=(0, 2),
                ha="center")
    ax.annotate("0.20", (0, 0.2), textcoords="offset points", xytext=(0, 2),
                ha="center")
    with pytest.raises(fig1.Figure1Error, match="overlap"):
        fig1.check_layout(fig, tmp_path / "dummy")
    plt.close(fig)

    # and the real figure must pass its own check
    fig2_obj, _paths = fig1.draw(grouped, tmp_path / "figure1")
    fig1.check_layout(fig2_obj, tmp_path / "figure1")
    plt.close(fig2_obj)
    assert (tmp_path / "figure1.png").exists()
    assert (tmp_path / "figure1.svg").exists()
    assert (tmp_path / "figure1.pdf").exists()


def test_figure2_summary_validation(tmp_path):
    summary = pd.DataFrame([
        {"model": m, "scenario": s, "mae_mean": 1.0, "mae_sd_seeds": 0.1,
         "mae_min_seed": 0.9, "mae_max_seed": 1.1,
         "r2_mean": 0.3, "r2_sd_seeds": 0.02, "r2_min_seed": 0.28,
         "r2_max_seed": 0.32, "rmse_mean": 2.0, "rmse_sd_seeds": 0.05,
         "rmse_min_seed": 1.9, "rmse_max_seed": 2.1}
        for m in fig2.MODELS for s in fig2.SCENARIOS])
    path = tmp_path / "multiseed_summary.csv"
    summary.to_csv(path, index=False)
    loaded = fig2.load_summary(path)
    assert len(loaded) == 12

    # a duplicated (model, scenario) cell must not be averaged away
    dup = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    dup.to_csv(path, index=False)
    with pytest.raises(fig2.Figure2Error, match="exactly 1 row"):
        fig2.load_summary(path)

    # a missing cell must not be silently skipped
    missing = summary[~((summary["model"] == "H2X") &
                        (summary["scenario"] == "E3"))]
    missing.to_csv(path, index=False)
    with pytest.raises(fig2.Figure2Error, match="exactly 1 row"):
        fig2.load_summary(path)

    with pytest.raises(fig2.Figure2Error, match="run scripts/analyze_multiseed"):
        fig2.load_summary(tmp_path / "does_not_exist.csv")


def test_figure2_uses_seed_spread_not_mask_spread(tmp_path):
    """The plotted error bar must equal the training-seed stdev column."""
    summary = pd.DataFrame([
        {"model": m, "scenario": s,
         "mae_mean": 1.0 + i * 0.1, "mae_sd_seeds": 0.05 + i * 0.01,
         "mae_min_seed": 0.9, "mae_max_seed": 1.1,
         "r2_mean": 0.3, "r2_sd_seeds": 0.02 + i * 0.01,
         "r2_min_seed": 0.28, "r2_max_seed": 0.32,
         "rmse_mean": 2.0, "rmse_sd_seeds": 0.05, "rmse_min_seed": 1.9,
         "rmse_max_seed": 2.1}
        for i, (m, s) in enumerate(
            (m, s) for m in fig2.MODELS for s in fig2.SCENARIOS)])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, _paths = fig2.draw(summary, tmp_path / "fig2", "mae", "MAE")
    ax = fig.axes[0]
    # errorbar containers are stored on the axes as LineCollection children
    plotted = [c for c in ax.containers if hasattr(c, "lines")]
    assert plotted, "no error bars were drawn"
    plt.close(fig)
