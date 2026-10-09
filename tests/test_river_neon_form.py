"""River-grain, visibility and dependence checks for the field replication."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_neon_form import (
    ARMS,
    CONTEXT,
    calendar_endpoints,
    gain_summary,
    heldout_predictions,
    upstream_overlap_groups,
)


def occasions():
    rows = []
    for month in range(1, 13):
        count = 10 if month == 1 else 1
        for j in range(count):
            rows.append({"siteID": "TEST", "sample_base_id": f"TEST.SS.2018{month:02}{j+1:02}.FIL",
                         "collectDate": f"2018-{month:02}-{j+1:02}",
                         "doc_mean_mgl": 100 if month == 1 else 1,
                         "waterbody_scope": "stream_or_river", "primary_surface_location": True,
                         "all_finite_nonnegative": True})
    rows.append(rows[0] | {"sample_base_id": "TEST.RE.20180101.FIL", "doc_mean_mgl": 1e6,
                           "primary_surface_location": False})
    return pd.DataFrame(rows)


def test_calendar_equalization_and_relocation_exclusion():
    endpoints, monthly = calendar_endpoints(occasions(), minimum_months=12, minimum_years=1)
    assert len(monthly) == 12
    assert endpoints.n_months.iloc[0] == 12
    assert endpoints.doc_level_mgl.iloc[0] == 1
    assert monthly.doc_mgl.max() == 100
    assert endpoints.calendar_eligible.all()


def test_replicate_aggregation_is_required():
    f = occasions()
    with pytest.raises(ValueError, match="must be unique"):
        calendar_endpoints(pd.concat([f, f.iloc[:1]], ignore_index=True))


def test_nested_overlap_is_transitive_not_sample_count():
    groups, pairs = upstream_overlap_groups({"A": {1, 2}, "B": {2, 3}, "C": {3, 4}, "D": {8}})
    assert groups == {"A": "A", "B": "A", "C": "A", "D": "D"}
    assert len(pairs) == 6
    assert pairs.shared_reaches.sum() == 2


def panel():
    rng = np.random.default_rng(42)
    f = pd.DataFrame({c: rng.normal(size=12) for c in [*CONTEXT, *ARMS["all_form"]]})
    f["site"] = [f"S{i}" for i in range(12)]
    f["network_group"] = [f"G{i//2}" for i in range(12)]
    f["doc_level_mgl"] = rng.uniform(1, 3, len(f))
    f["doc_relative_iqr"] = rng.uniform(.1, .6, len(f))
    return f


def test_whole_network_holdout_excludes_all_its_doc():
    f = panel()
    first = heldout_predictions(f)
    f.loc[f.network_group.eq("G0"), ["doc_level_mgl", "doc_relative_iqr"]] = 999
    changed = heldout_predictions(f)
    a = first[first.holdout_group.eq("G0")].y_pred.to_numpy()
    b = changed[changed.holdout_group.eq("G0")].y_pred.to_numpy()
    np.testing.assert_array_equal(a, b)
    assert first.n_training_sites.eq(10).all()


def test_gain_uses_paired_site_errors_and_full_group_resampling():
    rows = []
    for i in range(12):
        for model in ARMS:
            rows.append({"site": f"S{i}", "holdout_group": f"G{i//2}", "outcome": "doc_level_mgl",
                         "model": model, "y_true": 10, "y_pred": 8 if model == "context" else 9})
    summary, influence = gain_summary(pd.DataFrame(rows), draws=200)
    np.testing.assert_allclose(summary.gain_pct, 50)
    np.testing.assert_allclose(summary.ci_low_pct, 50)
    np.testing.assert_allclose(summary.ci_high_pct, 50)
    assert summary.n_groups.eq(6).all()
    assert len(influence) == 24


ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")


@pytest.mark.skipif(not (ROOT / "analysis" / "heldout_site_predictions.parquet").exists(),
                    reason="local independent NEON river-form results unavailable")
def test_result_coverage_and_recorded_gain_recompute():
    p = pd.read_parquet(ROOT / "analysis" / "heldout_site_predictions.parquet")
    assert not p.duplicated(["population", "site", "outcome", "model"]).any()
    assert np.isfinite(p[["y_true", "y_pred"]].to_numpy()).all()
    primary = p[p.population.eq("primary")]
    assert primary.site.nunique() == 13
    errors = primary.assign(error=abs(primary.y_true-primary.y_pred))
    gains = pd.read_csv(ROOT / "analysis" / "heldout_form_gains.csv")
    for row in gains[gains.population.eq("primary")].itertuples():
        sub = errors[errors.outcome.eq(row.outcome)]
        a = sub[sub.model.eq(row.candidate)].error.mean()
        b = sub[sub.model.eq("context")].error.mean()
        np.testing.assert_allclose(row.gain_pct, 100*(b-a)/b, rtol=1e-12)
    ledger = pd.read_csv(ROOT / "analysis" / "eligibility_ledger.csv")
    assert len(ledger) == 27
    assert ledger.analysis_eligible.sum() == 13
