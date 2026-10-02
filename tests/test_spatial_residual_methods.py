import numpy as np

from river_graph.experiments.spatial_residual_methods import (
    candidate_specs,
    huber_location,
    station_residual_adjustment,
)


def test_location_methods_use_only_support_residuals() -> None:
    support_z = np.array([0.2, 0.4, 0.6])
    residual = np.array([0.1, 0.2, 0.3])
    query_z = np.array([0.0, 1.0, 2.0])
    month = np.array([0, 6, 12])
    out = station_residual_adjustment(
        "mean", support_z, residual, query_z, month, month, alpha=0.5,
    )
    np.testing.assert_allclose(out, np.full(3, 0.1))
    changed_query = station_residual_adjustment(
        "mean", support_z, residual, query_z + 99, month, month, alpha=0.5,
    )
    np.testing.assert_array_equal(out, changed_query)


def test_huber_downweights_an_outlier_and_structured_methods_are_finite() -> None:
    values = np.array([0.0, 0.1, 0.2, 8.0])
    assert huber_location(values) < 1.0
    support_z = np.array([0.2, 0.4, 0.6])
    residual = np.array([0.1, 0.2, 0.3])
    query_z = np.array([0.0, 1.0, 2.0])
    support_month = np.array([0, 3, 6])
    query_month = np.array([1, 4, 7])
    for method in ("affine", "seasonal"):
        output = station_residual_adjustment(
            method, support_z, residual, query_z,
            support_month, query_month, regularization=10.0,
        )
        assert np.isfinite(output).all()
        assert output.shape == query_z.shape


def test_candidate_set_is_small_and_fixed() -> None:
    specs = candidate_specs()
    assert len(specs) == 18
    assert {str(spec["method"]) for spec in specs} == {
        "mean", "median", "huber", "affine", "seasonal",
    }
