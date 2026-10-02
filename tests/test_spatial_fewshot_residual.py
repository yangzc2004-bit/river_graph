import numpy as np
import pytest

from scripts.run_spatial_fewshot_residual import residual_correction


def test_residual_correction_uses_support_errors_only() -> None:
    values = np.array([
        3.0, 100.0, 4.0, 5.0,
        2.0, 200.0, 1.0, 3.0,
    ])
    base_query = np.array([1.0, 2.0, 1.0, 2.0])
    base_support = np.array([1.0, 1.0])
    support = np.array([0, 4])
    query = np.array([1, 2, 5, 6])
    corrected, deltas = residual_correction(
        base_query, base_support, values, support, query,
        n_months=4, alpha=0.5,
    )
    expected = np.log1p(base_query)
    expected[:2] += 0.5 * (np.log1p(3.0) - np.log1p(1.0))
    expected[2:] += 0.5 * (np.log1p(2.0) - np.log1p(1.0))
    np.testing.assert_allclose(corrected, np.expm1(expected))
    assert deltas[0] == pytest.approx(np.log(2.0))
    assert deltas[1] == pytest.approx(np.log(1.5))

    changed = values.copy()
    changed[query] = 999.0
    changed_corrected, _ = residual_correction(
        base_query, base_support, changed, support, query,
        n_months=4, alpha=0.5,
    )
    np.testing.assert_array_equal(corrected, changed_corrected)


def test_residual_correction_rejects_overlap_and_invalid_alpha() -> None:
    values = np.ones(6)
    with pytest.raises(ValueError, match="disjoint"):
        residual_correction(
            np.ones(2), np.ones(1), values,
            np.array([1]), np.array([1, 2]),
            n_months=3, alpha=0.5,
        )
    with pytest.raises(ValueError, match="alpha"):
        residual_correction(
            np.ones(2), np.ones(1), values,
            np.array([0]), np.array([1, 2]),
            n_months=3, alpha=1.5,
        )
