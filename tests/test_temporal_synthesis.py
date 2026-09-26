"""Statistical contracts for T8; no training or local datasets required."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'temporal_synthesis', Path(__file__).parents[1] / 'scripts/analyze_temporal_synthesis.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pairing_preserves_constant_cellwise_improvement():
    # Unequal clusters with very different baseline errors must stay paired.
    before = np.array([2., 30., 400., 500., 600.])
    errors = np.column_stack([before, before - 1])
    result = module.paired_bootstrap(errors, np.array(['a', 'a', 'b', 'b', 'b']))
    assert result['clusters'] == 2
    np.testing.assert_allclose([result['delta_lo'], result['delta_hi']], [-1, -1])


def test_two_cluster_distribution_has_known_endpoints():
    # Replicate deltas can only be -1, -4, -10; percentile endpoints are exact.
    result = module.paired_bootstrap(
        np.array([[1., 0.], [1., 0.], [10., 0.]]), np.array([0, 0, 1])
    )
    assert result['delta_lo'] == -10
    assert result['delta_hi'] == -1
    assert result['reduction_lo'] == result['reduction_hi'] == 100
