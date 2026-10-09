"""Source OOF padding must not create invalid attention references."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_doc_current_source_portable_v2_r1 import source_attention_reference


def test_absent_source_cells_are_zero_and_observed_oof_values_unchanged():
    base = np.array([[2., np.nan, 4.], [np.nan, 3., np.nan]])
    visible = np.isfinite(base)
    reference = source_attention_reference(base, visible)
    np.testing.assert_array_equal(reference[visible], base[visible])
    np.testing.assert_array_equal(reference[~visible], 0.)
    assert np.isfinite(reference).all()
    changed = base.copy()
    changed[~visible] = 1e9
    np.testing.assert_array_equal(reference, source_attention_reference(changed, visible))
    assert np.isnan(base[0, 1])
    with pytest.raises(ValueError, match="observed OOF"):
        source_attention_reference(base, np.ones(base.shape, bool))


def test_negative_observed_reference_and_wrong_mask_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        source_attention_reference(np.array([[-2.]]), np.ones((1, 1), bool))
    with pytest.raises(ValueError, match="aligned"):
        source_attention_reference(np.ones((1, 2)), np.ones((1, 1), bool))
