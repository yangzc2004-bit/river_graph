"""Regional folds exclude entire catchment groups and ignore concentration labels."""
from __future__ import annotations

import numpy as np
import pytest

from river_graph.models.kgml_local_transport import fold_split
from river_graph.models.regional_source_views import huc4_code, huc4_source_folds


def test_regional_folds_partition_sources_without_splitting_huc4():
    huc = [f"{region:04d}0101" for region in (1013, 1019, 708, 1030, 1101) for _ in range(3)]
    cells = np.array([site*8+month for site in range(14) for month in range(1+site % 5)])
    a = huc4_source_folds(cells, 8, huc, 42)
    b = huc4_source_folds(cells, 8, huc, 42)
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.unique(cells//8))
    for fold, other in zip(a, b, strict=True):
        np.testing.assert_array_equal(fold, other)
        regions = {huc[site][:4] for site in fold}
        assert all(site in fold for site in np.unique(cells//8) if huc[site][:4] in regions)
        view = fold_split({"train": cells, "context": cells.copy(), "val": np.array([], int)}, fold, 8)
        assert not np.isin(view["train"]//8, fold).any()
        assert not np.isin(view["context"]//8, fold).any()
    assert all(14 not in fold for fold in a)


def test_regional_groups_handle_leading_zero_and_reject_missing_identity():
    folds = huc4_source_folds(np.arange(12), 4, [7080101, 7080102, 10130101], 42)
    assert any(set(fold) == {0, 1} for fold in folds)
    with pytest.raises(ValueError):
        huc4_source_folds(np.arange(12), 4, [None, "07080102", "10130101"], 42)
    with pytest.raises(ValueError, match="two source"):
        huc4_source_folds(np.arange(8), 4, ["07080101", "07080102"], 42)


def test_mixed_huc8_huc12_keep_the_same_upstream_region():
    codes = ["05020005", "50200050803", "07080101", "070801010203", "10130101", "101301010101"]
    assert [huc4_code(value) for value in codes] == ["0502", "0502", "0708", "0708", "1013", "1013"]
    groups = huc4_source_folds(np.arange(24), 4, codes, 42)
    assert {tuple(group.tolist()) for group in groups} == {(0, 1), (2, 3), (4, 5)}
