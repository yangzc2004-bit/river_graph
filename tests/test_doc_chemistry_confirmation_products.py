"""Fixed query curves and value-blind fresh confirmation role assignments."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def driver():
    path = str(Path(__file__).resolve().parents[1] / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("run_doc_chemistry_confirmation_v1")


def products():
    module = driver()
    n, months = 10, 8
    data = {"y": np.arange(n * months).reshape(n, months), "y_mask": np.ones((n, months), bool),
            "site_no": np.array([f"station{i}" for i in range(n)]),
            "months": np.array([f"2000-{i+1:02d}" for i in range(months)])}
    split, _ = module.build_unified_spatial_split(data["y_mask"], seed=242)
    _, query = module.support_query_cells(split, target_role="test", k=0, n_months=months)
    cells = np.arange(n * months)
    full = pd.DataFrame({"cell": cells, "station": data["site_no"][cells // months],
                         "month": data["months"][cells % months], "point_pred": cells / 10})
    panels = []
    for model in module.MODELS:
        for k in module.KS:
            panel = full.iloc[query].copy()
            panel["model_name"], panel["k"] = model, k
            panels.append(panel)
    return module, data, split, full, pd.concat(panels, ignore_index=True), query


def test_fixed_query_curve_contract_and_truth_export_boundary():
    module, data, split, full, panels, query = products()
    np.testing.assert_array_equal(module.validate_products(full, panels, data, split), query)
    modified = panels.copy()
    modified.loc[0, "cell"] = split["train"][0]
    with pytest.raises(AssertionError):
        module.validate_products(full, modified, data, split)
    panels["y_true"] = data["y"].ravel()[panels.cell]
    with pytest.raises(ValueError, match="without query truth"):
        module.validate_products(full, panels, data, split)


def test_fresh_partition_freeze_uses_mask_only_and_rejects_changed_roles(tmp_path):
    module, data, _, _, _, _ = products()
    first, protocol, path = module.freeze_split(tmp_path, data, 242)
    data["y"] = data["y"] * -100000
    second, new_protocol, new_path = module.freeze_split(tmp_path, data, 242)
    assert protocol == new_protocol and path == new_path
    for role in first:
        np.testing.assert_array_equal(first[role], second[role])
    altered = {role: values.copy() for role, values in first.items()}
    altered["test"][0] = altered["train"][0]
    np.savez_compressed(path, **altered)
    with pytest.raises(AssertionError):
        module.freeze_split(tmp_path, data, 242)
