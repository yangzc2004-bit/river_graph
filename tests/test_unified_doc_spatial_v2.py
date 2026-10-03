"""Integration contracts for the fixed-expert spatial adaptation workflow."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def runner():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("unified_doc_v2_runner", Path(scripts) / "run_unified_doc_spatial_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_arrays():
    n, months = 10, 12
    cells = np.arange(n * months)
    split = {"train": cells[:24], "val": cells[24:96], "test": cells[96:],
             "context": np.empty(0, dtype=np.int64)}
    rng = np.random.default_rng(319)
    context = 2 + rng.random(len(cells))
    temporal = 1 + context * .7
    truth = context + .5 * np.sin(cells % months / 2) + (cells // months) * .04
    shapes = {"constant": np.zeros((len(cells), 2)),
              "gru_shape": np.column_stack([np.sin(cells % months / 2), np.cos(cells % months / 2)]),
              "tree_shape": rng.normal(size=(len(cells), 2))}
    full = pd.DataFrame({"cell": cells, "station": (cells // months).astype(str),
                         "month": (cells % months).astype(str), "analyte": "doc",
                         "visibility_role": "test", "ecological_novelty": 1., "upstream_support": 0.})
    return months, split, context, temporal, truth, shapes, full


def test_fusion_and_adapter_selection_ignore_outer_query_and_support_labels():
    run = runner()
    months, split, context, temporal, truth, shapes, _ = fixture_arrays()
    fusion, _, validation = run.fit_fusion(context, temporal, truth, split, months)
    fitted = run.fit_adapters({"context": context, "fusion": validation}, shapes, truth, split, months)
    changed = truth.copy()
    changed[split["test"]] += 10000
    fusion2, _, validation2 = run.fit_fusion(context, temporal, changed, split, months)
    fitted2 = run.fit_adapters({"context": context, "fusion": validation2}, shapes, changed, split, months)
    assert fusion.to_dict() == fusion2.to_dict()
    assert {key: value.to_dict() for key, value in fitted.items()} == {
        key: value.to_dict() for key, value in fitted2.items()}


def test_query_output_has_fixed_population_and_accepts_only_support_truth():
    run = runner()
    months, split, context, temporal, truth, shapes, full = fixture_arrays()
    _, combined, validation = run.fit_fusion(context, temporal, truth, split, months)
    fitted = run.fit_adapters({"context": context, "fusion": validation}, shapes, truth, split, months)
    support, query = run.support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full_like(truth, np.nan)
    labels[support] = truth[support]
    frame = run.make_predictions(full, {"context": context, "fusion": combined}, shapes,
                                 fitted, labels, split, months)
    assert set(frame.model_name) == set(run.MODELS)
    assert "y_true" not in frame
    for (name, k), group in frame.groupby(["model_name", "k"]):
        np.testing.assert_array_equal(group.cell.to_numpy(), query)
        assert np.isfinite(group.y_pred).all()
        if k == 0:
            base = context if name.startswith("context") else combined
            np.testing.assert_array_equal(group.y_pred.to_numpy(), base[query])
    for base in ("context", "fusion"):
        predictions = frame[frame.k.eq(1) & frame.model_name.str.startswith(base)]
        wide = predictions.pivot(index="cell", columns="model_name", values="y_pred")
        np.testing.assert_array_equal(wide[f"{base}_constant"], wide[f"{base}_gru_shape"])
        np.testing.assert_array_equal(wide[f"{base}_constant"], wide[f"{base}_tree_shape"])
