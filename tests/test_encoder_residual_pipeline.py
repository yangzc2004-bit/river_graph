"""Label isolation at the encoder/ecological-memory integration boundary."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_spec = importlib.util.spec_from_file_location("encoder_runner", Path("scripts/run_doc_encoder_residual_v1.py"))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def fixture():
    months, nodes = 12, 4
    cells = np.arange(nodes * months)
    split = {"train": cells[:24], "val": cells[24:36], "test": cells[36:], "context": cells[:0]}
    truth = np.full(nodes * months, 3.)
    truth[split["val"]] = 5.
    truth[split["test"]] = 5.
    context = np.full_like(truth, 3.)
    bases = {arm: context.copy() for arm in run.ARMS}
    memory = np.zeros_like(truth)
    shapes = {"constant": np.ones((len(truth), 1))}
    full = pd.DataFrame({"cell": cells, "station": cells // months, "month": cells % months,
                         "analyte": "doc", "visibility_role": "hidden", "ecological_novelty": 0.,
                         "upstream_support": 0.})
    return context, bases, memory, shapes, truth, split, months, full


def test_mixture_and_support_choices_ignore_every_nonvalidation_label():
    context, bases, memory, shapes, truth, split, months, _ = fixture()
    changed = truth.copy()
    changed[split["train"]] = np.inf
    changed[split["test"]] = -1e100
    a = run.fit_mixers(context, bases, memory, shapes, truth, split, months)
    b = run.fit_mixers(context, bases, memory, shapes, changed, split, months)
    assert {name: model.to_dict() for name, model in a.items()} == {name: model.to_dict() for name, model in b.items()}


def test_target_adaptation_reads_reserved_support_but_never_query_truth():
    context, bases, memory, shapes, truth, split, months, full = fixture()
    mixers = run.fit_mixers(context, bases, memory, shapes, truth, split, months)
    support, _ = run.support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full_like(truth, np.nan)
    labels[support] = truth[support]
    a = run.integrated_queries(full, context, bases, memory, shapes, mixers, labels, split, months)
    changed = labels.copy()
    changed[np.isnan(changed)] = np.inf
    b = run.integrated_queries(full, context, bases, memory, shapes, mixers, changed, split, months)
    np.testing.assert_array_equal(a.y_pred, b.y_pred)
    changed[support] += 1
    c = run.integrated_queries(full, context, bases, memory, shapes, mixers, changed, split, months)
    np.testing.assert_array_equal(a[a.k.eq(0)].y_pred, c[c.k.eq(0)].y_pred)
    assert not np.array_equal(a[a.k.eq(5)].y_pred, c[c.k.eq(5)].y_pred)
