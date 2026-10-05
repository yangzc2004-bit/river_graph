"""Donor fusion tests: preserve the old predictor and isolate source values."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_doc_source_retrieval_v2 import donor_contrast, fit_contrast, predict_contrast

from river_graph.models.doc_source_retrieval import SourceRetrievalAttention


def sample():
    rng = np.random.default_rng(42)
    return {"query": rng.normal(size=(12, 89)).astype(np.float32),
            "keys": rng.normal(size=(12, 3, 28)).astype(np.float32),
            "values": np.ones((12, 3, 7), dtype=np.float32), "valid": np.ones((12, 3), bool),
            "base": np.full(12, 4.), "context": np.full(12, 3.)}


def test_donor_contrast_does_not_add_the_forest_residual_twice():
    inputs = sample()
    contrasted = donor_contrast(inputs, inputs["context"], inputs["base"], 2.)
    np.testing.assert_array_equal(contrasted["values"][..., 0], .5)
    np.testing.assert_array_equal(contrasted["values"][..., 1:], 0)
    np.testing.assert_array_equal(inputs["values"], 1)
    model = SourceRetrievalAttention(89)
    delta, *_ = predict_contrast(model, contrasted, scale=2.)
    np.testing.assert_array_equal(inputs["base"]+delta, inputs["base"])


def test_donor_removal_retains_the_local_subtraction():
    inputs = sample()
    contrasted = donor_contrast(inputs, inputs["context"], inputs["base"], 2.)
    model = SourceRetrievalAttention(89, dropout=0.).eval()
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.weight[0, 0] = 1
    delta, *_ = predict_contrast(model, contrasted, scale=2., ablation="zero_source_values")
    np.testing.assert_allclose(delta, -np.ones(12), rtol=2e-7, atol=0)


def test_contrast_training_and_nonzero_checkpoint_round_trip():
    inputs = sample()
    inputs = donor_contrast(inputs, inputs["context"], inputs["base"], 2.)
    training = {**inputs, "truth": np.full(12, 4.5), "weights": np.ones(12)}
    model, selection = fit_contrast([training], training, scale=2., epochs=3)
    assert selection["best_epoch"] > 0
    assert selection["validation_mae"] < .5
    restored = SourceRetrievalAttention.from_payload(model.to_payload())
    np.testing.assert_array_equal(predict_contrast(model, inputs, scale=2.)[0],
                                  predict_contrast(restored, inputs, scale=2.)[0])
