"""A longer stopping allowance preserves the common optimization trajectory."""
from __future__ import annotations

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual


def test_extended_schedule_preserves_prefix_and_saved_prediction():
    short, args, *_ = fixture("last_self_ecology", epochs=3)
    long = EncoderNativeResidual(short.spatial, short.temporal, short.decay,
        **{**short._config(), "epochs": 6, "patience": 15})
    for model in (short, long):
        model.fit(*args, tail_threshold=4., selection_role="source_validation")
    assert long.trace_[:len(short.trace_)] == short.trace_
    assert long.epochs_run_ > short.epochs_run_
    assert long.validation_metrics_["validation_mae"] <= short.validation_metrics_["validation_mae"]
    restored = EncoderNativeResidual.from_payload(long.to_payload())
    np.testing.assert_array_equal(long.predict(args[4], args[5]),
                                  restored.predict(args[4], args[5]))
    for name in ("spatial", "temporal", "decay"):
        a = getattr(short, f"_initial_{name}_state")
        b = getattr(long, f"_initial_{name}_state")
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
