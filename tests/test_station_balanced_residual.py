"""Station-level monitoring effort changes loss, not model information flow."""
from __future__ import annotations

import numpy as np
import torch
from test_encoder_native_residual import fixture

from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.station_balanced_residual import StationBalancedEncoderResidual


def balanced_fixture():
    old, arrays, *_ = fixture("last_self_ecology", epochs=2)
    model = StationBalancedEncoderResidual(old.spatial, old.temporal, old.decay, **old._config())
    return model, arrays


def test_station_totals_equal_despite_monitoring_effort_and_tail_composition():
    model, _ = balanced_fixture()
    cells = np.array([0, 1, 2, 3, 14, 28, 29])
    truth = torch.tensor([1., 1., 5., 5., 1., 1., 5.], dtype=torch.float64)
    weights = model._source_weights(cells, truth, truth >= 4., 14)
    totals = [float(weights[cells//14 == i].sum()) for i in range(3)]
    np.testing.assert_allclose(totals, [7/3]*3)
    assert float(weights[2]/weights[0]) == 2
    assert float(weights[6]/weights[5]) == 2
    assert float(weights.mean()) == 1
    # Duplicating an ordinary observation cannot increase its station's total.
    duplicated = np.r_[cells, 4]
    new_y = torch.cat([truth, truth[:1]])
    changed = model._source_weights(duplicated, new_y, new_y >= 4., 14)
    np.testing.assert_allclose([float(changed[duplicated//14 == i].sum()) for i in range(3)], [8/3]*3)


def test_station_balanced_fit_save_load_and_hidden_label_isolation():
    a, args = balanced_fixture()
    b, changed = balanced_fixture()
    changed[2][~changed[3]] = 1e30
    changed[6][~changed[7]] = -1e30
    for model, data in ((a, args), (b, changed)):
        model.fit(*data, tail_threshold=4., selection_role="source_validation")
    np.testing.assert_array_equal(a.predict(args[4], args[5]), b.predict(changed[4], changed[5]))
    payload = a.to_payload()
    # Inference remains compatible with the existing portable encoder loader.
    for cls in (StationBalancedEncoderResidual, EncoderNativeResidual):
        loaded = cls.from_payload(payload)
        np.testing.assert_array_equal(loaded.predict(args[4], args[5]), a.predict(args[4], args[5]))
    assert "equal total station" in payload["summary"]["training_weighting"]
