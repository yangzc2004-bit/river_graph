"""Source hydro isolation, compatible initialization and causal source lookup."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_current_source_attention import _model

from river_graph.models.source_level_attention import SourceLevelAttentionResidual
from river_graph.models.source_monthly_hydro_attention import (
    SourceMonthlyHydroAttentionResidual,
    source_monthly_hydro_bank,
)


def fixture(mode="current"):
    current, _, args = _model()
    rng = np.random.default_rng(236)
    bank = rng.normal(size=(2, 14, 6))
    bank[..., [1, 3]] = 1.
    bank[0, :2, 1] = 0.
    for inputs in (args[0], args[4]):
        inputs["attention_reference"] = np.full(inputs["age"].shape, 3.)
        inputs["donor_monthly_hydro_bank"] = source_monthly_hydro_bank(bank, mode)
    settings = {**current._config(), **current.attention_config}
    old = SourceLevelAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    new = SourceMonthlyHydroAttentionResidual(current.spatial, current.temporal, current.decay, **settings)
    return old, new, args


def test_monthly_bank_uses_only_hydro_and_masks_hidden_values():
    raw = np.arange(2*14*9.).reshape(2, 14, 9)
    raw[..., [1, 3]] = 1.
    raw[0, :3, 1] = 0.
    current = source_monthly_hydro_bank(raw)
    changed = raw.copy()
    changed[..., 4:] = np.nan
    changed[0, :3, 0] = -1000.
    np.testing.assert_array_equal(current, source_monthly_hydro_bank(changed))
    assert not current[0, :3, 0].any()
    mean = source_monthly_hydro_bank(raw, "source_mean")
    np.testing.assert_array_equal(mean[..., [1, 3]], current[..., [1, 3]])
    np.testing.assert_allclose(mean[0, 3:, 0], current[0, 3:, 0].mean())
    assert not source_monthly_hydro_bank(raw, "zero").any()
    raw[0, 0, 1] = .5
    with pytest.raises(ValueError, match="binary"):
        source_monthly_hydro_bank(raw)


@pytest.mark.parametrize("mode", ["current", "source_mean", "zero"])
def test_initial_attention_and_training_reload_future_new_station(mode):
    old, new, args = fixture(mode)
    np.testing.assert_array_equal(old.predict(args[4], args[5]), new.predict(args[4], args[5]))
    np.testing.assert_array_equal(old.head.query.weight.detach(), new.head.query.weight.detach())
    np.testing.assert_array_equal(old.head.key.weight.detach(), new.head.key.weight[:, :-4].detach())
    assert new.trainable_parameter_count_ == old.trainable_parameter_count_ + 256
    assert not torch.count_nonzero(new.head.key.weight[:, -4:])
    new.fit(*args, tail_threshold=4., selection_role="source_validation")
    restored = SourceMonthlyHydroAttentionResidual.from_payload(new.to_payload())
    np.testing.assert_array_equal(new.predict(args[4], args[5]), restored.predict(args[4], args[5]))
    if mode != "zero":
        assert torch.count_nonzero(new.head.key.weight[:, -4:])
    future = deepcopy(args[4])
    for key in ("raw", "extra", "donor_values", "attention_reference"):
        future[key][:, 8:] += 100.
    future["donor_values"][..., -1] = 0.
    future["donor_values"][~future["donor_valid"]] = 0.
    for value, visible in ((0, 1), (2, 3)):
        future["donor_monthly_hydro_bank"][:, 8:, value] += (
            100. * future["donor_monthly_hydro_bank"][:, 8:, visible])
    future["donor_hydro_bank"][:, 8:] += 100.
    np.testing.assert_array_equal(new.predict_delta(args[4])[:, :8], restored.predict_delta(future)[:, :8])
    extended = deepcopy(args[4])
    for key in extended:
        if key not in ("donor_hydro_bank", "donor_monthly_hydro_bank"):
            extended[key] = np.concatenate([extended[key], extended[key][:1]], axis=0)
    assert restored.predict_delta(extended).shape == (3, 14)
