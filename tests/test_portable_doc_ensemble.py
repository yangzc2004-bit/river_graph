"""The portable ensemble reproduces its fixed native mean and support policy."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from portable_doc_ensemble_v1 import PortableDOCEnsemble
from test_portable_doc_reconstructor import sample as fixture_model


@pytest.fixture
def ensemble_sample():
    return fixture_model.__wrapped__()


def test_ensemble_mean_uses_source_policy_and_round_trips(ensemble_sample, tmp_path):
    member, inputs, *_ = ensemble_sample
    members = {seed: copy.deepcopy(member) for seed in (42, 43, 44, 45, 46)}
    policy = {"primary_k0_log_half_width": .3,
        "curve": {str(k): {"selected": {"alpha": 0. if k <= 1 else .5}, "log_half_width": .2}
                  for k in (0, 1, 3, 5)}}
    ensemble = PortableDOCEnsemble(members, policy)
    base = ensemble.predict(inputs)
    np.testing.assert_allclose(base, member.predict(inputs), rtol=1e-12, atol=1e-12)
    poisoned = copy.deepcopy(inputs)
    poisoned.update(y=np.full(base.shape, 1e20), ph=np.zeros(base.shape), spec_conductance=np.zeros(base.shape))
    np.testing.assert_array_equal(base, ensemble.predict(poisoned))
    supports = np.array([0, base.shape[1]])
    result = ensemble.predict_with_support(inputs, k=1, support_cells=supports, support_values=np.array([20., 30.]))
    np.testing.assert_array_equal(result["final_pred"], base)  # Source ensemble selected alpha0.
    ensemble.save(tmp_path/"ensemble")
    restored = PortableDOCEnsemble.load(tmp_path/"ensemble")
    np.testing.assert_array_equal(restored.predict(inputs), base)
    assert np.all(restored.predict_components(inputs)["pi_lower"] <= base)
