"""Current-site subtraction, causal lookup and unchanged matched inputs."""
import numpy as np
import pytest
import torch

from river_graph.models.dynamic_river_state import DynamicRiverState
from river_graph.models.river_state_contrast import receiver_contrast


def inputs():
    state = np.arange(3*8*9, dtype=np.float32).reshape(3, 8, 9)
    cells = np.array([2*8+3, 2*8+5])
    values = state[np.array([0, 1])[:, None, None], np.array([[3, 2, 0], [5, 4, 2]])[:, None]].copy()
    valid = np.ones((2, 1, 3), bool)
    valid[0, 0, 2] = False
    values[~valid] = 0.
    return {'states': values, 'valid': valid, 'features': np.zeros((2, 1, 3, 36), np.float32),
            'query': np.ones((2, 90), np.float32)}, cells, state


def test_receiving_time_and_other_inputs_are_exact():
    arrays, cells, states = inputs()
    original = arrays['states'].copy()
    actual = receiver_contrast(arrays, cells, states)
    expected = np.where(arrays['valid'][..., None],
        original-states.reshape(-1, 9)[cells, None, None], 0.)
    np.testing.assert_array_equal(actual['states'], expected)
    np.testing.assert_array_equal(arrays['states'], original)
    for key in ('valid', 'features', 'query'):
        assert actual[key] is arrays[key]
    assert not actual['states'][~arrays['valid']].any()
    inplace = receiver_contrast(arrays, cells, states, copy=False)
    assert inplace['states'] is arrays['states']
    np.testing.assert_array_equal(inplace['states'], expected)


def test_future_receiver_states_do_not_enter_current_contrast():
    arrays, cells, states = inputs()
    baseline = receiver_contrast(arrays, cells, states)
    states[:, 6:] += 12345
    altered = receiver_contrast(arrays, cells, states)
    np.testing.assert_array_equal(baseline['states'], altered['states'])
    states[2, 3] += 2
    changed = receiver_contrast(arrays, cells, states)
    np.testing.assert_array_equal(changed['states'][1], altered['states'][1])
    np.testing.assert_array_equal(changed['states'][0, 0, :2], altered['states'][0, 0, :2]-2)


def test_contrast_zero_head_and_save_load_preserve_anchor(tmp_path):
    arrays, cells, states = inputs()
    contrast = receiver_contrast(arrays, cells, states)
    base = np.array([2.3, 8.9])
    model = DynamicRiverState(90, 36, 9, seed=42, epochs=2, patience=2)
    np.testing.assert_array_equal(model.predict(contrast, base), base)
    model.fit(contrast, base, base+.2, contrast, base, base+.2, tail_threshold=5.)
    predicted = model.predict(contrast, base)
    torch.save(model.to_payload(), tmp_path/'model.pt')
    restored = DynamicRiverState.from_payload(torch.load(tmp_path/'model.pt', weights_only=False))
    np.testing.assert_array_equal(restored.predict(contrast, base), predicted)
    silent = {**contrast, 'valid': np.zeros_like(contrast['valid']),
              'states': contrast['states']*0, 'features': contrast['features']*0}
    np.testing.assert_array_equal(restored.predict(silent, base), base)


def test_invalid_global_cell_is_rejected():
    arrays, cells, states = inputs()
    cells[0] = -1
    with pytest.raises(ValueError, match='global receiving'):
        receiver_contrast(arrays, cells, states)
