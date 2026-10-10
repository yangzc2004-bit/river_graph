"""Actual readout isolation and explicit conditional, not full OOF, semantics."""
import numpy as np
import pytest
import torch

from river_graph.models.conditional_readout_crossfit_v2 import (
    combine_readout,
    conditional_crossfit,
    fit_readout,
    readout_design,
)


def toy():
    rng = np.random.default_rng(44)
    cells = np.arange(6*4)
    x = rng.normal(size=(len(cells), 5)).astype(np.float32)
    base = np.full(len(cells), 3.)
    y = base+.4*x[:, 0]
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    return x, base, y, cells, folds


def test_local_nonnegative_clip_precedes_memory_blending():
    head = torch.nn.Linear(1, 1)
    with torch.no_grad():
        head.weight.fill_(0.)
        head.bias.fill_(-2.)
    tree = np.array([1.])
    gamma, memory_delta = .25, np.array([.5])
    offset = tree+gamma*memory_delta
    actual = combine_readout(np.zeros((1, 1), np.float32), head, offset, .75,
        tree=tree, memory_gamma=gamma)
    expected = np.maximum(0., tree+.75*(np.maximum(0., tree-2.)-tree)+gamma*memory_delta)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == .375


def test_held_labels_do_not_enter_the_new_readout_loss():
    x, base, y, cells, folds = toy()
    pred, heads, records = conditional_crossfit(x, base, y, cells, folds, n_months=4, epochs=3)
    changed = y.copy()
    changed[cells//4 < 2] = np.nan
    altered, changed_heads, _ = conditional_crossfit(x, base, np.nan_to_num(changed, nan=100.),
        cells, folds, n_months=4, epochs=3)
    np.testing.assert_array_equal(pred[:8], altered[:8])
    for key in heads[0]:
        torch.testing.assert_close(heads[0][key], changed_heads[0][key], rtol=0, atol=0)
    head, _ = fit_readout(x, base, changed, np.arange(8, 24), epochs=3)
    np.testing.assert_array_equal(pred[:8], combine_readout(x[:8], head, base[:8], 1.))
    for record in records:
        assert not set(record['held_station_ids']) & set(record['fitted_station_ids'])
        assert record['readout_loss_is_station_held']
        assert record['complete_model_is_oof'] is False


def test_zero_readout_and_checkpoint_replay(tmp_path):
    x, base, y, _, _ = toy()
    head, _ = fit_readout(x, base, y, np.arange(len(y)), epochs=0)
    np.testing.assert_array_equal(combine_readout(x, head, base, .75), base)
    head, trace = fit_readout(x, base, y, np.arange(len(y)), epochs=4)
    assert len(trace) == 4
    torch.save(head.state_dict(), tmp_path/'head.pt')
    restored = torch.nn.Linear(5, 1)
    restored.load_state_dict(torch.load(tmp_path/'head.pt', weights_only=True))
    np.testing.assert_array_equal(combine_readout(x, head, base, .75), combine_readout(x, restored, base, .75))


def test_overlapping_or_missing_station_folds_rejected():
    x, base, y, cells, _ = toy()
    with pytest.raises(ValueError, match='partition'):
        conditional_crossfit(x, base, y, cells, [np.array([0, 1]), np.array([1, 2])], n_months=4)
    with pytest.raises(ValueError, match='unique permitted'):
        fit_readout(x, base, y, np.array([0, 0]))


def test_export_matches_real_head_and_does_not_read_future_covariates():
    from copy import deepcopy

    from test_current_source_attention import _model

    from river_graph.models.source_current_availability import (
        AvailableSourceAttentionResidual,
    )

    old, _, args = _model()
    source, val = args[0], args[4]
    for view in (source, val):
        view['attention_reference'] = np.full(view['age'].shape, 3.)
    model = AvailableSourceAttentionResidual(old.spatial, old.temporal, old.decay,
        **old._config(), **old.attention_config)
    model.fit(*args, tail_threshold=4., selection_role='source_validation')
    cells = np.array([0, 2])
    design = readout_design(model, val, cells)
    head = torch.nn.Linear(design.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.cat([model.head.linear.weight, model.head.output.weight], -1))
        head.bias.copy_(model.head.linear.bias)
        expected = model._delta_cells(model._prepare_inputs(val), cells).numpy()
        actual = head(torch.as_tensor(design)).squeeze(-1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    changed = deepcopy(val)
    for key in ('raw', 'age', 'support', 'extra'):
        changed[key][:, 3:] += 123.
    for key in ('donor_values', 'donor_hydro_bank'):
        changed[key][:, 3:] = 0.
    np.testing.assert_array_equal(readout_design(model, changed, cells), design)
