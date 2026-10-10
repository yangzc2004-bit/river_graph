"""Portable current-source DOC keeps receiver isolation and fitted components."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import test_portable_doc_reconstructor as base_fixtures
from portable_current_source_doc_v2 import PortableCurrentSourceDOC

from river_graph.experiments.provenance import sha256_file
from river_graph.models.source_current_availability import (
    AvailableSourceAttentionResidual,
)
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary

sample = base_fixtures.sample

@pytest.fixture
def current_sample(sample):
    local, inputs, dataset, split, _ = sample
    ids = np.unique(split["train"]//dataset["y"].shape[1])
    norm = local.preprocessing["regime"]
    eco = (torch.as_tensor(dataset["regime"], dtype=torch.float32)-torch.as_tensor(norm["mean"]))\
        /torch.as_tensor(norm["sd"])
    names, months = dataset["site_no"][ids], np.asarray(dataset["months"], str)
    aggregate = SourceDOCInnovationLibrary().fit(names, months, eco.numpy()[ids, 4:13], dataset["y"][ids]-3.)
    relative = SourceDOCInnovationLibrary().fit(names, months, eco.numpy()[ids, 4:13],
        np.log1p(dataset["y"][ids])-np.log1p(3.))
    settings = {**local.native._config(), "extra_dim": 41, "interaction_indices": (*local.native.interaction_indices, 38)}
    native = AvailableSourceAttentionResidual(local.native.spatial, local.native.temporal, local.native.decay, **settings)
    metadata = {"recipe": "available_real_integrated", "support_adapters": {
        str(k): {"selected": {"alpha": .5}} for k in (0, 1, 3, 5)}}
    current = PortableCurrentSourceDOC(local_helper=local, native=native, aggregate_library=aggregate,
        relative_library=relative, source_hydro=np.zeros((len(ids), len(months), 8), np.float32),
        memory_state=local.memory_state, metadata=metadata)
    _, base, view = current.prepare_inputs(inputs)
    valid = np.ones(base.shape, bool)
    native.fit(view, base, base+.2, valid, view, base, base+.2, valid,
               tail_threshold=5., selection_role="source_validation")
    with torch.no_grad():
        native.head.linear.weight.fill_(.001)
        native.head.linear.bias.fill_(.1)
        native.head.output.weight.fill_(.03)
    native.selected_scale_ = .5
    return current, inputs


def test_receiver_labels_and_future_inputs_do_not_change_earlier_predictions(current_sample):
    model, inputs = current_sample
    original = model.predict(inputs)
    changed = copy.deepcopy(inputs)
    changed.update(y=np.full(original.shape, 1e6), ph=np.full(original.shape, -1e5),
                   spec_conductance=np.full(original.shape, 8e8))
    np.testing.assert_array_equal(original, model.predict(changed))
    changed["x"][:, 10:] *= 100.
    changed["daily_features"][:, 10:] = 2.
    np.testing.assert_array_equal(original[:, :10], model.predict(changed)[:, :10])
    changed_model = copy.deepcopy(model)
    changed_model.relative_library.innovations_[:, 10:] += 50.
    changed_model.aggregate_library.innovations_[:, 10:] += 100.
    changed_model.source_hydro[:, 10:] = .75
    np.testing.assert_array_equal(original[:, :10], changed_model.predict(inputs)[:, :10])


def test_new_receiver_count_calendar_zero_prior_and_component_closure(current_sample):
    model, inputs = current_sample
    components = model.predict_components(inputs)
    np.testing.assert_allclose(components["final_pred"], components["environment_pred"]+
        components["local_temporal_correction"]+components["source_observation_correction"]+
        components["source_static_correction"]+components["river_correction"], rtol=0, atol=1e-14)
    assert components["source_support_count"].shape == (2, 16)
    assert components["attention_prior_mass"].shape == (2, 16, 2)
    assert np.count_nonzero(components["source_observation_correction"]) > 0
    one = {key: value if key == "months" else value[:1] for key, value in inputs.items()}
    np.testing.assert_allclose(model.predict(one), components["final_pred"][:1], rtol=1e-6, atol=1e-6)
    outside = copy.deepcopy(inputs)
    outside["months"] = np.arange("2100-01", "2101-05", dtype="datetime64[M]")
    result = model.predict_components(outside)
    np.testing.assert_array_equal(result["source_support_count"], 0)
    np.testing.assert_array_equal(result["attention_prior_mass"], 1.)
    assert np.isfinite(result["final_pred"]).all()


def test_save_load_and_explicit_support_are_replayable(current_sample, tmp_path):
    model, inputs = current_sample
    expected = model.predict(inputs)
    model.save(tmp_path/"current")
    restored = PortableCurrentSourceDOC.load(tmp_path/"current")
    np.testing.assert_array_equal(expected, restored.predict(inputs))
    with pytest.raises(FileExistsError):
        model.save(tmp_path/"current")
    corrected = restored.predict_with_support(inputs, k=1,
        support_cells=np.array([0, 16]), support_values=np.array([7., 8.]))
    np.testing.assert_array_equal(corrected[:, 0], expected[:, 0])
    assert not np.allclose(corrected[:, 1:], expected[:, 1:])
    np.testing.assert_array_equal(restored.predict_with_support(inputs, k=0,
        support_cells=np.array([], dtype=int), support_values=np.array([])), expected)


GEO = Path("experiments/phase4_transfer/doc_current_availability_attention_geographical_v1/runs/huc4_1013_seed42")


def _geographical_fit_assets_present():
    """Completions survive archival; replay also needs their local fit files."""
    if not (GEO/"complete.json").exists() or not (GEO/"config.json").exists():
        return False
    config = json.loads((GEO/"config.json").read_text())
    parent = Path(config["parent_run"])
    for run, stages in ((GEO, ("complete.json",)), (parent,
            ("complete.json", "backbone_complete.json", "trees_complete.json", "current_complete.json"))):
        for stage in stages:
            if not (run/stage).exists():
                return False
            files = json.loads((run/stage).read_text())["files"]
            if not all((run/name).is_file() for name in files):
                return False
    return True


@pytest.mark.skipif(not _geographical_fit_assets_present() or
    not Path("data/processed/mississippi_graph_graphfix_st357.pt").exists(),
    reason="restored local geographical fit assets (including archived forests) and ST357 required for portable replay")
def test_evaluated_geographical_prediction_replays_on_named_receivers():
    from run_doc_daily_hydro_residual_v1 import load_daily_pack
    from run_doc_geographical_confirmation_v1 import DAILY_ROOT, DATASET

    model = PortableCurrentSourceDOC.from_geographical_run(GEO)
    dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    panel = pd.read_parquet(GEO/"predictions.parquet")
    expected = panel[panel.model_name.eq("available_real_integrated")]
    t = dataset["y"].shape[1]
    ids = np.unique(expected.cell.to_numpy()//t)
    daily, _, _ = load_daily_pack(DAILY_ROOT, sha256_file(DATASET), tuple(dataset["y"].shape))
    inputs = {name: np.asarray(dataset[name])[ids] for name in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=dataset["months"], daily_features=daily[ids])
    cells = expected.cell.to_numpy()
    actual = model.predict(inputs)[np.searchsorted(ids, cells//t), cells % t]
    np.testing.assert_allclose(actual, expected.y_pred, rtol=1e-6, atol=1e-6)
