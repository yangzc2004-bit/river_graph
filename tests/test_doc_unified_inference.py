"""A receiving reconstruction must remain bound to its source normalization."""
import importlib
import json
from pathlib import Path

import pytest

from river_graph.experiments.provenance import sha256_file


@pytest.mark.parametrize("rewrite_receipt", [False, True])
def test_receiving_inference_rejects_altered_target_statistics(tmp_path, monkeypatch, rewrite_receipt):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    runner = importlib.import_module("reconstruct_doc_unified_v2")
    protocol = {"tasks": [{"id": "primary", "arm": "direction_heads"}]}
    choice = {"primary_checkpoint_id": "primary"}
    directory = tmp_path/"fits/primary"
    directory.mkdir(parents=True)
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    (tmp_path/"selected_model.json").write_text(json.dumps(choice))
    (directory/"info.json").write_text(json.dumps({"target_normalization": {"mean": 1., "std": .3}}))
    receipt = {"files": {"info.json": sha256_file(directory/"info.json")}}
    (directory/"selection_complete.json").write_text(json.dumps(receipt))
    gate = {"protocol_hash": runner.digest(protocol), "selected_model_sha256": sha256_file(tmp_path/"selected_model.json"),
        "fit_receipts": {"primary": sha256_file(directory/"selection_complete.json")}}
    (tmp_path/"all_selections.json").write_text(json.dumps(gate))
    (directory/"info.json").write_text(json.dumps({"target_normalization": {"mean": 9., "std": .3}}))
    if rewrite_receipt:
        receipt["files"]["info.json"] = sha256_file(directory/"info.json")
        (directory/"selection_complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="fit altered|changed package artifact"):
        runner.reconstruct(tmp_path, tmp_path/"nonexistent_inputs.npz", tmp_path/"predictions.npz")
    assert not (tmp_path/"predictions.npz").exists()
