"""The execution amendment cannot exceed its neural-memory cap."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_scheduler():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("dynamic_resources", scripts / "resume_doc_dynamic_resources_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.select_pending


def test_memory_cap_fills_tree_slot():
    choose = load_scheduler()
    pending = [{"arm": "nn"}, {"arm": "rf"}, {"arm": "nn"}]
    assert choose(pending, ["nn"], 4, 4) == 1


def test_available_neural_slot_preserves_order():
    choose = load_scheduler()
    assert choose([{"arm": "nn"}, {"arm": "rf"}], ["nn"], 3, 4) == 0


def test_full_neural_cap_waits():
    choose = load_scheduler()
    assert choose([{"arm": "nn"}], ["nn"], 4, 4) is None
