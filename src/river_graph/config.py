"""Experiment configuration loading."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mvp.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load an experiment config YAML (defaults to configs/mvp.yaml)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
