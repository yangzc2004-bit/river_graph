"""Contract tests for the metrics-only Stage-2 decomposition evaluator."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_phase4_stage2_information_decomposition import (
    TASK_KEYS,
    _pair,
    _read_metrics,
    _verify_shared_keys,
)


def _metrics(model: str, values: list[float], *, k: int = 0) -> pd.DataFrame:
    rows = []
    for i, value in enumerate(values):
        rows.append(
            {
                "analyte": "doc",
                "basin": "510020",
                "task_seed": 42,
                "task_index": i,
                "month": f"2000-{i + 1:02d}",
                "k": k,
                "model_name": model,
                "mae": value,
            }
        )
    return pd.DataFrame(rows)


def test_pair_delta_uses_comparator_minus_reference() -> None:
    left = _metrics("full", [1.0, 2.0])
    right = _metrics("ref", [2.0, 3.0])
    pair = _pair(
        left,
        right,
        comparator_name="full",
        reference_name="ref",
        label="x",
        k=0,
    )
    assert pair["delta_mae"].tolist() == [-1.0, -1.0]
    assert (pair["delta_mae"] == pair["comparator_mae"] - pair["reference_mae"]).all()


def test_pair_rejects_missing_task_keys() -> None:
    left = _metrics("full", [1.0, 2.0])
    right = _metrics("ref", [2.0])
    with pytest.raises(ValueError, match="unpaired"):
        _pair(
            left,
            right,
            comparator_name="full",
            reference_name="ref",
            label="x",
            k=0,
        )


def test_shared_keys_reject_different_task_inventory() -> None:
    left = pd.concat([_metrics("climatology", [1.0])], ignore_index=True)
    right = _metrics("h2x_full", [1.0], k=0)
    right.loc[0, "task_index"] = 99
    with pytest.raises(ValueError, match="task key sets differ"):
        _verify_shared_keys(left, right)


def test_metrics_reader_rejects_query_labels(tmp_path) -> None:
    frame = _metrics("climatology", [1.0])
    frame["y_true"] = 1.0
    path = tmp_path / "metrics.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="forbidden label"):
        _read_metrics(path, "synthetic")


def test_metrics_reader_canonicalizes_basin_and_requires_finite_mae(tmp_path) -> None:
    frame = _metrics("climatology", [1.0])
    path = tmp_path / "metrics.csv"
    frame.to_csv(path, index=False)
    loaded = _read_metrics(path, "synthetic")
    assert loaded.loc[0, "basin"] == "510020"
    assert set(TASK_KEYS).issubset(loaded.columns)
    bad = frame.copy()
    bad.loc[0, "mae"] = np.nan
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid MAE"):
        _read_metrics(path, "synthetic")
