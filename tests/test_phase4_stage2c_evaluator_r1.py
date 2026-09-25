import pandas as pd
import pytest

from scripts.evaluate_phase4_stage2c_controls_r1 import (
    expected_product_keys,
    validate_prediction_frame,
)


def _manifest():
    return {
        "tasks": [
            {
                "basin": "b",
                "seed": 42,
                "analyte": "doc",
                "task_index": 0,
                "month": "2000-01",
                "month_index": 0,
                "k": 0,
                "support_count": 0,
                "query_cells": [2, 3],
            },
            {
                "basin": "b",
                "seed": 42,
                "analyte": "doc",
                "task_index": 0,
                "month": "2000-01",
                "month_index": 0,
                "k": 5,
                "support_count": 5,
                "query_cells": [2, 3],
            },
        ]
    }


def _unit():
    return {"unit": "ecorf__doc__b__seed42", "arm": "ecorf", "analyte": "doc", "basin": "b", "seed": 42, "config_hash": "c"}


def _frame():
    return pd.DataFrame(
        [
            {"task_index": 0, "month": "2000-01", "month_index": 0, "k": 0, "flat": 2, "support_count": 0,
             "y_pred": 1.0, "config_hash": "c", "query_labels_used_for_prediction": False,
             "model_name": "ecorf", "analyte": "doc", "basin": "b", "task_seed": 42},
            {"task_index": 0, "month": "2000-01", "month_index": 0, "k": 0, "flat": 3, "support_count": 0,
             "y_pred": 1.0, "config_hash": "c", "query_labels_used_for_prediction": False,
             "model_name": "ecorf", "analyte": "doc", "basin": "b", "task_seed": 42},
            {"task_index": 0, "month": "2000-01", "month_index": 0, "k": 5, "flat": 2, "support_count": 5,
             "y_pred": 1.0, "config_hash": "c", "query_labels_used_for_prediction": False,
             "model_name": "ecorf", "analyte": "doc", "basin": "b", "task_seed": 42},
            {"task_index": 0, "month": "2000-01", "month_index": 0, "k": 5, "flat": 3, "support_count": 5,
             "y_pred": 1.0, "config_hash": "c", "query_labels_used_for_prediction": False,
             "model_name": "ecorf", "analyte": "doc", "basin": "b", "task_seed": 42},
        ]
    )


def test_expected_inventory_and_valid_frame():
    expected = expected_product_keys(_manifest(), "b", 42, "doc")
    validate_prediction_frame(_frame(), _unit(), expected)


@pytest.mark.parametrize("column", ["y_true", "query_label"])
def test_forbidden_label_columns_fail(column):
    frame = _frame()
    frame[column] = 0.0
    with pytest.raises(ValueError, match="label column"):
        validate_prediction_frame(frame, _unit(), expected_product_keys(_manifest(), "b", 42, "doc"))


def test_missing_or_duplicate_inventory_fails():
    frame = _frame().iloc[:-1].copy()
    with pytest.raises(ValueError, match="inventory"):
        validate_prediction_frame(frame, _unit(), expected_product_keys(_manifest(), "b", 42, "doc"))
    frame = _frame().copy()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_frame(frame, _unit(), expected_product_keys(_manifest(), "b", 42, "doc"))
