from __future__ import annotations

import numpy as np
import pytest
import torch

from river_graph.experiments.transfer import (
    K_VALUES,
    availability_tasks,
    canonical_huc8,
    isolated_view,
    load_bundle,
    validate_dataset,
)
from river_graph.experiments.transfer_data import external_raw_inventory


def toy_dataset(n=8, t=12):
    edge_index = torch.tensor([list(range(n - 1)), list(range(1, n))], dtype=torch.long)
    return {
        "site_no": [f"s{i}" for i in range(n)],
        "months": [f"2000-{m:02d}-01" for m in range(1, t + 1)],
        "y": torch.ones(n, t), "y_mask": torch.ones(n, t, dtype=torch.bool),
        "x": torch.ones(n, t, 2), "x_mask": torch.ones(n, t, 2),
        "feature_channels": ["temperature", "discharge"],
        "static": torch.ones(n, 2), "regime": torch.ones(n, 13),
        "edge_index": edge_index, "edge_attr": torch.ones(n - 1, 6),
    }


def test_validate_dataset_rejects_noncontiguous_months():
    ds = toy_dataset()
    ds["months"] = ["2000-01-01", "2000-02-01"] + [
        f"2000-{m:02d}-01" for m in range(4, 13)
    ] + ["2001-01-01"]
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_dataset(ds)


def test_validate_dataset_binds_identity_and_stats():
    summary = validate_dataset(toy_dataset())
    assert summary["stations"] == 8
    assert summary["observed_station_months"] == 96
    assert summary["largest_component_fraction"] == 1.0
    assert len(summary["identity"]["sites"]) == 64


def test_availability_tasks_are_nested_and_have_fixed_query():
    mask = np.ones((8, 12), dtype=bool)
    tasks = availability_tasks(mask, range(8), [f"2000-{m:02d}-01" for m in range(1, 13)], 42)
    assert len(tasks) == 12
    for task in tasks:
        assert set(task["support_cells_by_k"]) == {str(k) for k in K_VALUES}
        query = set(task["query_cells"])
        for k in K_VALUES:
            support = task["support_cells_by_k"][str(k)]
            assert not query.intersection(support)
            assert len(support) == k


def test_load_bundle_rejects_mismatched_target_grid(tmp_path):
    nodes = [
        {"site_no": f"s{i}", "site_tp_cd": "ST", "huc_cd": "05010001"}
        for i in range(8)
    ]
    import pandas as pd
    pd.DataFrame(nodes).to_csv(tmp_path / "nodes.csv", index=False)
    edges = pd.DataFrame({"source": [f"s{i}" for i in range(7)], "target": [f"s{i+1}" for i in range(7)]})
    edges.to_csv(tmp_path / "edges.csv", index=False)
    paths = {}
    for analyte in ("doc", "ph", "spec_conductance"):
        ds = toy_dataset()
        if analyte == "ph":
            ds["months"] = [f"2001-{m:02d}-01" for m in range(1, 13)]
        path = tmp_path / f"{analyte}.pt"
        torch.save(ds, path)
        paths[analyte] = str(path)
    with pytest.raises(ValueError, match="differs from DOC"):
        load_bundle(paths, tmp_path / "nodes.csv", tmp_path / "edges.csv")


def test_isolated_view_hides_target_query_labels():
    y = np.arange(3 * 8 * 4, dtype=float).reshape(3, 8, 4)
    observed = np.ones_like(y, dtype=bool)
    train = np.zeros_like(observed)
    val = np.zeros_like(observed)
    train[1, 4:, :2] = True
    val[1, 4:, 2:] = True
    train[0, 4:, :2] = True
    val[0, 4:, 2:] = True
    support = np.array([0, 5], dtype=int)  # target basin rows 0..3, target analyte cells
    query = np.array([2, 7], dtype=int)

    view = isolated_view(y, observed, train, val, target=2,
                         target_rows=range(4), support=support, query=query)

    assert not view["fit_mask"][2].any()
    assert not view["selection_mask"][2].any()
    assert np.all(view["support_labels"].ravel()[support] == y[2].ravel()[support])
    assert np.all(view["support_labels"].ravel()[query] == 0)
    assert np.all(view["fit_labels"][2] == 0)
    assert np.all(view["selection_labels"][2] == 0)


def test_isolated_view_rejects_target_label_leakage():
    y = np.ones((3, 4, 2), dtype=float)
    observed = np.ones_like(y, dtype=bool)
    train = np.zeros_like(observed)
    val = np.zeros_like(observed)
    train[2, 0, 0] = True
    with pytest.raises(ValueError, match="target analyte/basin"):
        isolated_view(y, observed, train, val, target=2, target_rows=[0], support=[], query=[])


def test_legacy_huc_alias_is_canonicalized_from_authoritative_metadata():
    assert canonical_huc8("510020") == "051002"
    assert canonical_huc8("101302010107") == "10130201"


def test_external_raw_inventory_rejects_incomplete_result(tmp_path):
    stations = """Location_Identifier,Location_HUCEightDigitCode,Location_Type,Location_LatitudeStandardized,Location_LongitudeStandardized\nUSGS-s1,02040104,Stream,41,-74\n"""
    station_dir = tmp_path / "stations"
    result_dir = tmp_path / "results"
    station_dir.mkdir(); result_dir.mkdir()
    for analyte in ("doc", "ph", "spec_conductance"):
        (station_dir / f"{analyte}_stations.csv").write_text(stations)
    header = "Org_Identifier,Location_Identifier,Activity_StartDate,Result_Characteristic,Result_SampleFraction,Result_Measure,Result_MeasureUnit,Result_ResultDetectionCondition,USGSpcode\n"
    for analyte in ("doc", "ph", "spec_conductance"):
        (result_dir / f"{analyte}_results.csv").write_text(header + "ERROR: INCOMPLETE DATA\n")
    report = external_raw_inventory(
        "02040104",
        {a: station_dir / f"{a}_stations.csv" for a in ("doc", "ph", "spec_conductance")},
        {a: result_dir / f"{a}_results.csv" for a in ("doc", "ph", "spec_conductance")},
        {"minimum_stations": 1, "minimum_months": 1, "minimum_station_months_per_analyte": 1},
    )
    assert report["eligible"] is False
    assert report["stage1_passed"] is False
    assert all(not row["result_complete"] for row in report["analytes"].values())
    assert all(row["station_months"] is None for row in report["analytes"].values())
