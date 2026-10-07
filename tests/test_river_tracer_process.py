"""Checks for isotope response, observation grain and unsupported gap handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from river_graph.analysis.river_tracer_process import (
    STANDARD_RATIO,
    atom_fraction,
    bounded_response,
    clock_seconds,
    paired_joint_area,
    pulse_core_metrics,
    resolve_clock_points,
)


def test_isotope_conversion_against_scalar_ratio():
    delta = np.array([-25.0, 0.0, 1000.0, np.nan])
    expected = []
    for value in delta:
        ratio = (value / 1000 + 1) * STANDARD_RATIO
        expected.append(ratio / (ratio + 1))
    np.testing.assert_allclose(atom_fraction(delta), expected, rtol=0, atol=0)
    assert np.isnan(atom_fraction(delta)[-1])
    with pytest.raises(ValueError, match="negative isotope"):
        atom_fraction([-1001])


def test_rounded_csv_minutes_match_real_second_clocks_without_creating_new_samples():
    actual = [25 + 5/60, 66 + 10/60]
    exported = [25.0833333333333, 66.1666666666667]
    np.testing.assert_array_equal(clock_seconds(actual), clock_seconds(exported))
    with pytest.raises(ValueError, match="whole-second"):
        clock_seconds([1.012])


def test_triangle_area_quantiles_and_clock_shift():
    result = bounded_response([10, 11, 12], [0, 1, 0])
    assert result["area"] == 1
    assert result["centroid_min"] == 11
    assert result["duration80_min"] == pytest.approx(2 - 2*np.sqrt(0.2))
    translated = bounded_response([17, 18, 19], [0, 1, 0])
    assert translated["centroid_min"] - result["centroid_min"] == 7
    assert translated["duration80_min"] == pytest.approx(result["duration80_min"])
    assert translated["sampled_peak_first_min"] == 18


def test_missing_measurement_does_not_create_a_bridge():
    result = bounded_response([0, 1, 2, 3], [0, np.nan, 1, 0])
    assert result["area"] == 0.5
    assert result["covered_min"] == 1
    assert result["gap_count"] == 2
    assert result["centroid_min"] == pytest.approx(7/3)


def test_large_gap_is_not_extrapolated():
    result = bounded_response([0, 10, 100, 110], [0, 1, 1, 0], max_gap=30)
    assert result["area"] == 10
    assert result["covered_min"] == 20
    assert result["gap_count"] == 1
    assert result["span_min"] == 110


def test_reference_and_doc_integrate_identical_supported_segments():
    result = paired_joint_area([0, 1, 2, 3], [0, 2, 4, 0], [0, np.nan, 1, 0])
    assert result["joint_reference_area"] == 2
    assert result["joint_doc_area"] == 0.5
    assert result["joint_doc_reference_fraction"] == 0.25
    assert result["joint_covered_min"] == 1


def test_core_fraction_is_not_a_mass_recovery_estimate():
    time = [0, 1, 2, 3, 4]
    ref = np.array([0, 1, 4, 2, 0], dtype=float)
    result = pulse_core_metrics(time, ref, 0.3*ref)
    assert result["core_slope"] == pytest.approx(0.3)
    assert result["core_ratio_median"] == pytest.approx(0.3)
    assert result["n_core_points"] == 3
    missing = pulse_core_metrics(time, ref, [0, np.nan, np.nan, np.nan, 0])
    assert missing["n_core_points"] == 0
    assert np.isnan(missing["core_slope"])


def test_paired_core_ratio_cancels_shared_reference_calibration():
    t = [0, 1, 2, 3]
    salt_up = np.array([0, 2, 4, 0], dtype=float)
    salt_down = np.array([0, 1, 2, 0], dtype=float)
    doc_up, doc_down = 0.6*salt_up, 0.2*salt_down
    ratios = []
    for calibration in [0.2, 1.0, 7.0]:
        up = pulse_core_metrics(t, salt_up*calibration, doc_up)["core_slope"]
        down = pulse_core_metrics(t, salt_down*calibration, doc_down)["core_slope"]
        ratios.append(down/up)
    np.testing.assert_allclose(ratios, 1/3)


def test_conflicting_salt_is_unresolved_but_lab_replicates_are_averaged():
    source = pd.DataFrame({"time_min": [0, 0, 1, 1], "spc_raw": [10, 11, 20, 20],
                           "doc_label_raw": [0.1, 0.3, 0.8, np.nan],
                           "doc_total_raw": [1, 3, 8, np.nan], "record_id": [1, 2, 3, 4],
                           "is_lab_record": [True, True, True, False],
                           "source_notes": [None, "conflict", None, None]})
    result = resolve_clock_points(source)
    assert result.salt_conflict.tolist() == [True, False]
    assert np.isnan(result.spc_raw.iloc[0])
    assert result.doc_label_raw.iloc[0] == pytest.approx(0.2)
    assert result.n_lab_measured.tolist() == [2, 1]
    assert result.n_lab_records.tolist() == [2, 1]
    assert result.record_ids.tolist() == ["1|2", "3|4"]


def test_invalid_clock_and_limits_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        bounded_response([1, 1], [0, 1])
    with pytest.raises(ValueError, match="limits"):
        bounded_response([0, 1], [0, 1], max_gap=0)
    with pytest.raises(ValueError, match="Unaligned"):
        pulse_core_metrics([0], [0, 1], [0, 1])


def test_saved_download_manifest_restores_pinned_sources_and_preserves_conflicts(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import json
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("tracer_fetch", Path("scripts/fetch_doc_river_tracer_process_v1.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RAW", tmp_path / "data/raw/river_tracer_process_v1")
    monkeypatch.setattr(module, "OUT", tmp_path / "experiment")
    module.OUT.mkdir()
    content = b"actual,pinned,source\n"
    url = "https://raw.githubusercontent.com/robohall/DOC_uptake/pinned-commit/data_doc.csv"
    source = module.RAW / "author/data_doc.csv"
    manifest = module.OUT / "retrieval_manifest.json"
    item = {"path": str(source.relative_to(tmp_path)), "url": url,
            "sha256": hashlib.sha256(content).hexdigest()}
    manifest.write_text(json.dumps({"objects": [item]}))
    original_manifest = manifest.read_bytes()
    requests = []

    def get(expected):
        requests.append(expected)
        return SimpleNamespace(content=content)

    def new_ref_is_forbidden(*args, **kwargs):
        raise AssertionError("Restoring a saved source must not resolve a new author branch")

    monkeypatch.setattr(module, "get", get)
    monkeypatch.setattr(module.subprocess, "run", new_ref_is_forbidden)
    module.main()
    assert source.read_bytes() == content
    assert requests == [url]
    assert manifest.read_bytes() == original_manifest
    module.main()
    assert requests == [url]
    source.write_bytes(b"conflicting existing file")
    with pytest.raises(ValueError, match="Changed local public source"):
        module.main()
    assert source.read_bytes() == b"conflicting existing file"
    assert requests == [url]


@pytest.mark.skipif(
    not all(Path(path).exists() for path in [
        "data/raw/river_tracer_process_v1/blaine/site_data.csv",
        "data/raw/river_tracer_process_v1/blaine/13CAdditionData_CrestonGlucose_20190808_analyzed.csv",
        "data/raw/river_tracer_process_v1/blaine/13CAdditionData_CrestonGlucose_20190815_analyzed.csv",
        "data/raw/river_tracer_process_v1/blaine/13CAdditionData_CrestonLeachate_20190809_analyzed.csv",
        "data/raw/river_tracer_process_v1/author/data_doc.csv",
    ]),
    reason="Public Blaine paired-tracer raw archive is not present locally",
)
def test_original_three_additions_and_no_raw_doc_imputation():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("tracer_analysis", Path("scripts/analyze_doc_river_tracer_process_v1.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables, products, summary = module.build()
    assert summary["n_series"] == 6
    assert summary["n_additions"] == 3
    assert len(tables["paired_responses"]) == 3
    raw = products["original_sample_points"]
    unavailable = raw.doc_total_raw.isna() | raw.delta13c_raw.isna()
    assert raw.loc[unavailable, "doc_label_raw"].isna().all()
    source = pd.read_csv("data/raw/river_tracer_process_v1/blaine/13CAdditionData_CrestonGlucose_20190808_analyzed.csv")
    up = raw[(raw.code == "b_g1_up") & raw.record_id.eq(1)].iloc[0]
    assert up.time_min == 6
    assert np.isnan(up.spc_raw)
    assert up.doc_total_raw == source.DOC_conc.iloc[0]
    assert tables["author_processing_comparison"].n_author_only_clocks.eq(0).all()
    added = tables["author_processing_comparison"].set_index("code").n_author_carbon_without_raw_lab
    assert added["b_l_up"] == 3
    assert added["b_l_down"] == 0
