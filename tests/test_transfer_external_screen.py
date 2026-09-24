"""Raw WQP screening must distinguish missing evidence from zero observations."""

from __future__ import annotations

import csv
import hashlib
import io

import pytest

from river_graph.experiments.transfer import ANALYTES
from river_graph.experiments.transfer_data import _wqp_complete, external_raw_inventory

RESULT_COLUMNS = [
    "Org_Identifier", "Location_Identifier", "Activity_StartDate",
    "Result_Characteristic", "Result_SampleFraction", "Result_Measure",
    "Result_MeasureUnit", "Result_ResultDetectionCondition", "USGSpcode",
]
STATION_COLUMNS = [
    "Location_Identifier", "Location_HUCEightDigitCode", "Location_Type",
    "Location_LatitudeStandardized", "Location_LongitudeStandardized",
]
LIMITS = {
    "minimum_stations": 1, "minimum_months": 1,
    "minimum_station_months_per_analyte": 1,
}
COUNT_FIELDS = (
    "active_stations", "common_active_stations", "common_station_months",
    "station_months", "months", "accepted_rows",
)


def _csv_text(columns, rows=()):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def _result_row(analyte, station="s1", month="2000-01-15"):
    characteristic, fraction, value, unit, pcode = {
        "doc": ("Organic carbon", "Dissolved", "2.1", "mg/L", "00681"),
        "ph": ("pH", "Unfiltered", "7.2", "standard units", "00400"),
        "spec_conductance": ("Specific conductance", "Unfiltered", "100", "uS/cm", "00095"),
    }[analyte]
    return ["USGS", f"USGS-{station}", month, characteristic, fraction,
            value, unit, "", pcode]


def _fixture_paths(tmp_path, *, independent_stations=False):
    station_paths, result_paths = {}, {}
    for index, analyte in enumerate(ANALYTES):
        station = f"s{index}" if independent_stations else "s1"
        station_paths[analyte] = tmp_path / f"{analyte}_stations.csv"
        result_paths[analyte] = tmp_path / f"{analyte}_results.csv"
        station_paths[analyte].write_text(_csv_text(STATION_COLUMNS, [
            [f"USGS-{station}", "02040104", "Stream", "41", "-74"],
        ]))
        result_paths[analyte].write_text(_csv_text(RESULT_COLUMNS, [
            _result_row(analyte, station),
        ]))
    return station_paths, result_paths


@pytest.mark.parametrize("payload", [
    "Request blocked by the Water Quality Portal web application firewall.\n",
    "<html><body>HTTP 200, provider unavailable</body></html>\n",
    "Org_Identifier,Location_Identifier\nUSGS,USGS-s1\n",
    _csv_text(RESULT_COLUMNS) + "USGS,USGS-s1\n",
    _csv_text(RESULT_COLUMNS) + 'USGS,"unterminated record\n',
    _csv_text(RESULT_COLUMNS) + "ERROR: INCOMPLETE DATA\n",
    # A provider warning must not become valid merely because a later row exists.
    _csv_text(RESULT_COLUMNS) + "ERROR: INCOMPLETE DATA\n"
    + _csv_text(RESULT_COLUMNS, [_result_row("doc")]).split("\n", 1)[1],
])
def test_wqp_complete_rejects_unusable_payload(tmp_path, payload):
    path = tmp_path / "response.csv"
    path.write_text(payload)
    assert _wqp_complete(path) is False


def test_missing_result_path_is_not_complete(tmp_path):
    assert _wqp_complete(tmp_path / "absent.csv") is False


@pytest.mark.parametrize("payload", [
    "Request blocked by the Water Quality Portal web application firewall.\n",
    _csv_text(RESULT_COLUMNS) + "ERROR: INCOMPLETE DATA\n",
    _csv_text(RESULT_COLUMNS) + "USGS,USGS-s1\n",
])
def test_bad_http_body_is_unknown_not_zero(tmp_path, payload):
    stations, results = _fixture_paths(tmp_path)
    results["doc"].write_text(payload)
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    doc = report["analytes"]["doc"]
    assert doc["result_complete"] is False
    assert all(doc[field] is None for field in COUNT_FIELDS)
    assert doc["result_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
    assert report["eligible"] is False
    assert report["stage1_passed"] is False


def test_missing_results_remain_unknown_without_fabricated_hash(tmp_path):
    stations, results = _fixture_paths(tmp_path)
    results["ph"].unlink()
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    ph = report["analytes"]["ph"]
    assert ph["result_complete"] is False
    assert ph["result_sha256"] is None
    assert all(ph[field] is None for field in COUNT_FIELDS)
    assert report["eligible"] is False


def test_valid_header_only_is_known_zero_and_ineligible(tmp_path):
    stations, results = _fixture_paths(tmp_path)
    for path in results.values():
        path.write_text(_csv_text(RESULT_COLUMNS))
        assert _wqp_complete(path) is True
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    for row in report["analytes"].values():
        assert row["result_complete"] is True
        assert all(row[field] == 0 for field in COUNT_FIELDS)
        assert row["min_month"] is None
        assert row["max_month"] is None
    assert report["eligible"] is False
    assert report["stage1_passed"] is False


def test_analyte_specific_active_masks_can_pass_raw_screen_not_full_gate(tmp_path):
    stations, results = _fixture_paths(tmp_path, independent_stations=True)
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    assert report["common_metadata_stations"] == 0
    assert report["eligible"] is True
    assert report["stage1_passed"] is False
    assert report["selection_role"] == "availability_screen_only"
    assert report["completeness_scope"] == "structural_only_provider_total_unverified"
    for analyte, row in report["analytes"].items():
        assert row["active_stations"] == 1
        assert row["station_months"] == 1
        assert row["common_station_months"] == 0
        assert row["eligible"] is True
        station = report["station_sets"][analyte]
        assert station["path"] == str(stations[analyte])
        assert station["file_sha256"] == hashlib.sha256(stations[analyte].read_bytes()).hexdigest()
        assert row["result_sha256"] == hashlib.sha256(results[analyte].read_bytes()).hexdigest()


def test_stream_huc_coordinate_filter_and_month_deduplication(tmp_path):
    stations, results = _fixture_paths(tmp_path)
    metadata = [
        ["USGS-s1", "02040104", "Stream", "41", "-74"],
        ["USGS-lake", "02040104", "Lake", "41", "-74"],
        ["USGS-away", "02040105", "Stream", "41", "-74"],
        ["USGS-no_coord", "02040104", "Stream", "", "-74"],
        ["USGS-bad_coord", "02040104", "Stream", "100", "-74"],
    ]
    for analyte in ANALYTES:
        stations[analyte].write_text(_csv_text(STATION_COLUMNS, metadata))
        rows = [_result_row(analyte, name) for name in
                ("s1", "lake", "away", "no_coord", "bad_coord")]
        rows += [_result_row(analyte, "s1", "2000-01-20"),
                 _result_row(analyte, "s1", "2000-02-15")]
        results[analyte].write_text(_csv_text(RESULT_COLUMNS, rows))
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    for analyte, row in report["analytes"].items():
        assert report["station_sets"][analyte]["count"] == 1
        assert row["accepted_rows"] == 3
        assert row["station_months"] == 2
        assert row["months"] == 2
        assert row["min_month"] == "2000-01"
        assert row["max_month"] == "2000-02"


def test_qc_filters_invalid_ph_and_conductance_values(tmp_path):
    stations, results = _fixture_paths(tmp_path)
    for analyte, invalid_value in (("ph", "15"), ("spec_conductance", "-1")):
        row = _result_row(analyte)
        row[5] = invalid_value
        results[analyte].write_text(_csv_text(RESULT_COLUMNS, [row]))
    report = external_raw_inventory("02040104", stations, results, LIMITS)
    assert report["analytes"]["doc"]["station_months"] == 1
    for analyte in ("ph", "spec_conductance"):
        assert report["analytes"][analyte]["result_complete"] is True
        assert report["analytes"][analyte]["station_months"] == 0
    assert report["eligible"] is False
