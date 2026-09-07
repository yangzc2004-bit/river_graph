"""Unit tests for NWIS RDB parsing (no network access)."""

import pandas as pd

from river_graph.data.nwis import read_rdb

RDB_SAMPLE = """#
# US Geological Survey
#
agency_cd\tsite_no\tstation_nm\thuc_cd
5s\t15s\t50s\t16s
USGS\t05200020\tMISSISSIPPI RIVER AT HWY 200\t07010101
USGS\t05200170\tMISSISSIPPI RIVER NEAR VERN\t07010101
"""


def test_read_rdb_drops_comments_and_format_row(tmp_path):
    f = tmp_path / "sample.rdb"
    f.write_text(RDB_SAMPLE, encoding="utf-8")
    df = read_rdb(f)
    assert len(df) == 2
    assert list(df["site_no"]) == ["05200020", "05200170"]
    # all columns stay strings (site numbers keep leading zeros)
    assert df["site_no"].str.startswith("05").all()


def test_read_rdb_preserves_dtypes_as_str(tmp_path):
    f = tmp_path / "sample.rdb"
    f.write_text(RDB_SAMPLE, encoding="utf-8")
    df = read_rdb(f)
    assert pd.api.types.is_string_dtype(df["huc_cd"])
    pd.testing.assert_series_equal(df["huc_cd"], pd.Series(["07010101", "07010101"], name="huc_cd"))
