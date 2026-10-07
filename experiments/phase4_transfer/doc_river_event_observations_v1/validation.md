# Validation

- Nine laboratory/geometry tables and four accompanying Parquets replayed
  from preserved public raw files and the two original dense connection-months.
  Four optical-audit tables, one paired-hours Parquet and its summary replayed
  separately. Both sets retain their own source and output records.
- Actual source coordinates and directed reach geometry determine station
  connections; paths are cropped at the receiving sampling position. Metadata
  polygons remain contextual. No missing connection is added by proximity.
- Tests cover the fixed UTC+1 clock, censored values, duplicate conflicts,
  flow-only window selection, sampling on both sides of a flow peak, directed
  connectivity, exact receiving-station cropping and one-use campaign matching
  with the full three-timestamp span bounded by 12 hours.
- Optical overlapping-event duplicates are recorded, with two conflicting
  site-times excluded. Published source-clock timestamps remain unchanged.
  Record counts and elapsed hours are separate: nine blocks have at least 24
  records, while seven span at least 24 hours.
- All four English/Chinese PNG figures were visually inspected. Sampling
  points are displayed without interpolated laboratory DOC curves. Actual
  mapped paths and daily-flow context are retained. A first draft's legend
  was moved above the sample panels, and its collection label was corrected
  to include the separate Degerö site in the 16-site audit.
- `pytest`: 1,207 passed, two skipped, eight pre-existing warnings.
- `ruff check .`: passed.
- Historical `audit_artifacts.py --verify`: exit 0, with 83 parquet-only
  verifications, one zero-coverage row and the known excluded G0 conflict.
  All 85 legacy predictions keep their recorded `no_sidecar` status.

This is an exploratory real-observation mechanism study. Eleven paired spring
windows describe three receiving sites within one research catchment; eight
windows share C7. The separate optical pair is not independent laboratory DOC
or an external validation of the prediction model. Previous experiments and
classifications are unchanged.
