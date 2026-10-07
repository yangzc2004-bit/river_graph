# Validation

- All 295 selected station-corridor storage partitions replayed from the saved
  directed routes and the NHDPlus VAA. They represent 293 distinct receiving
  COMIDs. Cropped segment lengths reproduce the original fixed selections.
- All nine analysis tables replayed, including 5,000-draw summaries and the
  original 22 paired river comparisons. CSV verification compares row order
  and values; the intentionally omitted pandas in-memory index is not data.
- Whole-network path CV reproduces the original path descriptor. Storage
  allocations remain between zero and the available weighted path length.
- Both figure receipts checked against actual inputs and PNG/PDF outputs.
  All six English/Chinese figures were visually inspected. An overlapping
  title/subtitle in the first English draft was corrected before delivery.
- `pytest`: 1,200 passed, two skipped, eight pre-existing warnings.
- `ruff check .`: passed.
- Historical `audit_artifacts.py --verify`: exit 0, with 83 parquet-only
  verifications, one zero-coverage row and the one known excluded G0 conflict.
  All 85 legacy prediction files retain their recorded `no_sidecar` status.

No training was started and no previous experiment, mask or endpoint was
modified. The scope is exploratory river geometry and existing source-role
DOC evidence within ST357.
