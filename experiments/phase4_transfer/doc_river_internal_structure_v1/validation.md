# Validation

Date: 2026-10-07. Environment: repository uv-managed Python 3.12.14.

- Full `pytest`: **1,129 passed, 2 skipped, 8 warnings** in 44.65 seconds.
  The four added tests cover structural assignments, DOC-independent thresholds
  and examples, area/path scale invariance, and fixed-mean conserved routing.
  Warnings are the existing Torch, exponential overflow, tiny-sample edge
  scaling and Shapely warnings; no test failed.
- `ruff check .`: all checks passed.
- Study verifier: replayed 322 structural assignments, 966 routing scenarios,
  14 summary/diagnostic tables and the representative pulse product; checked
  actual cached map geometry, figure manifests and the 5,000-draw system
  analyses. Single-system reference-profile intervals remain unavailable.
- English and Chinese PNG figures were opened and inspected. Repairs addressed
  crowded English heat-map labels, log-axis minor labels, annotation placement
  and the Chinese description of shared monitoring systems.
- The initial verifier read station `05357245` as numeric `5357245` in the
  influence table. Explicit string types for receiver and omitted-system IDs
  fixed CSV replay; the analysis values and receiver identity were unchanged.
- Historical `audit_artifacts.py --verify`: exit 0. It reports the existing
  83 parquet-only checks, one zero-coverage artifact and the documented corrupt
  G0 artifact. Historical predictions have no sidecars; this audit is not
  represented as complete modern provenance validation.

No DOC prediction model was trained or selected. Earlier outline, prediction,
protocol and mechanism artifacts remain intact. Unrelated working-tree changes
and historical untracked files are excluded from this study's commit.
