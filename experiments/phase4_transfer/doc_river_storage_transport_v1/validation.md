# Validation record

## Completed checks

- `uv run python scripts/verify_doc_river_storage_transport_v1.py`: full replay
  of 1,416 scenarios, 708 matched contrasts, receiver/class summaries, selected
  traces, source/product bindings and both figure languages.
- The complete 59-footprint grid is repeated at time steps 0.00125 and 0.000625.
  Maximum peak difference is 0.00003225 of unit input peak; peak-time difference
  is 0.000625; central 80% duration difference is 0.00001025.
- Integrated centroids agree with the fixed analytic mean within 5.2e-13;
  integrated anomaly area with one within 1.7e-14; pulse SD with its analytical
  value within 8.3e-12. Constant-concentration gain is one.
- Fourteen dedicated tests cover closed-form routing against independent causal
  quadrature, fixed mean/variance/area, small-storage numerical stability,
  branch symmetry, no-common-trunk behaviour and invalid inputs.
- Full `uv run pytest -q`: **1,158 passed, 2 skipped**, eight existing warnings
  (torch API/empty-edge statistics, extreme historical interval overflow and
  Shapely envelope diagnostics).
- `uv run ruff check .`: **all checks passed**.
- `uv run python scripts/audit_artifacts.py --verify`: **exit 0**. Historical
  results retain their existing limitations: 83 parquet-only verified items,
  one known G0 conflict, one zero-coverage result and 85 `no_sidecar` identities.
  This historical audit does not replace the new simulation replay above.

## Figure inspection

Both English and Chinese PNGs were actually opened and inspected. Shortened
English map titles to avoid the adjacent curve panel, wrapped the long vertical
axis label, and made pulse duration/storage settings explicit in the captions.
Gold underlays identify actual mapped waterbody reaches. Every map uses cached
NHDPlus geometry in EPSG:5070 cropped at the same station endpoints. Every curve
is labelled as a controlled response rather than a measured concentration event.

## Research scope

No DOC values select a storage fraction, pulse duration or display example. No
new model is trained, no physical travel-time scale is fitted and no previous
experiment tree is modified. Parameter ranges are scenario sensitivities,
not statistical confidence intervals. Matched corridor geometry and the three
whole-network outline labels have distinct roles in the interpretation.
