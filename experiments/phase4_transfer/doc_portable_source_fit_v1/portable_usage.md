# Portable DOC prediction

The source fits and individual `exports/seed42`–`seed46` are complete. The
five-seed `ensemble/` bundle uses native-mg/L arithmetic averaging and the
pre-external source-validation ensemble support/interval policy. Large fitted
models are retained locally, separate from the small research records.

Run Python through the managed `uv` environment. Add the repository `scripts`
directory to Python's import path for this prototype:

```python
from portable_doc_ensemble_v1 import PortableDOCEnsemble

model = PortableDOCEnsemble.load(
    "experiments/phase4_transfer/doc_portable_source_fit_v1/ensemble"
)
prediction = model.predict(new_inputs)                  # [new stations, months]
components = model.predict_components(new_inputs)
adapted = model.predict_with_support(
    new_inputs, k=3, support_cells=designated_cells,
    support_values=designated_DOC_values,
)
```

`new_inputs` follows the existing dataset conventions and contains:

- `site_no[N]`: unique station names outside the saved source library;
- `months[T]`: consecutive calendar months;
- `x[N,T,2]`, `x_mask[N,T,2]`: existing temperature/discharge representations
  and binary visibility, not manually restandardized inputs;
- `static[N,2]`, `regime[N,13]`: the same coordinate/ecology feature semantics
  used by the training dataset;
- `daily_features[N,T,8]`: the fixed daily feature builder's bounded values,
  coverage and availability flags;
- optional named `river_edges[E,2]` for source context; absent links give zero
  directional context.

DOC, pH and conductance arrays are not input requirements. Their presence
does not affect K0 prediction. Saved source preprocessing and experience are
applied to any newN/T; do not fit scales on the new target cohort. Include
preceding hydrology months if available to initialize the12-month window.
Source DOC context is calendar aligned and zero outside saved source dates.

Components include environmental prediction, local temporal correction,
source-memory fusion correction, optional river correction, native and final
DOC, hydro availability and retrieval provenance. In this deployment memory
gamma=0 and river correction=0; they remain explicit component outputs rather
than unsubstantiated contributions. Member prediction SD is descriptive.
`pi_lower/pi_upper` are source-validation empirical intervals, jointly reported
with width; the external evaluation gives84.92% coverage for a nominal90%.

Support indices are flattened `station_index*T+month_index`. Only designated
support DOC enters adaptation. All five candidate support cells should remain
excluded from scored queries for everyK. K1 alpha0 is retained from source
selection. These are retrospective corrections; support can follow query dates.

The single-member `PortableDOCReconstructor` exposes `fit/predict/
predict_components/save/load`. New scientific fits use `run_ladder.py` so that
source roles and execution records remain reproducible. A different basin's
DOC must not calibrate the K0 model merely to improve its reported test score.

## Command-line grid export

```bash
uv run python scripts/predict_portable_doc_v1.py \
  --inputs new_station_inputs.npz \
  --output new_station_doc.parquet
```

The NPZ uses the input arrays above (`daily_features`, or the prepared `daily`
key). The exporter reads no water-quality arrays at K0. It writes every
station-month with prediction, environmental/temporal/source/river components,
empirical interval bounds and width, descriptive member SD, hydro availability
and model identity, plus a provenance sidecar and donor-source diagnostics.
An existing output is preserved; choose a new path for a new product.

For explicit support correction add `--k 3 --support designated_support.npz`.
That support file contains `cells` (integer flattened station-month indices)
and `values` (DOC in mg/L), with at mostK values per station. Support rows are
marked separately and must be excluded from scored queries. Their reported
point prediction remains the model prediction rather than replacing it with
the supplied measurement. All other station-months receive the fixed
source-selected correction. Retrospective supports may follow a query date.

A real three-station/24-month command-line export reproduced the corresponding
independent-basin grid, with sidecar and unique row identity verified. This
checks the deployment interface; it supplies no new performance estimate.
