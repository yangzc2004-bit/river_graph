# Current-source DOC release

The fixed release uses all five source fits, seeds 42--46. It combines the saved
station-hidden environmental forest, ecological encoder, twelve-month
observation-aware GRU and twenty-candidate current-source residual attention.
All source fitting, external replication and temporal compatibility runs are
complete. The existing fitting snapshots and earlier products are retained.

## Prediction interface

Use the managed environment and add the repository `scripts/` directory to the
Python import path for this research prototype:

```python
from portable_current_source_ensemble_v2 import PortableCurrentSourceEnsemble

model = PortableCurrentSourceEnsemble.load(
    "experiments/phase4_transfer/doc_current_source_portable_v2_r1/release.json"
)
prediction = model.predict(new_inputs)           # [new stations, months], mg/L
parts = model.predict_components(new_inputs)
intervals = model.predict_interval(new_inputs)
sources = model.source_candidates(new_inputs)

adapted = model.predict_with_support(
    new_inputs, k=3,
    support_cells=designated_cells,
    support_values=designated_DOC_values,
)
```

New inputs contain unique `site_no[N]`, consecutive `months[T]`,
`x[N,T,2]`, binary `x_mask[N,T,2]`, `static[N,2]`, `regime[N,13]` and
`daily_features[N,T,8]`. These use the existing temperature/discharge,
coordinate/ecology and bounded daily-hydrology feature definitions. Do not
manually refit or replace the saved source normalization. Optional named
`river_edges[E,2]` supply source context; missing external links give zero
directional context. Source similarity attention is a separate mechanism.

Receiving DOC, pH and conductivity are not input requirements and never feed K0.
New station names must exclude the saved source library. No ST357 node index
is required. Include earlier hydrology months when available to initialize the
causal window. Saved source observations align by calendar month; dates beyond
the source record receive the zero-source prior.

Components expose environmental prediction, local temporal correction,
dynamic source-observation correction, static source-memory correction, their
combined source correction, river correction and final DOC. The local/source
decomposition holds fitted weights fixed while removing the new dynamic source
readout inputs; it is not a causal or orthogonal process decomposition.
Static-memory fusion and neural river correction are zero in this release.

`source_candidates()` reports each seed's candidate station names and counts
of actually observed donor months. Candidate order is ecological ranking,
not learned attention weight. Per-cell mean prior mass and entropy describe
attention allocation; empty static-memory retrieval is expected when gamma=0.

## Explicit DOC support and intervals

Support indices are flattened `station_index*T + month_index`. Only the
designated DOC values enter the separate output adapter, with at most K values
per station. K is 1, 3 or 5. All five support candidates stay outside scored
queries for every K. K1 alpha=0, K3 alpha=0.25 and K5 alpha=0.5 were selected on
source-validation ensemble predictions. These are retrospective corrections;
support dates can follow a query. `predict_with_support()` returns a prediction
array; `predict_interval(..., k=3, support_cells=..., support_values=...)`
returns its center and matching empirical interval.

Nominal 90% intervals use source-validation log1p errors. The validation role
also selected model states, so this is empirical calibration. Always report
coverage and width together. The updated external replication has 83.267%
coverage and median width 2.737396 mg/L.

`save(new_manifest_path)` writes a small manifest referencing the already saved
five member fits, and `load()` restores it. Move the manifest and its member
directories together to retain relative paths. Forest/neural weights and source
libraries are local fitted caches. New source fitting continues through
`run_ladder.py`; the deployment facade does not fit on receiving DOC.

## Complete-grid command

```bash
uv run python scripts/portable_current_source_ensemble_v2.py \
  --release experiments/phase4_transfer/doc_current_source_portable_v2_r1/release.json \
  --inputs new_station_inputs.npz \
  --output new_station_doc.parquet
```

The input NPZ may use `daily` in place of `daily_features`. The command opens
only the permitted feature arrays and writes every station-month, its DOC
prediction/components, support and hydrological availability, empirical
interval bounds/width and mean attention diagnostics. Separate `.sources.json`
and `.meta.json` files preserve candidate identities and product provenance.
Choose a new output path for a new product.

The real 130-station × 520-month export is
`doc_current_source_external_v2/products/doc_new_station_release.parquet`.
All 67,600 rows agree with the evaluated fixed release within 1e-12 mg/L;
component closure, interval width, unique identities and 650 seed/station
candidate records verify. This interface replay supplies no new performance
estimate.
