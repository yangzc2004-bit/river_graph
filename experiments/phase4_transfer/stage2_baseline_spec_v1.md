# Stage 2A baseline diagnostic specification

Document path retained from the first Stage 2 draft. Status: **corrected
descriptive specification for v2, after exploratory query results were seen,
2026-09-24**. The original assertion that the whole Stage 2 gate had passed
is withdrawn. This specification governs a bounded non-learning diagnostic;
it does not replace the frozen transfer endpoints.

## Scientific question and label regime

How much do K local observations improve same-month reconstruction relative
to a same-analyte climatology from stations outside the entire target HUC6?

This is a **same-analyte support diagnostic with privileged source labels**.
Climatology uses observations of the target analyte outside the held-out
HUC6. It is not a leave-one-analyte experiment. The complete source record is
available to this descriptive same-month reference; consequently the result
does not establish future-month forecasting. No shared representation,
ecological-information gain, or cross-analyte transfer is tested here.

## Fixed task construction

- Targets: DOC, pH, and specific conductance.
- Canonical HUC6 values: `{101302, 101900, 102701, 103001, 051002}`;
  legacy `510020` aliases `051002` only.
- Read and hash the authoritative `hide_rows` and `task_component_rows` from
  `experiments/kshot_protocol_v2/regions.json`; verify their metadata and graph
  membership against the current ST357 bundle.
- Exclude the **whole target HUC6** (`hide_rows`) from every source-label
  statistic. Use only `task_component_rows` to construct target tasks.
- For task seeds `{42, 43, 44}`, a month is eligible with at least seven
  observed target-analyte stations in the component. Five stations are
  reserved by the fixed permutation; remaining stations form the fixed query.
  Support sets are nested first-K subsets for K in `{0, 1, 3, 5}`. The query
  remains identical across K within a task realization.
- Freeze exact support/query membership in a task manifest. Neither query
  values nor a metric may change eligible tasks or select a method.

## Four descriptive reference methods

Report month-of-year climatology, local support mean, mean-bias correction,
and the fixed analytic support blend. At K=0, all four equal the privileged
same-analyte climatology. This is an explicit diagnostic definition, not a
solution to the target-unseen K=0 output-scale problem.

Report all four methods without selecting a winner from query error. Analytic
blend remains a named diagnostic because it was used in the exposed draft;
its designation does not become a new confirmatory primary endpoint. Record
the full K curve. These four methods do not complete the broader planned
baseline set or its ecological/temporal/local/topological decomposition.

## Error and tail definitions

For each method and K report native-unit MAE, paired Delta-MAE against K=0,
relative reduction when its denominator is nonzero, and the declared
descriptive transformed error. Report transformations and any clipping
explicitly; log1p error for pH is a numerical diagnostic, not a new physical
interpretation.

Q90 is calculated from observed target-analyte values in the source set
outside the whole target HUC6. Freeze that threshold before reading query
values. Query values determine tail membership only during evaluation. Report
the threshold and unique tail station-month count; repeated task seeds do not
increase that count. Mark tail estimates with fewer than 20 unique cells as
unstable. Do not use query-derived quantiles as this endpoint.

## Frozen descriptive v2 estimator and resampling

Use the same estimator in tables, paired differences, and bootstrap:

1. Query-cell mean within month and task seed;
2. Equal task-seed mean within calendar month;
3. Equal calendar-month mean within basin;
4. Equal basin mean within analyte.

Bootstrap the calendar month as a shared cluster across basins, keeping all
methods and K values paired. For each analyte, draw from the union of its
eligible months and apply shared sampled-month multiplicities to every basin
containing that month. Recompute basin means and equal basin pooling. Do not
resample basins or treat task-seed realizations as independent clusters.
Document any empty-stratum redraw policy and its count. Use 2,000 valid draws
and record the bootstrap RNG seed. Report unique months, task realizations,
and unique query cells separately. Do not pool raw errors across analytes with
different units.

All intervals are descriptive after prior query exposure. They cannot be
advertised as a fresh confirmatory test.

## Provenance and acceptance

Write new products under `stage2a_same_analyte_v2/`. Retain the flawed
`stage2a_same_analyte_v1/` bytes and mark their results invalidated. The first
`stage2_baselines_v1/` products were accidentally deleted; only a withdrawal
record and transcript evidence remain, as documented in
`stage2a_baseline_correction.md`.

Bind exact tasks, dataset/node/edge hashes, source visibility, configuration,
specification, runtime dependencies, prediction content, and evaluator
identity. The evaluator must verify those bindings before producing a report.
Contract tests must establish that perturbing non-support labels anywhere in
the hidden HUC6 leaves predictions and source-derived thresholds unchanged.

Acceptance here means that this limited diagnostic is reproducible and obeys
its label and statistical contracts. **The overall Stage 2 gate remains
not evaluated** until all required baselines, controls, missingness regimes,
and information comparisons are complete. The unresolved target-unseen K=0
scale and source-only selection protocol block transfer launch. Neither this
diagnostic nor any apparent pass in its earlier versions authorizes Stage 3.
