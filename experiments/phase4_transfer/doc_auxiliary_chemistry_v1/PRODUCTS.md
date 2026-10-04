# Auxiliary-chemistry DOC products

## Model and information

The existing DOC expert stays fixed. Three additional native-unit linear
residual heads compare no auxiliary inputs, availability flags and measured
monthly pH/conductance. Three matched forest probes compare the same modes.
DOC remains the only output target.

All new modes share the actual pH-or-conductance availability footprint.
The no-aux mode zeros the four input slots but retains that shared gate;
it controls for conditional correction capacity on chemistry-observed cells.
The masks mode retains the two visibility flags. The chemistry mode retains
both flags and pH/14 and log1p(conductance in uS/cm). Calendar-month means
can come from different sample times.

Only 19.74% of genuinely DOC-missing grid cells have either auxiliary
measurement. Availability is much higher in artificially hidden observed-DOC
queries. The experiment reports these populations separately.

## Package contents

Each of nine `runs/split{142,143,144}_seed{42,43,44}/` packages contains:

- `neural_{no_aux,masks,chemistry}.pt` and `.json`: 683-parameter native
  correction heads, source-only normalizers, checkpoint/scale choices,
  source active counts and training traces.
- `tree_{no_aux,masks,chemistry}.joblib` and `.json`: matched fixed-parameter
  ExtraTrees probes with 151 columns. These do not replace the original
  forest or supply a new OOF neural training base.
- `source_training.npz`: source station/cell identities and native bases,
  validation query identities/bases, and explicit auxiliary-active flags.
- `input_definition.json`: original550-feature and augmented682-feature
  identities for source/validation, with all three modes.
- `adapters.json`, `mixers.json`: source-validation-selected calibration.
  New modes score active validation queries; references retain their original
  all-query calibration. Fixed inactive fallback error cannot change ranking.
- `source_validation.csv`: final full-query and active-query MAE for each
  of thirteen models at K0/1/3/5.
- `full_grid.parquet`: all 233,478 station-months, original and new native
  components, integrated K0 products and auxiliary/DOC observation flags.
  No hidden DOC truth is attached.
- `predictions.parquet`: thirteen models × four K values on the unchanged
  held-station queries. `candidate_y_pred` records the pre-fallback candidate;
  `y_pred` is the actual final prediction. `aux_fallback` marks copied-parent
  rows. Truth is attached only after fitting and source-validation selection.
- `reference_checks.json`: exact original point/context/current-daily-tree
  references across sixteen panels per package.
- configuration, sidecars, completion manifest and timing.

## Final fallback

When neither auxiliary is observed, each new neural product copies the original
point model after its corresponding direct or ecological/support adaptation.
Each new tree product copies the original current-daily-tree direct product.
The fallback includes `y_pred`, `base_pred`, `adaptation_delta` and
`regional_gamma`. Changing calibration choices for chemistry-observed cells
therefore cannot silently alter the final prediction at chemistry-absent cells.

The original forest's parallel prediction sum may vary in its last bit.
Its replay is checked numerically; the saved parent prediction is canonical
for exact reference/fallback comparisons. New forest state replay uses strict
numerical tolerance, while neural/fallback products are checked exactly.

## Reproduction

```bash
OMP_NUM_THREADS=2 uv run python scripts/run_ladder.py --experiment doc-auxiliary-chemistry-v1
uv run python scripts/verify_doc_auxiliary_chemistry_v1.py
uv run python scripts/analyze_doc_auxiliary_chemistry_v1.py
```

Execution code is archived under `code_snapshot/`. `availability_audit.json`
and `.md` record raw unit/QC reconciliation and dataset alignment. The
original engineering smoke stopped on a 1.42e-14 parallel-forest roundoff
comparison; the corrected `_smoke2` uses canonical saved parent values and
completed. Neither smoke enters the nine-package scientific analysis.

These are reused development partitions. Target support and the retained
calendar-anchor basis describe retrospective reconstruction. No new uncertainty
calibration, new output analyte or measured physical transport coefficient is
introduced.
