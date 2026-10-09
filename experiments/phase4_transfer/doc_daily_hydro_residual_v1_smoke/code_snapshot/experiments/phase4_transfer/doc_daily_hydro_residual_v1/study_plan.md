# Within-month hydrologic information for DOC spatial transfer

Monthly mean discharge cannot distinguish a stable month from one with short
flow pulses. The existing native residual already uses monthly anomalies and
lags, but source-validation high-DOC errors remain large when flow is observed.
This experiment tests whether daily discharge summaries provide useful state
information beyond the current monthly input and its availability.

## Model and three matched information arms

Keep the existing final self/ecological encoder, observation-aware GRU and
native concentration-conditioned residual head. Append eight channels to the
existing 30 head features; all arms use the same 38-dimensional head and the
same hidden-feature interactions. The three daily value channels also interact
with the GRU hidden state, using indices (0,2,4,28,30,31,32).

1. **monthly:** all eight new channels zero; a fresh matched control.
2. **availability:** daily value channels zero, coverage and validity retained.
3. **daily:** all daily values, coverage and validity retained.

Daily information enters the existing residual readout. The GRU's original
historical input and 12-month memory are retained. This is an input/readout
extension, with no new graph layers, classifier, loss penalty or prediction
gate. Empty-edge/self-path operation remains as in the preceding spatial model.

## Daily feature construction

Use the first NWIS parameter00060/statistic00003 series, as in the existing
monthly extraction, with explicit alternative-series inventory. Apply the
existing signed discharge QC (finite and absolute value at most3e6cfs).
Collapse identical station/date duplicates; exclude conflicting duplicates.
Do not interpolate gaps or fill missing days.

For each calendar month, retain:

- Bounded distribution width: (Q90-Q10)/(mean absolute flow + Q90-Q10).
- Bounded consecutive-day flashiness: sum absolute daily changes divided by
  sum paired absolute flow magnitudes; only genuine adjacent days within month.
- Rising-day fraction: positive changes divided by valid adjacent-day pairs.
- Valid-day/calendar-day fraction and valid-pair/(calendar-days-1) fraction.
- Three separate descriptor validity flags.

Quantiles use linear interpolation. Width requires at least80% of calendar
days; pair descriptors require at least80% of possible within-month pairs,
rounded upward. Adequately observed all-zero flow has valid zero descriptors.
Missing or insufficient data give zero values with validity zero, while coverage
fractions remain visible. These ratios require no fitted scale and preserve
signed flows and positive-unit invariance. Zero the whole new block outside
the frozen monthly discharge mask, so richer summaries do not expand the
hydrologic observation footprint.

Daily features depend only on station/date/discharge and hydro availability,
never DOC truth. Use current and earlier months only. Full current-month flow
is information for retrospective month-end reconstruction, not a month-start
forecast or a prediction at an individual within-month DOC sampling date.

## Training and comparisons

Three existing station partitions142–144 and seeds42–44 give **27 fits**.
Start from the original expert; retain tail weight2, learning rates, source
OOF context, source-fold-hidden target inputs, source-validation checkpoint and
scale selection, fixed ecological residual memory and GRU support basis.
Maximum epochs120 with patience5. Include epoch0. All K=0/1/3/5 products use
the existing fixed queries and retrospective reserved supports.

Primary information comparisons are daily versus availability, availability
versus monthly, and daily versus monthly. Report direct and ecological-integrated
products separately, with all K curves, native/log MAE, RMSE, Q90 error and
recall, ordinary error and false-high rate. Keep all three arms and the prior
encoder/ecological references. The fresh zero-extra model is the matched
optimization control; an enlarged zero-padded head need not replay the old
optimizer trajectory bitwise.

Use the current equal-partition estimator and5,000 paired whole-station
bootstrap draws, jointly resampling repeated station IDs. Source validation
alone selects checkpoints and adaptation/mixing weights. These previously
examined partitions remain model-development data. No older endpoints or
experiment products are changed.

## Scientific reading

Value over availability is evidence for within-month hydrologic information;
availability over monthly indicates a monitoring-support effect. Improvements
must be read with high-DOC sensitivity and ordinary error, and with the fraction
of stations/months actually supplied by daily records. A lack of improvement
will inform the next state-information experiment rather than trigger a daily
feature threshold search on target results.
