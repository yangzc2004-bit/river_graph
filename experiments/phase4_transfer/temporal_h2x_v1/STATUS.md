# H2X-T temporal extension status

## Engineering gates

- **T0: passed.** Spatial encoder/head split, causal rolling windows, analyte
  transforms, visibility isolation, and V3 provenance contracts are covered
  by `tests/test_temporal_h2x.py` and the provenance/cache tests.
- **T1: passed.** The DOC/E2a smoke path produced a finite nonconstant full
  station-month product with a recomputable V3 sidecar.  An E3 one-epoch
  smoke was also used to exercise the held-out-site metadata path.
- The initial DOC-frozen mask was audited across targets.  pH and specific
  conductance have additional missing labels, so the runner now materializes
  deterministic target-specific intersections under `target_masks/` before
  training; this is part of the T2 input contract.

## T2 pilot

The complete 72-run pilot uses the frozen 3-analyte × 4-mask × 3-seed ×
{H2X, H2X-T} design. Both arms use the explicit `matched_full_grid` training
protocol (one optimizer update after the full monthly sequence per epoch),
with a pilot budget of 10 epochs and patience 3.

`evaluate_temporal_h2x.py --stage pilot` audits all 72 products, including
dataset/mask/prediction hashes, station-month alignment, role visibility,
metric recomputation, and paired protocol identity. The audit is **72/72
pass**. The feasibility gate is also **pass**: all three analytes improve on
average in E2a/E2b and none shows an E1 mean degradation. Detailed numbers are
in `pilot_summary.csv`; the machine-readable gate is in `pilot_verdict.json`.

This is a pilot result, not authorization to start T3. Before formal training,
review the matched-budget comparison and decide whether to retain the current
10-epoch feasibility budget or increase the formal H2X-T budget.

## T3 formal confirmation

T3 was authorized after the pilot review and completed under `t3_formal/`.
The initial 50-epoch trial was stopped before any product was written because
the measured runtime was impractical. The frozen confirmation budget was set
to 10 epochs with patience 3, applied identically to all five seeds; this is a
runtime decision recorded in `t3_formal/run_plan.json`, not a change to the
model or endpoints.

All 60/60 products passed the V3 audit. Using the pre-existing matched
snapshot baseline for seeds 42–44, the mean E2a/E2b reductions were 46.1% for
DOC, 5.1% for pH, and 56.9% for specific conductance. The five-seed H2X-T MAE
standard deviations were small relative to the analyte scales. T3 therefore
passes the seed-confirmation gate; pH remains a smaller-effect analyte and is
reported separately rather than pooled into the DOC/EC effect size.

## T4 history ablations

T4 completed 72/72 runs for two diagnostic arms: `h2x_t_no_history`
(reverse the preceding history steps) and `h2x_t_hydro_only` (remove the
target-value and target-visibility channels from the sequence). All products
pass the V3 identity audit. Against matched H2X-T runs on seeds 42--44, the
MAE changes are below 0.6% in every analyte/mask family. The current
10-epoch result therefore does not isolate a measurable contribution from
chronological ordering or target-history channels.

## T5 current-only control

The next diagnostic keeps the GRU wrapper and optimizer protocol but sets
`lookback=1`, so the model receives only the current month. It is frozen in
`t5_current_only/run_plan.json` as 36 runs (three analytes, four masks, three
seeds). This separates a genuine multi-month signal from effects of the
temporal wrapper and matched training budget.

## T6 temporal window curve

T6 adds lookback-3 and lookback-6 arms for E2a and E2b (three analytes,
three seeds, 36 runs). The already completed current-only and full H2X-T
products provide the lookback-1 and lookback-12 endpoints. The purpose is to
identify whether the temporal benefit appears gradually or is concentrated in
the full 12-month window.
