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

## T7 matched snapshot completion

T7 adds H2X snapshot seeds 45 and 46 for all three analytes and four masks
(24 runs). These runs complete a five-seed matched baseline for the existing
five-seed H2X-T confirmation products.

## T8 matched-budget synthesis

T8 combines the audited T2/T3--T7 products without new training. The main
five-seed H2X-T versus snapshot table uses paired station- and month-cluster
bootstrap intervals (2,000 replicates, seed 20260927), with no resampling of
training seeds. The 1/3/6/12-month window curve and history ablations remain
descriptive three-seed diagnostics. The complete analysis, input ledger,
summary tables, confidence intervals, and figures are under `t8_synthesis/`.

The ledger records 144 historical sidecars whose declared temporal fields do
not fully describe the runtime arm (the predictions are retained unchanged).
This is reported as a provenance metadata discrepancy; it does not alter the
already frozen predictions or their original audit records.

## T9 final products and figures

T9 builds the five-seed H2X-T median product for all three analytes and four
masks (2,801,736 full-grid rows; 32,400 test rows), plus seed spread SD/IQR,
station-level test error, and a manifest of all source prediction hashes.
Publication figures include the analyte-by-missingness reduction heatmap,
descriptive station traces, and E2a station error maps. Seed spread is labeled
as ensemble dispersion and is not presented as a calibrated prediction
interval.

## T10 manuscript package

The first English results chapter, figure captions, and claim-evidence table
are frozen as `docs/paper/h2x_t_results_v1.md`,
`docs/paper/h2x_t_figure_captions_v1.md`, and
`docs/paper/h2x_t_claim_evidence_v1.csv`. They use the T8/T9 outputs directly
and preserve the conditional pH result, the window/ablation interpretation,
and the previously closed blind-spot claim.

The full draft is `docs/paper/h2x_t_manuscript_draft_v1.md`. It contains the
abstract, introduction, methods, results, discussion, and conclusion based on
the frozen T8/T9 package.

## T11 LaTeX manuscript draft

The first compiled manuscript is `docs/paper/latex/ecohydrograph_draft.tex`,
with the rendered PDF at `docs/paper/latex/ecohydrograph_draft.pdf`. It uses
the T8/T9 tables and figures directly, adds a vector architecture diagram,
and uses the paper name EcoHydroGraph; H2X-T remains an internal experiment
identifier. The source was compiled twice with a user-local TeX Live 2026
installation at `/Users/yzccc/local_texlive/texlive/2026`. The PDF is a seven-page
working draft; no new training or endpoint changes were made for manuscript
generation.

The manuscript now uses descriptive missingness names throughout: random gaps,
unobserved-period extrapolation, observation-assisted extrapolation, and
unmonitored stations. Publication-only figures are rendered by
`scripts/render_ecohydrograph_figures.py`; internal mask identifiers remain in
the experiment directories for reproducibility.
