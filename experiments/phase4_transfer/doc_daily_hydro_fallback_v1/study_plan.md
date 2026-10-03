# Availability-preserving daily hydrology for DOC spatial reconstruction

The completed daily-hydrology experiment improved support-assisted reconstruction,
but its source-validation predictions deteriorated where all numerical daily
descriptors were unavailable. Test a fixed information-availability rule using
the saved monthly and daily encoder/GRU experts. This is a development iteration
after inspecting the preceding results.

## Model change

Route native predictions before ecological integration and station adaptation.
If any of the three existing numerical validity flags is one, use the daily
expert; otherwise use the monthly expert. The calendar/QC thresholds and input
footprint remain those of the saved daily feature pack. Valid constant or zero
discharge uses the daily expert. No new threshold, learned gate or neural fit
is introduced. Monthly and daily checkpoints remain fixed.

Refit the existing direct support adapters and ecological integration choices
separately for monthly, daily and hybrid bases using source-validation labels
only. Frozen context forests, ecological profiles and GRU support bases are
identical. Native hybrid predictions without daily information must equal the
monthly predictions exactly. Final adapted or ecologically mixed predictions
can differ because the downstream parameters are reselected for each base.

## Comparison

Use station partitions142/143/144, seeds42/43/44, K0/1/3/5, constant and frozen
GRU-shaped support adapters and the unchanged fixed query cells. Preserve the
six preceding encoder/ecological reference products. Monthly/daily direct and
integrated controls must reproduce their parent results.

Fit and inspect source-validation choices before comparative target analysis.
Report hybrid versus daily and monthly, direct versus integrated, and the
earlier ecological-affine reference. Use the existing equal-partition/seed
MAE estimator and paired whole-station bootstrap with5000 replicates. Report
raw/log-space errors, Q90/ordinary errors, recall and false-high rates and the
all-valid, partially-valid and no-valid daily descriptor groups. Positive K
uses the existing retrospective support protocol; this is monthly reconstruction.

## Products

Save nine independently bound packages, each containing full-grid native and
integrated K0 predictions, query predictions for all K, the reloadable fixed
router, source-validation-selected adapters/mixers and source-validation
diagnostics. Link their saved parent checkpoints and verified products without
duplicating or overwriting them. Reproduce routing and downstream inference
independently before selecting the next model iteration.
