# Reproduction and visual checks

- **Project tests:** 1,259 passed, 2 skipped, 8 pre-existing warning instances.
  Twelve new tests cover isotope arithmetic, whole-second clock matching,
  bounded integration/quantiles, missing and long-gap exclusion, common support
  for paired areas, concentration slope versus mass recovery, cancellation of
  the common calibration in paired ratios, genuine replicate handling and
  preservation of unavailable original laboratory values. Download replay also
  restores missing gitignored inputs from the saved URLs, retains the manifest,
  and refuses to overwrite conflicting cached content or resolve a new author ref.
- **Ruff:** all checks passed.
- **New replay:** all 433 original rows, six series, three addition pairs,
  54 sensitivity rows and analysis products reproduce. Independent scalar
  trapezoids differ by at most 1.78e-15. Original missing DOC remains missing.
  All eight English/Chinese PNG/PDF source bindings match.
- **Historical artifact audit:** exit 0, preserving 83 parquet-only verifications,
  the reported zero-coverage case, the excluded known G0 conflict and 85 old
  no-sidecar statuses. Historical predictions and frozen tables were not edited.
- **Actual visual review:** all four exported PNGs were opened and inspected.
  Legends, sampling clocks, raw laboratory dots, missing-assay marks and paired
  contrast labels are readable in both languages. No line connects across a
  missing/conflicting salt clock or a gap over 30 minutes. DOC panels show their
  own labelled concentration axes; salt curves use the same 0–300-minute window.

Raw isotope-labelled carbon and total DOC remain distinct. Author-processed
points are separate from original lab measurements. Sensitivity ranges are not
confidence intervals; three releases in one stream are not replicated network
forms. Nominal catalogue distances are not used to infer velocity or processing
per metre. No pulse tail was extrapolated and no total mass-recovery claim is made.

Only source acquisition and an observation reanalysis were performed. No model
training, endpoint replacement, morphology relabeling or historical result
overwriting occurred.
