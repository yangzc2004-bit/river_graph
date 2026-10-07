# Validation

- All five analysis tables, both campaign Parquets and the result summary
  replayed from the unchanged preceding experiment and preserved public
  daily-discharge files.
- All 11 original windows and 74 matched campaigns are accounted for. There
  are 57 complete-flow campaigns, with 53 in eight windows meeting the
  five-campaign requirement. C7 retains its original 47 campaigns in seven
  eligible years; C9 retains six of seven, with the subset change reported.
- Flow matching uses each sample's own fixed UTC+1 date. Missing daily flow
  is neither interpolated nor filled. All 53 comparison campaigns match the
  same calendar day at the three locations.
- Six new tests verify source-calendar alignment, physical mixture/flux
  arithmetic, retained non-closure, missing-flow behavior, zero-flow handling,
  duplicate-gauge rejection, exact antiphase variance cancellation and the
  distinction between CV and absolute SD.
- Fixed-weight mixture variance replays as individual terms plus covariance.
  Daily-weighted mixture and receiver statistics use identical campaigns.
- English and Chinese PNG figures were viewed. Points, variability measures,
  partial flow shares and covariance components were compared with the saved
  calculations. Source/output records cover all PNG/PDF exports.
- `pytest`: 1,213 passed, two skipped, eight pre-existing warnings.
- `ruff check .`: passed.
- Historical `audit_artifacts.py --verify`: exit 0, retaining 83 parquet-only
  verifications, one zero-coverage result, the known excluded G0 conflict,
  and the recorded `no_sidecar` status of 85 legacy predictions.

No training, new morphology assignments or old endpoint changes. The repeated
observational result comes primarily from one confluence, within one research
catchment. This stage evaluates partial-input mixing, not DOC removal or an
externally validated prediction model.
