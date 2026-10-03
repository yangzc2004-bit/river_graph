# Next DOC iteration: retain the monthly prediction when daily information is absent

Daily hydrologic descriptors improve source-validation and support-assisted
spatial reconstruction. The remaining issue is deterioration at stations/months
where their numerical values are unavailable. Source-validation analysis shows
the same asymmetry as the development-query analysis. Test this specific
inference change before adding more temporal parameters.

Reuse the saved monthly and daily encoder/GRU checkpoints. Define availability
using the existing three descriptor validity flags: if every flag is zero,
use the monthly native base; otherwise use the daily native base. Adequately
observed constant/zero flow has valid flags and uses the daily branch. Retain
the existing calendar/QC rules; do not choose a new threshold.

Compare the monthly base, daily base and availability-preserving base. Fit
their ecological mixing and K-shot support calibration separately using the
same source-validation episodes/grids. Frozen forests, ecological memory,
support representation, fixed queries and all K values remain unchanged.
No neural retraining or free prediction gate is needed for this comparison.

Evaluate overall, Q90 and ordinary errors, recall and false-high rates, with
the same paired station estimator. Check whether source-validation chooses
useful mixtures and whether the preserved branch removes missing-information
deterioration without losing the daily-information benefit. Report direct and
integrated effects separately. This is a new development iteration after the
current results, not an additional confirmatory test of them.

Save its products separately; preserve this completed experiment. If the
fallback is useful, integrate the two saved paths as an explicit model
interface. Daily-state integration into the GRU and an RF comparison supplied
with the same daily covariates follow as separate questions.
