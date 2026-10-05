# Donor-to-local contrast: source-validation results

Nine packages completed. K0 MAE is **1.828568 mg/L**, compared with **1.829579**
for the current complete model: **0.055%** reduction, paired station-bootstrap
interval **−0.265–0.300%**. Two of three partition means improve, but the result
does not establish a useful performance increase.

Uniform donor weights yield **1.828413 mg/L**. Removing donor residual values
while retaining the local subtraction yields **1.828459 mg/L**. The fitted
attention does not outperform either mechanism control. The tiny correction
can be obtained through re-scaling the existing local correction; it is not
evidence of useful source selection.

Keep version 2 as a completed development experiment. Its zero-initialized
contrast integration works, all nine checkpoints replay bitwise, and hidden
outer target roles remain unevaluated. The current complete model remains the
reference. Do not move this weak candidate into geographical confirmation.

Next, improve the **source response representation**, keeping the contrast
operator fixed: fit robust season/hydro-dependent donor profiles from the same
nested OOF forest residuals, with moderate rather than dominant shrinkage.
Persist the source-only nested residual arrays so future representations can
reuse the data without repeated forest fitting. A donor-profile experiment is
useful only if it improves reconstruction and its source-value controls support
an information-transfer interpretation.
