# Auxiliary chemistry for cross-station DOC reconstruction

## Scientific question

Does routinely measured pH and specific conductance provide useful local
chemical information for reconstructing DOC at held-out stations? Small
distribution-head gains did not resolve high-DOC underprediction. This study
adds measured information to the existing model, preserving its spatial,
ecological and recurrent representation.

DOC remains the only training target. This is reconstruction with known
auxiliary chemistry, rather than transfer to a wholly unmonitored station.
Same-calendar-month measurements are available inputs, not a month-start
forecast. Their monthly means need not come from the same sampling instant.

## Data

Use the existing aligned ST357 DOC, pH and specific-conductance tensors.
Preserve their current raw QC, units, masks and monthly aggregation.
pH inputs are pH/14; conductance inputs are log1p(uS/cm). Missing values are
zero with explicit visibility flags. No auxiliary imputation is added.
The four auxiliary columns are pH value, EC value, pH visible, EC visible.

These exogenous measurements are allowed at source and held-DOC stations.
Current/future hidden DOC remains unavailable to feature construction.
Auxiliary input at month t reads only that month, with no future-month lookup.
Source-only feature normalization uses observed source DOC loss rows.

The availability audit is based on data masks, not DOC prediction outcomes.
Either auxiliary is observed for about96–99% of the fixed observed-DOC
holdout queries, but only19.74% of genuinely DOC-missing grid cells. Report
both populations. A high holdout benefit cannot be extrapolated to every
missing cell.

## Fixed model comparison

Use partitions142/143/144 and seeds42/43/44. Retain the selected daily-head
expert, its source forest OOF base, source-trained neural correction,
ecological residual memory and legacy GRU support representation.

Three neural correction arms share the same682-dimensional feature layout:
the original550 head features, four auxiliary slots and hidden64 × two
auxiliary-value interactions. A zero-initialized float64 linear native-unit
residual head learns only an additional correction. Modes are:

1. `neural_no_aux`: all auxiliary slots/interactions zero.
2. `neural_masks`: visibility slots retained; values/interactions zero.
3. `neural_chemistry`: measured values and visibility retained.

All use the same real auxiliary-availability gate. If both auxiliary
measurements are absent, the correction is zero. After support/ecological
adaptation, copy the corresponding original point-model prediction and
components at those query cells. The final neural product thus preserves
the retained model exactly wherever the added measurements are absent.
This shared gate also applies to the no-aux control; it isolates values
from the availability-conditioned footprint of the experiment.

Source native MAE uses the inherited Q90 weight2, one full-source weighted
denominator, Adam LR.001, batch512, gradient norm1, at most120 epochs and
patience10. Source-validation K0 unweighted native MAE selects checkpoint
and correction scale{0,.25,.5,1}, including epoch0. Ties favor smaller
scale then earlier checkpoint. Source std below1e-6 is replaced by1.
Only the forest baseline is OOF; the frozen neural expert is source-trained.

Three matched ExtraTrees arms receive the same current-month auxiliary modes.
Their151 columns are the existing147-column current-daily forest features
plus the same four slots. Clone the existing tree-current parameters;
only n_jobs=2 may change. Each arm fits the same source labels without
hyperparameter search. Reuse the147-column parent tree as a reference.
At auxiliary-absent cells, each tree probe copies that parent tree prediction;
after direct support adaptation, copy its corresponding parent tree query
prediction. These are stand-alone controls, not new neural residual bases.

Total:27 neural head fits and27 forest fits. Also retain the original point,
context forest and current-daily tree predictions. All direct arms use the
legacy GRU support representation; neural arms and the point reference also
use the existing source-validation-selected ecological integration.
K=0/1/3/5 uses unchanged nested support and fixed query identities.

New support/mixing choices score auxiliary-active validation queries only.
Inactive final predictions are copied from a fixed parent and contribute the
same constant error for every candidate, so this gives the same ranking as
full-query final-pipeline MAE. If a validation station has no active query,
exclude that station's support from these calibration episodes. Reference
models retain their original full-validation selection. Report full-query
and active-query validation errors separately.

## Twelve specified primary comparisons

At K0 and K5:

- Neural chemistry versus neural no-aux, direct and integrated (four).
- Neural chemistry versus neural masks, integrated (two).
- Neural chemistry versus retained point, integrated (two).
- Tree chemistry versus tree no-aux, direct (two).
- Integrated neural chemistry versus direct tree chemistry (two).

Report all13 model curves. Report native/log MAE, RMSE, R2, Q90/ordinary
error and bias, recall/false-high rate, and both/ph-only/EC-only/neither
availability strata. Use5000 paired whole-station bootstrap draws, seed
means within partition and equal partition weights; repeated station
identities have joint bootstrap multiplicity across partitions.

No target outcome selects a mode, correction scale, support route or K.
If values improve over masks in both model families, chemical values add
information. If trees capture the gain while the neural branch does not,
the next model iteration should integrate that chemical representation.
If the neural correction helps only chemistry-observed cells, describe
its supported reconstruction setting and retain the original fallback.

## Products

Save auxiliary dataset/provenance identities, availability audit, actual
source/validation feature and native-base identities, all fitted heads and
trees, source-validation traces/choices, full-grid components, final
fixed-query products and exact fallback checks. Archive execution source.
The preceding density-head study remains unchanged; existing station
partitions continue to be development data.
