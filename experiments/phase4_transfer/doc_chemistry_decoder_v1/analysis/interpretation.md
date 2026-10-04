# Interpretation: nonlinear chemistry decoding for DOC reconstruction

## Result and model status

The nonlinear chemistry decoder establishes a useful K0 improvement on the current station-development partitions. Its ecological-integrated prediction reduces MAE from1.803538 to1.764947mg/L, a2.140% reduction [relative95% interval0.883%,3.399%]. It improves over matched no-auxiliary and mask-only decoder controls, and over the preceding linear chemistry branch. Both high-DOC and ordinary-value error improve against the retained point model.

Retain this decoder as the next chemistry-informed model candidate and as a demonstrated advance for stations with no supplied DOC support. Do not describe it as a complete replacement across K: at K5 its aggregate MAE is slightly worse than retained point, the interval includes zero, and the fixed chemistry tree remains better. No K-specific winner route is introduced.

All nine packages passed independent replay before interpretation:27 trained heads,81 direct and36 integrated validation refits,540 query panels and324 copied controls. The source554 features, SiLU chemistry basis, hidden interactions, native predictions and final fallback all replay exactly. The analysis reports fifteen curves and sixteen fixed contrasts using5,000 joint whole-station bootstrap draws.

## What changed

Only the additional decoder is trained. The existing DOC encoder, GRU, base point head, forests, ecological profile and old GRU support representation are frozen. The new decoder maps four auxiliary value/mask inputs to eight SiLU coordinates, combines them with the original550 features and hidden-state interactions, and produces a zero-initialized native residual. Source-only standardization is applied to554 inputs. It has1,111 parameters, compared with683 in the preceding linear correction.

Three modes have the same new architecture: no auxiliary values or flags, flags only, and measured chemistry with flags. All share the actual pH-or-EC availability gate and identical absent-chemistry fallback. Matched no-aux/mask comparisons distinguish additional measured-value information within this architecture. The old-linear contrast changes both representation form and parameter count; it does not uniquely attribute the difference to nonlinearity alone.

Source fitting retains tail-weighted native MAE. Full source-validation native MAE selects checkpoint and residual scale, while active validation queries select support and ecological parameters with constant inactive fallback. Source forest predictions are OOF, but the frozen source-trained neural correction is not claimed to be OOF.

## Principal curves and contrasts

| Model | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Retained point, direct |1.806996|1.758826|1.631317|1.589387|
| Linear chemistry, direct |1.795167|1.749001|1.627932|1.585300|
| Nonlinear chemistry, direct |1.773213|1.741419|1.621943|1.579294|
| Retained point, integrated |1.803538|1.755368|1.620090|1.564984|
| Linear chemistry, integrated |1.792258|1.746092|1.619351|1.571309|
| Nonlinear chemistry, integrated |1.764947|1.738600|1.614240|1.568820|
| Fixed chemistry tree |1.802261|1.768522|1.565090|1.537905|

All fifteen curves, including controls, are preserved in `findings.md` and `k_curves.csv`. Negative deltas below favor the candidate.

| Integrated nonlinear chemistry versus | K | Delta MAE [95% interval] |
|---|---:|---|
| Retained point |0|−0.038591[−0.066865,−0.015185]|
| Matched no-aux decoder |0|−0.036933[−0.063709,−0.015899]|
| Matched mask-only decoder |0|−0.035339[−0.062586,−0.014613]|
| Old linear chemistry |0|−0.027311[−0.046387,−0.012595]|
| Fixed chemistry tree |0|−0.037315[−0.092347,+0.015480]|
| Retained point |5|+0.003836[−0.009955,+0.016600]|
| Matched no-aux decoder |5|+0.005592[−0.006987,+0.017343]|
| Matched mask-only decoder |5|+0.004669[−0.007270,+0.015566]|
| Old linear chemistry |5|−0.002490[−0.010717,+0.005152]|
| Fixed chemistry tree |5|+0.030915[+0.002488,+0.062056]|

The direct nonlinear-versus-linear K0 delta is−0.021954[−0.040814,−0.007256], showing that the native/support branch itself improves; the K0 benefit does not rely only on reselecting ecological mixing. Its K5 delta−0.006006[−0.018716,+0.004229] is uncertain. The copied tree comparisons exactly reproduce the preceding study, so they are reference evidence rather than a second independent replication.

## High-DOC and ordinary-value tradeoff

At K0, integrated nonlinear chemistry versus retained point yields:

- Q90 MAE7.019544 versus7.179187; delta−0.159643[−0.316980,−0.018144].
- Ordinary MAE1.163428 versus1.188732; delta−0.025304[−0.048207,−0.005344].
- Q90 signed bias−5.399239 versus−5.591957: underprediction is reduced but remains substantial.
- Ordinary signed bias+0.191181 versus+0.203547.
- Q90 recall+0.831 percentage points[+0.187,+1.625], and false-Q90 rate+0.089 points[+0.006,+0.205].

Thus reconstruction error improves in both concentration ranges, while high-value detection trades some extra false alarms for higher recall. Versus matched masks, both tail and ordinary MAE improve at K0. Versus the old linear chemistry branch, ordinary improvement is established but its extra Q90 improvement is not; do not credit every K0 tail gain uniquely to the nonlinear decoder.

At K5, integrated Q90 MAE6.587181 is essentially the same point estimate as retained6.586994, but its difference interval[−0.041405,+0.036001] does not establish equality. Ordinary MAE0.990359 exceeds0.985640 with an interval crossing zero. The fixed chemistry tree remains better in overall K5 MAE, across all three partitions and nine fits; the isolated Q90 neural-versus-tree interval still crosses zero.

## Partition, seed and station heterogeneity

Each delta is candidate minus reference overall MAE. Counts are improved/worse/tied fits out of nine; repeated seeds are not new ecological samples.

| Fixed contrast | Partition142 | Partition143 | Partition144 | Better/worse/tie |
|---|---:|---:|---:|---:|
| neural_chemistry_vs_neural_masks_integrated_k0 | -0.056005 | -0.041574 | -0.008437 | 9/0/0 |
| neural_chemistry_vs_neural_masks_integrated_k5 | +0.025450 | -0.006676 | -0.004767 | 5/4/0 |
| neural_chemistry_vs_no_aux_integrated_k0 | -0.055079 | -0.044593 | -0.011126 | 8/1/0 |
| neural_chemistry_vs_no_aux_integrated_k5 | +0.026522 | -0.004361 | -0.005384 | 4/5/0 |
| neural_chemistry_vs_no_aux_k0 | -0.055079 | -0.028252 | -0.012882 | 8/1/0 |
| neural_chemistry_vs_no_aux_k5 | +0.004316 | -0.007040 | -0.006403 | 5/4/0 |
| neural_chemistry_vs_point_integrated_k0 | -0.059063 | -0.041547 | -0.015163 | 7/1/1 |
| neural_chemistry_vs_point_integrated_k5 | +0.025527 | -0.005939 | -0.008081 | 4/4/1 |
| neural_integrated_vs_tree_chemistry_k0 | -0.122937 | +0.014254 | -0.003261 | 7/2/0 |
| neural_integrated_vs_tree_chemistry_k5 | +0.040247 | +0.017538 | +0.034959 | 0/9/0 |
| nonlinear_vs_linear_chemistry_integrated_k0 | -0.050691 | -0.029292 | -0.001951 | 7/2/0 |
| nonlinear_vs_linear_chemistry_integrated_k5 | -0.000330 | -0.005420 | -0.001719 | 5/4/0 |
| nonlinear_vs_linear_chemistry_k0 | -0.050691 | -0.012951 | -0.002220 | 6/3/0 |
| nonlinear_vs_linear_chemistry_k5 | -0.008964 | -0.007298 | -0.001757 | 6/3/0 |
| tree_chemistry_vs_no_aux_k0 | -0.073137 | -0.017810 | -0.044712 | 9/0/0 |
| tree_chemistry_vs_no_aux_k5 | -0.041160 | -0.028697 | -0.049217 | 9/0/0 |

For integrated nonlinear chemistry versus retained point at K0,96 of172 station contributions favor the new model,75 favor the retained model and one is unchanged. The five largest positive station contributions account for41.1% of positive improvement mass. Partition gains are−0.059063,−0.041547 and−0.015163; none is omitted.

At K5 the same pair has partition deltas+0.025527,−0.005939 and−0.008081. The deterioration in partition142 outweighs the smaller improvements elsewhere. This is a support-adaptation heterogeneity issue to investigate, not permission to select a partition-specific model.

## Auxiliary availability

Availability is unchanged from the preceding study. Of10,520 unique observed-DOC query cells,10,216 have both auxiliary indicators,55 pH only,84 EC only and165 neither. Evaluation is dominated by both-known chemistry. Among210,907 genuinely DOC-missing grid cells,41,623 (19.735%) have either indicator;80.265% use the exact parent fallback. Their missing-DOC accuracy cannot be inferred from these observed-query results.

In the both-known group, nonlinear integrated MAE is1.808785 at K0 versus retained1.847751 and old-linear1.838974. At K5 it is1.608341 versus retained1.605484 and old-linear1.612157. EC-only and pH-only tails are unstable because their sample counts are small; EC-only error worsens relative to the old linear model, so the dominant group result should not be generalized to all availability patterns. Neither-known cells remain exactly unchanged at every K.

The setting is monthly DOC reconstruction with measured conventional chemistry. pH/EC monthly means need not be collected at the same instant, and the model is not a forecast before those observations arrive or a solution for entirely unmonitored sites.

## Source evidence and the next performance question

Source-validation integrated nonlinear chemistry MAE is1.769265 at K0 versus linear1.794341 and retained1.804067. At K5 it is1.598153 versus linear1.603447 and retained1.604602. The source-validation K5 advantage is much smaller than K0, and it does not establish a target K5 advantage. All chemistry checkpoints stop by epoch83; selected epochs range0–73 with one exact native fallback. A longer budget is not the first intervention suggested by these traces.

The remaining focused question is whether the existing support correction preserves the newly learned chemical response. Its GRU support basis was learned before chemistry was added. A next experiment can keep the successful K0 native predictions frozen and compare the existing support representation with a small representation from the selected chemistry decoder, fitting any calibration exclusively on source episodes/validation. That would test representation alignment while keeping the chemical-information gain intact. Current results motivate this hypothesis but do not prove that basis mismatch uniquely caused the K5 behavior. Retain the chemistry-tree comparison and all K curves; do not introduce a target-selected winner route.
