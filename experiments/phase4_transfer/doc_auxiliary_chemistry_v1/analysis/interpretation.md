# Interpretation: auxiliary chemistry for DOC reconstruction

## Scientific result and recommendation

Measured pH and specific conductance add useful information in the observed-chemistry reconstruction setting. The matched ExtraTrees model captures a repeatable overall gain. The current frozen-feature neural correction shows a K0 high-DOC gain, but does not sustain an overall advantage after five support observations. Retain the original point model as the general neural candidate, preserve the chemistry tree as a strong conditional benchmark, and retain the neural chemical correction as a mechanism result. Do not create a new K-specific or availability-specific winner route from target scores.

All thirteen curves and twelve specified comparisons have been evaluated with 5,000 joint whole-station bootstrap draws. Independent replay passed all nine packages: 27 heads, 27 tree probes, 81 direct and 36 integrated validation refits, 468 query panels and 144 parent-control panels. Neural, query and fallback products reproduce exactly; forest numerical replay differs by at most1.066e−13 within the declared1e−12 tolerance. Final analysis bindings include the stable verification report. These are reused development station partitions, not a newly untouched external confirmation dataset.

## Principal results

All MAEs are native DOC mg/L, using the unchanged GRU support basis. Negative delta favors the named candidate.

| Comparison | K | Candidate MAE | Reference MAE | Delta MAE [95% interval] |
|---|---:|---:|---:|---|
| Tree chemistry vs tree no-aux | 0 | 1.802261 | 1.847481 | −0.045220 [−0.084230, −0.010683] |
| Tree chemistry vs tree no-aux | 5 | 1.537905 | 1.577596 | −0.039691 [−0.056866, −0.024888] |
| Neural chemistry integrated vs retained point | 0 | 1.792258 | 1.803538 | −0.011280 [−0.027004, +0.003412] |
| Neural chemistry integrated vs retained point | 5 | 1.571309 | 1.564984 | +0.006325 [−0.003346, +0.017163] |
| Neural chemistry integrated vs chemistry tree | 0 | 1.792258 | 1.802261 | −0.010003 [−0.061093, +0.040881] |
| Neural chemistry integrated vs chemistry tree | 5 | 1.571309 | 1.537905 | +0.033405 [+0.005082, +0.064418] |

The tree's reduction against its matched no-aux control is 2.448% at K0 and 2.516% at K5. Both improve in all three partitions and all nine fits. At K5, both ordinary and Q90 error improve: ordinary delta −0.028570 [−0.042739, −0.016310]; Q90 delta −0.138584 [−0.225612, −0.045865]. The gain is not just a changed high-value detection threshold: Q90 recall and false-positive-rate intervals both include zero.

Neural chemistry integrated improves against the retained point estimate in all nine fits at K0, but the station-level interval still crosses zero. Consistent training-seed directions do not replace ecological station-level uncertainty. At K5 its average error rises, with a large deterioration in partition142 and small improvements in the other two. It loses to the chemistry tree in all nine fits at K5.

The tree chemistry estimate is also lower than the retained integrated neural point at K5 (1.537905 vs1.564984), a descriptive difference of 1.73%. That particular pair was not a primary bootstrap contrast; the specified, supported comparison is to the matched tree no-aux control and to the new chemistry neural candidate.

## Chemical values versus measurement availability

All six new arms use exactly the same actual pH-or-EC availability footprint. The no-aux arm zeros auxiliary inputs; the masks arm retains availability flags; the chemistry arm adds measured values and, for the neural correction, hidden-state interactions. Thus values and the selected measurement footprint are not conflated.

The neural integrated chemistry-minus-masks overall deltas are −0.007271 [−0.019566, +0.003668] at K0 and +0.006808 [−0.001350, +0.016020] at K5. Overall value-added is not established for this neural add-on. Its K0 Q90 MAE improves beyond masks by −0.054005 [−0.095942, −0.010459]. Recall rises by0.439 percentage points [0.041,0.946], accompanied by a false-Q90 rise of0.051 points [0.009,0.109]. This is a detection/error tradeoff, not an unqualified success.

The tree masks control remains close to its no-aux control: K0/K5 MAE1.851629/1.579903 vs1.847481/1.577596. Chemistry reaches1.802261/1.537905. These descriptive mask-control curves are consistent with value information beyond availability. The fixed tree bootstrap contrast was chemistry versus no-aux; no additional chemistry-versus-masks significance test is added after seeing results.

## High DOC and ordinary DOC

Compared with retained point integration, neural chemical integration reduces K0 Q90 MAE from7.179187 to7.105102 (delta −0.074084 [−0.127043, −0.019321]). Q90 signed bias moves from−5.591957 to−5.487527: severe underprediction remains. Ordinary signed bias rises from+0.203547 to+0.223680. Q90 recall rises0.639 points [0.225,1.228] while false-Q90 rises0.077 points [0.027,0.145].

At K5, neural chemical integration has Q90 MAE6.609833 versus6.586994 for retained point, and ordinary MAE0.990460 versus0.985640. Both uncertainty intervals for error differences cross zero; the K0 tail improvement does not carry through the fixed K5 support procedure.

Relative to the chemistry tree, neural integration has lower Q90 MAE at K0 by0.267180 [0.082109,0.459733], with higher recall and higher false-Q90 rate. At K5 the tree has lower overall MAE, while the Q90-specific pairwise interval includes zero. The two models distribute error differently; low overall MAE and high-DOC sensitivity should be presented together.

## Partition and seed consistency

Each number is candidate-minus-reference overall MAE. Counts are improved fits out of nine, not independent new ecological samples.

| Fixed contrast | Partition142 | Partition143 | Partition144 | Improved fits |
|---|---:|---:|---:|---:|
| neural_chemistry_vs_neural_masks_integrated_k0 | -0.005812 | -0.011206 | -0.004795 | 8/9 |
| neural_chemistry_vs_neural_masks_integrated_k5 | +0.021092 | +0.000028 | -0.000697 | 5/9 |
| neural_chemistry_vs_no_aux_integrated_k0 | -0.007448 | -0.013137 | -0.008663 | 8/9 |
| neural_chemistry_vs_no_aux_integrated_k5 | +0.026737 | +0.000859 | -0.000436 | 2/9 |
| neural_chemistry_vs_no_aux_k0 | -0.007448 | -0.013137 | -0.009588 | 8/9 |
| neural_chemistry_vs_no_aux_k5 | +0.012920 | -0.000014 | -0.000947 | 6/9 |
| neural_chemistry_vs_point_integrated_k0 | -0.008372 | -0.012255 | -0.013213 | 9/9 |
| neural_chemistry_vs_point_integrated_k5 | +0.025857 | -0.000518 | -0.006362 | 5/9 |
| neural_integrated_vs_tree_chemistry_k0 | -0.072246 | +0.043546 | -0.001310 | 5/9 |
| neural_integrated_vs_tree_chemistry_k5 | +0.040577 | +0.022959 | +0.036678 | 0/9 |
| tree_chemistry_vs_no_aux_k0 | -0.073137 | -0.017810 | -0.044712 | 9/9 |
| tree_chemistry_vs_no_aux_k5 | -0.041160 | -0.028697 | -0.049217 | 9/9 |

For tree chemistry versus no-aux at K5,116 of172 unique station contributions favor chemistry,55 favor no-aux and one is unchanged. The top five improved stations account for32.7% of positive improvement mass. At K0,106 favor chemistry,65 favor no-aux, and the top-five positive share is46.7%. Gains are repeatable across partitions, while ecological heterogeneity remains.

## Availability and scope

The fixed evaluation contains10,520 unique observed DOC station-months:10,216 have both auxiliary indicators,55 only pH,84 only EC and165 neither. The both-known group dominates the evaluation. Neither-observed predictions exactly retain the original parent pipeline: neural arms copy point direct/integrated products, and tree arms copy the prior tree. There are no Q90 events in the neither group; its tail performance is unidentifiable here.

The genuinely DOC-missing grid has210,907 cells. Only41,623 (19.735%) have either auxiliary indicator, of which31,913 (15.131%) have both. The remaining169,284 cells (80.265%) use the existing fallback. These fractions describe coverage, not validated missing-cell accuracy. The study cannot establish a chemistry benefit over the whole DOC-missing reconstruction grid.

Within the both-known group, tree chemistry versus tree no-aux MAE is1.840961 vs1.891378 at K0 and1.574785 vs1.616577 at K5. Neural chemical integration versus retained point is1.838974 vs1.847751 at K0 and1.612157 vs1.605484 at K5. EC-only and pH-only groups are small; their tail estimates are flagged unstable, and the EC-only tree overall error does not show the same benefit. Do not extrapolate the dominant both-known result to those subgroups.

The auxiliary values are retrospective same-calendar-month means, potentially sampled at different instants. This is DOC reconstruction when conventional water chemistry is measured, not a forecast before sampling or a wholly unmonitored-station task.

## Source validation and the next focused model experiment

Mean source-validation integrated chemistry MAE is1.794341 at K0 and1.603447 at K5, compared with retained point1.804067/1.604602. The K5 validation gain is already small. The matched tree chemistry improves source validation versus tree no-aux at both K0 (1.910344 vs1.940955) and K5 (1.612055 vs1.634779). These source trends and target results favor adding informative covariates; they do not establish why this neural readout captures less of the benefit.

A scale-zero head reproduces the native point base. It does not by itself force identical positive-K predictions on auxiliary-active cells, because each new arm fits its support and ecological choices on the active validation subset. The inactive final product is nevertheless copied exactly at every K. Reference arms continue to use their full-validation choices.

All27 neural fits stop before the120-epoch cap (maximum72); chemistry checkpoints are epochs2–15 with all scales nonzero. The frozen-feature head has683 parameters and learns a linear native residual over original features, auxiliary slots and bilinear hidden-value terms. More training of this head is not the immediate bottleneck suggested by the traces.

The next focused hypothesis is that a trainable nonlinear chemistry representation can use pH/EC jointly with the existing ecological/hydrological context more effectively than this frozen linear add-on. Test one compact chemical encoder integrated into the existing model, with matched no-aux/mask controls, fixed tree-chemistry benchmark, honest source OOF base and the exact absent-chemistry fallback. Keep support adaptation explicit because K0 improvements did not persist at K5. This is a proposed next experiment, not a demonstrated explanation or an invitation to select routes from these target scores.
