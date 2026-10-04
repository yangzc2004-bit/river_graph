# Interpretation: source-trained readout for the fixed DOC expert

## Decision

Retain the existing off expert with its legacy support representation and ecological integration as the main candidate. The new source-trained readout is a completed development experiment, not a replacement. Training the readout gives no clear incremental target benefit over refreshing the hidden representation with the existing fixed readout. It also does not resolve the K5 disadvantage of the refreshed representation relative to the existing legacy method. No K-dependent route is constructed from these target results.

The comparison isolates a small support-adaptation change: the native DOC predictions, recurrent expert, forest, ecological memory and K0 ecological mixture stay fixed. Only the 64-by-2 readout is trained, followed by the already specified source-validation calibration choices. Direct results therefore measure readout plus alpha/ridge reselection; integrated results additionally include positive-K ecological-mixture reselection.

## Complete primary curves

MAE is in mg/L. Columns give the legacy, refreshed-fixed and refreshed-learned representations.

| Stage | K | Legacy | Refreshed fixed | Refreshed learned |
|---|---:|---:|---:|---:|
| Direct | 0 | 1.806996 | 1.806996 | 1.806996 |
| Direct | 1 | 1.758826 | 1.758826 | 1.758826 |
| Direct | 3 | 1.631317 | 1.607645 | 1.613335 |
| Direct | 5 | 1.589387 | 1.590976 | 1.589982 |
| Integrated | 0 | 1.803538 | 1.803538 | 1.803538 |
| Integrated | 1 | 1.755368 | 1.755368 | 1.755368 |
| Integrated | 3 | 1.620090 | 1.599087 | 1.604441 |
| Integrated | 5 | 1.564984 | 1.572464 | 1.570974 |

K0 is exactly unchanged. K1 is also exactly unchanged: centering one support observation yields a rank-zero shape design, and the adapter enables the shape term only for K>1. K1 equality is a structural negative control, not evidence of inadequate statistical power.

## Does training the readout help?

All deltas below are learned minus reference; negative MAE deltas favor the learned readout. Intervals use 5,000 paired whole-station bootstrap draws with identical station multiplicities across overlapping partitions, seed means within partition, then equal partition weighting.

| Stage | K | Reference | Delta MAE | 95% interval |
|---|---:|---|---:|---|
| Direct | 3 | Refreshed fixed | +0.005690 | [-0.003904, +0.015861] |
| Direct | 5 | Refreshed fixed | -0.000994 | [-0.003334, +0.001160] |
| Integrated | 3 | Refreshed fixed | +0.005354 | [-0.004141, +0.015427] |
| Integrated | 5 | Refreshed fixed | -0.001490 | [-0.003791, +0.000481] |
| Direct | 3 | Legacy | -0.017982 | [-0.033791, -0.001869] |
| Direct | 5 | Legacy | +0.000596 | [-0.004223, +0.006357] |
| Integrated | 3 | Legacy | -0.015649 | [-0.031047, +0.000022] |
| Integrated | 5 | Legacy | +0.005990 | [-0.001234, +0.014262] |

The learned direct model retains a 1.10% K3 advantage over legacy, but the refreshed-fixed model already gave a larger advantage. This does not establish additional value from learning the readout. The integrated K3 interval crosses zero by a small amount; it is reported as estimated, without rounding its upper bound into a positive finding. K5 changes relative to the fixed readout are only 0.06% direct and 0.09% integrated. None of the four learned-versus-fixed MAE intervals excludes zero; this is not an equivalence claim.

## Source validation and heterogeneous transfer

Five of nine fits select a nonzero checkpoint, at epochs 1–5; four retain epoch zero. Mean checkpoint-selection loss decreases from 1.731654 to 1.730017. No selected fit is duration-limited. After final alpha/ridge selection, source-validation direct K3 MAE improves from 1.660738 to 1.659677 and K5 from 1.615358 to 1.613119. Integrated source-validation K3 improves from 1.648791 to 1.647541 and K5 from 1.604835 to 1.602288. These small source improvements do not establish target transfer.

At K3, learned-minus-fixed direct deltas by partitions 142/143/144 are -0.000520, -0.009245 and +0.026834; integrated deltas are -0.000556, -0.010226 and +0.026843. Thus two partitions improve, but degradation in partition 144 dominates their mean. Three of nine seed-partition fits improve, two worsen and four are exactly unchanged, for both stages.

At K5, learned-minus-fixed direct partition deltas are -0.000194, -0.000150 and -0.002638. Integrated deltas are 0, -0.001834 and -0.002635. Direct fits improve in 3/9, worsen in 1/9 and tie in 5/9; integrated fits improve in 2/9, worsen in 1/9 and tie in 6/9. Exact ties can arise from epoch-zero selection or calibration suppressing the shape term. The mean K5 improvement is small even where directions agree.

Against legacy, integrated K5 worsens in all three partitions (+0.015486, +0.001317, +0.001166), with 3/9 fits improving and 6/9 worsening. The K3 advantage over legacy is concentrated: partition 144 contributes most of the mean improvement, and the five largest positive station contributions supply about 56% of total positive station-gain mass. The 172 unique target stations contribute 10,520 unique station-months; repeated seeds do not create additional ecological samples.

## Tail and detection consequences

Learning the readout does not materially repair high-DOC underprediction. Relative to refreshed-fixed, K3 Q90 MAE changes by +0.006630 direct and +0.007066 integrated; K5 changes by -0.002556 and -0.002893. All four tail intervals cross zero. At K5, learned integrated Q90 MAE is 6.634382 versus legacy 6.586994, a delta of +0.047388 with interval [-0.002223, +0.094747]. Its Q90 signed bias is -5.045860 mg/L, compared with -4.965678 for legacy.

The K3 learned-versus-legacy advantage is an ordinary-DOC result: ordinary MAE decreases by -0.023217 direct (interval [-0.041027, -0.006689]) and -0.021038 integrated ([-0.038154, -0.005146]); Q90 point changes are unfavorable and uncertain. At K5, learned-integrated recall is 65.105%, precision 77.837% and false-positive rate 2.188%, close to legacy's 65.134%, 77.807% and 2.188%. No planned recall or false-positive-rate interval establishes a directional change; one direct K5 false-positive interval ends exactly at zero.

## Scientific implication

Refreshing the representation can improve three-observation adaptation, but an additional global 128-parameter projection does not reliably strengthen that result or improve five-observation adaptation. The tiny source-selection gain, early checkpoints and target heterogeneity suggest that extending this same optimization is a low-priority next step. The current evidence supports preserving the strong native model and legacy station adaptation while addressing the remaining high-DOC residual structure through a separately motivated experiment. These data do not identify a unique cause of the limited readout transfer.

All eight planned contrasts, including unfavorable outcomes, remain in the analysis. The partitions have been examined in prior development rounds; this is an incremental development comparison, not independent external confirmation. No new model, K route, threshold or support schedule was selected from target performance.
