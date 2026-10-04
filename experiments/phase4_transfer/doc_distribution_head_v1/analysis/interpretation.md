# Interpretation: conditional distribution heads for DOC reconstruction

## Recommendation

Keep the existing point model with ecological integration as the main candidate. Retain the mixture head as a secondary candidate with slightly lower aggregate MAE, rather than declare it a replacement or a resolved high-DOC improvement. The single-Gaussian head does not establish an advantage over the point model, and the mixture does not establish a further advantage over the single Gaussian. All twelve planned overall-MAE intervals cross zero.

The new heads are small linear models on the same frozen 550-dimensional native-head features: hidden state, existing extras and their existing interactions. The encoder, GRU, point head, forests, ecological profile and old GRU support basis are frozen. Source-only feature normalization and density initialization are fitted. New heads minimize unweighted log-residual NLL; source-validation native MAE chooses the checkpoint and correction scale. Integrated comparisons also include the separately selected ecological mixture and support calibration.

The fixed target analysis is complete and independently verified. All nine packages passed replay: 18 saved density heads, 504 query panels, 126 source-validation adapter/mixer refits and 144 point-control panels were reproduced. Saved-state full-grid distributions and query predictions are bitwise exact; independent median and native-formula checks agree within floating-point precision. No model or threshold was selected from these target results.

## Complete principal K curves

MAE is in mg/L. All rows use the unchanged GRU support basis; constant-only curves remain in `findings.md` and the CSVs.

| Model | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Context forest | 1.902823 | 1.866563 | 1.634818 | 1.596001 |
| Point direct | 1.806996 | 1.758826 | 1.631317 | 1.589387 |
| Single direct | 1.807848 | 1.761706 | 1.630858 | 1.587454 |
| Mixture direct | 1.801902 | 1.755767 | 1.634876 | 1.582850 |
| Point integrated | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| Single integrated | 1.805217 | 1.759075 | 1.620028 | 1.563876 |
| Mixture integrated | 1.799271 | 1.753136 | 1.623813 | 1.562012 |

Mixture integrated has lower K0/K5 estimates than point integrated by 0.237%/0.190%, while K3 is worse. These observed differences do not justify choosing a different model by K. Density mixture components remain statistical components; they are not assigned to physical DOC sources or ecological processes.

## All twelve fixed point-performance contrasts

Delta is candidate minus reference; negative favors the candidate. Intervals use 5,000 paired whole-station bootstrap draws, joint station multiplicities across overlapping partitions, seed means within partition and equal partition weighting.

| Comparison | Stage | K | Delta MAE | 95% interval |
|---|---|---:|---:|---|
| Single minus point | Direct | 0 | +0.000852 | [-0.019416, +0.020469] |
| Single minus point | Direct | 5 | -0.001933 | [-0.013086, +0.007971] |
| Mixture minus point | Direct | 0 | -0.005094 | [-0.027677, +0.015423] |
| Mixture minus point | Direct | 5 | -0.006536 | [-0.018790, +0.006107] |
| Mixture minus single | Direct | 0 | -0.005946 | [-0.020508, +0.006932] |
| Mixture minus single | Direct | 5 | -0.004604 | [-0.014506, +0.005864] |
| Single minus point | Integrated | 0 | +0.001679 | [-0.020358, +0.022722] |
| Single minus point | Integrated | 5 | -0.001108 | [-0.005610, +0.003585] |
| Mixture minus point | Integrated | 0 | -0.004267 | [-0.028346, +0.017739] |
| Mixture minus point | Integrated | 5 | -0.002972 | [-0.011097, +0.004623] |
| Mixture minus single | Integrated | 0 | -0.005946 | [-0.020508, +0.006932] |
| Mixture minus single | Integrated | 5 | -0.001864 | [-0.006868, +0.002679] |

The intervals do not establish equivalence, but they also do not support a performance breakthrough from the new density head. The lower mixture estimates should be presented together with their uncertainty and the unfavorable partition/seed outcomes.

## Heterogeneity

Mixture-versus-point direct K0 deltas for partitions 142/143/144 are +0.006332, -0.010043 and -0.011571; direct K5 deltas are +0.001899, -0.008955 and -0.012553. Four of nine fit pairs improve, four worsen and one is an exact fallback, at both K0 and K5.

After ecological integration, K5 deltas are +0.002463, +0.000341 and -0.011719: only partition 144 improves. Across seed-partition fits, 3/9 improve, 5/9 worsen and 1/9 is unchanged. At station level, 91/172 unique stations have positive contributions and 81 have negative contributions; the largest five positive station contributions provide 42.4% of positive gain mass. The aggregate improvement is therefore not a uniform cross-station gain.

Mixture-versus-single direct K5 is positive in 6/9 fit pairs, with one fallback tie, but the paired overall interval still crosses zero. Both the single and mixture head retain exact point fallback for partition144/seed44. That does not create additional independent ecological observations; all comparisons use the same 172 unique target stations and 10,520 unique station-months.

## High and ordinary DOC

No planned Q90-MAE interval excludes zero. Mixture-versus-point integrated K5 Q90 MAE changes by -0.008754 [-0.040835, +0.023025], from 6.586994 to 6.578240. Ordinary MAE changes by -0.002274 [-0.011285, +0.006016], from 0.985640 to 0.983366. Q90 signed bias becomes slightly less negative (-4.965678 to -4.921038 mg/L), but the tail-error reduction is not established.

At K0, the mixture's lower overall estimate is driven by ordinary-DOC point improvement; its Q90 point estimate worsens. Integrated ordinary MAE changes by -0.007698, while Q90 MAE changes by +0.018866; both intervals cross zero. Its Q90 bias becomes more negative, from -5.591957 to -5.711557 mg/L.

High-value detection also needs its own readout. Direct K0 mixture recall declines by 1.076 percentage points, with interval [-2.342, -0.014]; direct single recall declines by 2.054 points [-4.281, -0.519]. Integrated single recall declines by 1.915 points [-4.137, -0.374]. Mixture integrated K0 recall declines from 64.064% to 63.128%, but that contrast's interval crosses zero. All false-positive-rate intervals cross zero. Thus lower overall MAE is not evidence of improved high-DOC detection.

At K5, mixture integrated recall/precision/false-positive rate are 65.296%/77.804%/2.191%, compared with point integrated 65.134%/77.807%/2.188%. Their recall and false-positive-rate differences remain uncertain. Full native/transformed errors, biases and all detection contrasts are retained in the analysis tables.

## Source fitting, density diagnostics and objective mismatch

Source-validation direct MAE improves from a mean baseline of 1.804428 to 1.786599 for single (0.99%) and 1.774160 for mixture (1.68%). These gains are larger than their target aggregate gains. That difference is compatible with validation selection effects or differences between station cohorts; this experiment does not identify one unique explanation.

Checkpoint choice follows native MAE, not NLL. Only 2/18 fits select the same epoch that minimizes recorded validation NLL. For example, mixture partition143/seed44 selects epoch50 for point performance: its selected source NLL is 0.040791 and validation NLL is 0.665039, compared with the lowest validation NLL of 0.572227 at epoch14. The selected validation NLL is also worse than its initial 0.610545. This is a concrete example of likelihood and point-reconstruction objectives favoring different checkpoints; low source NLL is not held-out point evidence.

All fits stop before the 100-epoch cap (single at most35, mixture at most60). One head per family returns scale zero and exactly preserves point predictions through every K. Epoch zero alone is not the same as fallback: single partition142/seed42 selects its source-initialized constant density at epoch0 with scale1. The analyzer distinguishes these cases and verifies every zero-scale product exactly.

The limited point gain is not accompanied by a general component collapse under the recorded diagnostics. Source and validation fractions near the sigma floor are zero, and mixture mean gaps below 0.01 are absent. Validation component means remain separated; their average gaps range about 0.251–0.417 in transformed-residual units. A small subset of rows has highly concentrated component weights (up to4.77% with high-component weight below0.01; up to6.36% above0.99), which does not imply universal collapse. Component summaries describe the unscaled latent residual density, not physical process states or calibrated native-DOC uncertainty.

## Implication

A more flexible residual distribution fits source data and modestly improves aggregate target point estimates, but it has not resolved the high-DOC reconstruction problem or demonstrated a stable advantage over the retained point model. Preserve the mixture as a documented candidate and keep the existing main model. Longer training of this same head is not motivated by the early stopping traces. Any further iteration should address why source-station improvements fail to transfer or add information relevant to the remaining errors, rather than treating lower NLL alone as success.

No density interval coverage or uncertainty-success claim is made. These remain previously examined development partitions, with all twelve contrasts reported and no target-based model, K, threshold or station route selected.
