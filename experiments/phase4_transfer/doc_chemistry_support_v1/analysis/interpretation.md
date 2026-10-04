# Chemical-state support adaptation: interpretation

## Decision

Retain the source-validation-selected chemical support representation as a **conditional DOC reconstruction candidate**. It improves transfer from three DOC observations; the five-observation improvement is smaller and uncertain. Keep the complete source-selected K curve, rather than assembling a new curve from target-set winners. The chemistry tree remains an important strong comparator.

This study changes support coordinates and their existing validation-fitted calibration parameters. Native neural, forest and ecological predictions are fixed. It provides evidence about how observations transfer across months, without attributing the gain to new backbone training or river messages.

## What the fixed experiments show

MAE is in mg/L. Each estimate averages training seeds within a station partition, then averages the three partitions equally. Confidence intervals use 5,000 paired whole-station draws with joint station multiplicities across the overlapping partitions. There are 172 unique query stations and 10,520 unique station-month queries; the three seeds repeat these observations.

| Pipeline / representation | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Neural, legacy recurrent basis | 1.773213 | 1.741419 | 1.621943 | 1.579294 |
| Neural, masks augmented | 1.773213 | 1.741419 | 1.619626 | 1.580424 |
| Neural, chemistry augmented | 1.773213 | 1.741419 | 1.604624 | 1.566684 |
| Neural, source-selected basis | 1.773213 | 1.741419 | 1.603418 | 1.569843 |
| Neural + ecology, legacy | 1.764947 | 1.738600 | 1.614240 | 1.568820 |
| Neural + ecology, masks augmented | 1.764947 | 1.738600 | 1.613222 | 1.569553 |
| Neural + ecology, chemistry augmented | 1.764947 | 1.738600 | 1.595816 | 1.556278 |
| Neural + ecology, source-selected | 1.764947 | 1.738600 | 1.595851 | 1.556939 |
| Chemistry tree, legacy | 1.802261 | 1.768522 | 1.565090 | 1.537905 |
| Chemistry tree, chemistry augmented | 1.802261 | 1.768522 | 1.560229 | 1.535715 |
| Chemistry tree, source-selected | 1.802261 | 1.768522 | 1.560343 | 1.535534 |
| Retained general neural + ecology reference | 1.803538 | 1.755368 | 1.620090 | 1.564984 |

The full thirteen-model table, including tree mask augmentation, is in `findings.md` and `k_curves.csv`.

### Three observations: useful additional transfer information

Direct chemistry augmentation reduces neural MAE by **1.068%** relative to the legacy basis: ΔMAE −0.017319, 95% CI [−0.029367, −0.006163]. It also improves on the matched four-dimensional mask-state basis: −0.015001 [−0.026953, −0.003758]. Thus, the result is supported before ecological integration and is not explained solely by adding two coordinates or by encoding auxiliary availability.

After ecological integration, chemistry augmentation improves on legacy by −0.018423 [−0.029765, −0.008136] and on masks by −0.017406 [−0.028567, −0.007286]. The source-selected representation retains essentially the same estimated gain: **1.139%**, ΔMAE −0.018389 [−0.029182, −0.008617]. Its improvement over the retained general reference is **1.496%**, −0.024239 [−0.044780, −0.007424]. These are comparisons of complete pipelines with their source-validation-fitted support parameters, not fixed-coefficient feature effects.

### Five observations: encouraging estimates, heterogeneous evidence

Integrated chemistry augmentation improves by an estimated 0.799% relative to legacy, −0.012542 [−0.029491, +0.001753]. The source-selected curve improves by 0.757%, −0.011881 [−0.028914, +0.002077]. Both intervals include zero. Against the retained general reference, the selected curve is lower by 0.514%, −0.008045 [−0.023271, +0.003925]. The direction is favorable, but this does not establish a broad five-observation advantage.

The direct chemistry-versus-mask contrast at K5 is favorable, −0.013740 [−0.028601, −0.000359]. This narrow interval exclusion should be read alongside the other fixed contrasts rather than substituted for the uncertain integrated comparison.

### The tree comparison remains unresolved in favor of the neural model

The selected chemistry tree achieves 1.560343 at K3 and 1.535534 at K5. The selected integrated neural model has higher error by +0.035508 [+0.007203, +0.067219] at K3 and +0.021405 [−0.004193, +0.049284] at K5. All three partition means favor the tree at both K values; 8/9 and 9/9 individual partition-seed fits favor it, respectively. The neural model has not demonstrated superiority over the chemistry tree.

Chemical support augmentation itself adds little to this tree: selected-versus-legacy deltas are −0.004747 [−0.012791, +0.001445] at K3 and −0.002371 [−0.011774, +0.006778] at K5. This is consistent with, but does not uniquely prove, more chemical information already being captured by the tree's native prediction.

## Tail, ordinary concentrations and station heterogeneity

For the integrated selected curve at K3, improvement over legacy occurs in both regions:

- Q90 MAE: 6.829922 → 6.793270; Δ−0.036652 [−0.065764, −0.008071].
- Ordinary MAE: 1.012325 → 0.995840; Δ−0.016485 [−0.028377, −0.006114].

At K5, Q90 MAE falls from 6.587181 to 6.518383 and ordinary MAE from 0.990359 to 0.984046, but both paired intervals include zero. Q90 signed bias moves from −5.099975 to −4.904412; substantial underprediction remains, and the bias-change interval includes zero.

This is not an established high-DOC detection improvement. Selected-versus-legacy recall changes are +0.048 percentage points [−0.207, +0.329] at K3 and −0.233 [−0.744, +0.242] at K5. False-Q90 changes are +0.017 [−0.004, +0.043] and +0.008 [−0.042, +0.065] percentage points. Selected precision is 79.534% and 77.791%, respectively. Error reconstruction and threshold detection should remain separate findings.

Selected integrated-versus-legacy partition deltas are:

| Station partition | K3 ΔMAE | K5 ΔMAE |
|---|---:|---:|
| 142 | −0.008837 | −0.033612 |
| 143 | −0.005730 | −0.000837 |
| 144 | −0.040599 | −0.001194 |

All three partition means improve, with 7/9 individual fits improving at K3 and 5/9 at K5. The largest contribution switches partitions. At K3, 103 unique stations improve and 68 worsen; the top five improving stations account for 52.6% of positive gain mass. At K5, 98 improve and 73 worsen; the top-five positive-gain share is 66.9%. One station is unchanged. These figures explain why consistent partition signs do not imply uniform station benefits.

## Source validation and the role of chemical measurements

The source-selected integrated representation uses chemistry coordinates in 8/9 packages at K3 and 6/9 at K5; the remaining choices are one mask representation at K3 and two legacy/one mask at K5. Its source-validation MAE is 1.651356 at K3 and 1.591066 at K5, versus legacy 1.654174 and 1.598153. The new target comparison evaluates that fixed selection rule; it does not replace its choices with the target-best representation.

All eighteen PCA fits identify both added dimensions. Each uses 232 source stations, of which 231 have active chemistry, and 40,091–42,162 active source station-months. No DOC label or DOC-observation mask selects the PCA rows. Within-source-station covariance and a fixed source-global projection center preserve target-row-local chemistry information. The chemical representation still originates in a source-trained DOC decoder, so it is not a wholly unsupervised ecological embedding.

K0 and K1 are exactly unchanged across representations. At K0 there is no support update. At K1 the centered shape design has rank zero: only the ordinary support offset is available, regardless of the added coordinates. This is an algebraic control.

Most evaluated queries have both auxiliary indicators: 10,216 of 10,520 unique queries. In this group, integrated selected MAE improves from 1.655317 to 1.634837 at K3 and from 1.608341 to 1.596326 at K5. The pH-only group is small (55 cells at 7 stations) and worsens at K3, 0.903887 → 1.111871. Its K5 estimate improves slightly. The EC-only group is also small (84 cells); it should not support strong subgroup conclusions. Neither-available predictions are exactly unchanged by construction.

Only 19.735% of genuinely missing-DOC station-months have at least one measured auxiliary indicator, compared with much higher availability in the observed evaluation. The chemistry-conditioned gain therefore does not establish the same gain across the entire missing-DOC grid. Retain the general predictor and the explicit unavailable-chemistry fallback.

## Scientific reading and retained scope

Chemical-state coordinates help determine **where a few observed DOC residuals should be transferred across a station's months**. The clearest evidence is K3 reconstruction, with matched mask controls and fixed native experts. This supports representation alignment as a useful component of the model; it does not establish a unique physical mechanism, neural superiority to trees, or improvement at all support budgets.

Retain the complete source-selected chemical pipeline as the new conditional candidate, keep the general model and chemistry tree in all comparisons, and preserve the full K curve. Do not convert the K3 result into a new target-driven route, or promote the marginal K5 estimate into a guaranteed improvement.

## Verification and figures

All nine packages passed independent replay: 18 source-PCA refits, 54 adapters, 27 mixers and 108 representation choices reproduce exactly. All 468 query panels (2,006,940 rows), 144 legacy controls and 162 low-K controls reproduce exactly. No neural or forest was refitted in this study.

- `chemical_support_k_curves.pdf` / `.png`: all four representation curves within each of the three pipelines.
- `chemical_support_paired_effects.pdf` / `.png`: all 22 fixed paired MAE contrasts with 95% station-bootstrap intervals.
- `plot_figures.py`: reproducible plotting from the saved metric tables. Both raster outputs were visually inspected for scale, labels, overlap and clipping.
- `analysis_manifest.json`: analysis source, fixed input identities, independent replay and output bindings.
