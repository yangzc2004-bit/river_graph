# Decision: retain the chemical-coordinate parent, omit the extra kernel

## Source-only conclusion

The bounded RBF interpolation does **not** provide coherent incremental benefit in source-validation station cross-validation. Retain the completed chemical-coordinate model and leave this kernel out of the candidate for fresh-partition confirmation. Do not open outer target outcomes for this kernel or select a target-specific route.

This decision concerns an extra interpolator of the residual left after the fixed linear support update. It does not retract the preceding study's evidence that chemical-state support coordinates improve K3 reconstruction.

## Conditional station CV is the relevant diagnostic

Five shared folds split source-validation stations; eta and bandwidth are selected using active query error on the other four folds. Held-fold errors are pooled within each training seed, seeds are averaged within partition and partitions receive equal weight. The parent expert, linear calibration and representation were already selected on source validation. Thus this is **conditional incremental CV**, not an independent or fully out-of-fold evaluation of the complete parent pipeline.

| Pipeline | K | Frozen parent MAE | Chemistry-kernel CV MAE | ΔMAE | Relative error increase | Better partitions | Better seed fits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct neural | 3 | 1.658708 | 1.662812 | +0.004104 | 0.247% | 0/3 | 2/9 |
| Direct neural | 5 | 1.603627 | 1.604917 | +0.001290 | 0.080% | 1/3 | 2/9 |
| Neural + ecology | 3 | 1.651356 | 1.656726 | **+0.005370** | **0.325%** | **0/3** | **2/9** |
| Neural + ecology | 5 | 1.591066 | 1.592138 | **+0.001071** | **0.067%** | **1/3** | **1/9** |
| Chemistry tree | 3 | 1.673123 | 1.678272 | +0.005149 | 0.308% | 0/3 | 1/9 |
| Chemistry tree | 5 | 1.603398 | 1.607420 | +0.004021 | 0.251% | 1/3 | 2/9 |

The integrated chemistry kernel also loses to the availability-only kernel by +0.005365 at K3 and +0.001057 at K5. The availability-only kernel is effectively inactive in many station groups and has very small unfavorable CV differences from the parent (+0.000005 and +0.000014).

The integrated chemistry-kernel partition deltas are:

| Station partition | K3 ΔMAE | K5 ΔMAE |
|---|---:|---:|
| 142 | +0.007928 | −0.008987 |
| 143 | +0.003112 | +0.003837 |
| 144 | +0.005072 | +0.008364 |

At K5, one individual seed fit accounts for the favorable direction in partition142. The other partitions do not reproduce it. Active-query CV gives the same reading: integrated deltas are +0.005372 and +0.001071. This result is not being diluted by a large inactive-chemistry cohort in source validation.

## Why the fitted validation curve was insufficient

Full-validation tuning chooses eta and bandwidth on exactly the observations used to score the selected candidate. Because eta0 copies the parent, its selected active MAE cannot exceed the parent's selection score. Inactive predictions are fixed, so the same ranking holds for the full-query score.

For the integrated neural model, this optimistic fit gives:

| K | Parent | Full-validation selected kernel | Apparent improvement |
|---|---:|---:|---:|
| 3 | 1.651356 | 1.645588 | 0.349% |
| 5 | 1.591066 | 1.584281 | 0.426% |

Those fitted improvements reverse sign under held-station evaluation of the incremental settings. The contrast is evidence of limited station-to-station stability in the added calibration, rather than evidence for a useful kernel awaiting a favorable target partition. No confidence interval or statistical significance claim is attached to this source-only pilot.

## The operator had sufficient support to act

Source-validation active support counts are essentially K: 3.000 at K3 and 4.9997 at K5, with every evaluated station having at least two active donors. Full-validation selection activates the chemistry kernel in 6/9 packages at each K. Conditional CV selects nonzero eta in 32/45 K3 folds and 27/45 K5 folds for the integrated model.

In the full-validation selected integrated model, average absolute native corrections are about 0.078 mg/L at K3 and 0.079 mg/L at K5. Approximately two thirds of query predictions change. The failure is therefore not simply that the branch never activates or that nearly all support readings lack chemistry.

Source distances are finite and nondegenerate in all 18 definitions. They come from 232 source stations, 231 with active chemistry, with at most 24 sampled active months per station. No DOC labels or held-station coordinates fit these distance scales. K0/K1 remain exact parent controls; inactive-query corrections remain zero.

## Tail errors and station concentration

The full-validation selected integrated kernel lowers fitted Q90 MAE from 9.266300 to 9.245454 at K3 and from 9.020913 to8.989323 at K5; fitted ordinary MAE decreases from 0.991832 to 0.988650 and from 0.956492 to 0.952970. These are **selection-set diagnostics**, not cross-validated tail improvements. The saved candidate station statistics support conditional CV for full/active MAE, not a separate cross-validated tail analysis.

For incremental-CV integrated chemistry versus parent:

- K3: 64 stations improve, 70 worsen, 6 are unchanged. Positive gain mass is 0.008049 but harm mass is 0.013419. The five largest harmful stations account for 50.3% of harm.
- K5: 84 stations improve, 50 worsen, 6 are unchanged. Positive gain mass is 0.007820 but harm mass is 0.008891. The five largest harmful stations account for 51.5% of harm.

More improved stations therefore does not necessarily imply lower cell-weighted MAE. All gains and harms remain in the analysis rather than being used to invent a station route.

## Scope and next step

The source-validation population contains 7,197 unique station-months at 140 unique stations, with 8,047 partition-cell occurrences. Seeds repeat the same observations. These scores must not be compared numerically with the preceding outer-station development scores as if they were the same cohort.

A coherent next step is fresh-partition confirmation of the **existing source-selected chemical-coordinate model**, with the retained general model and chemistry tree as fixed comparators. The added kernel remains a completed negative calibration experiment. No further bandwidth grid, K-specific promotion or target evaluation is justified by this pilot.

The source-only analysis script validates all 36 model/K groups per package, all kernel candidate scores and 540 held-fold choices. Independent product replay is recorded separately. `analysis_manifest.json` binds the fixed execution sources, products, interpretation and verification report.
