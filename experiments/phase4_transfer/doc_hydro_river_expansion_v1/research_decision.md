# Hydro node expansion and DOC reconstruction

## Result and decision

The expanded real-upstream branch achieves **1.714729 mg/L MAE**. This is
**0.480%** below the complete current model (95% paired station interval
**0.082 to 0.858%**), but only **0.063%** below the preceding environmental-state
branch (**-0.065 to 0.204%**). Coverage expansion contributes a small increment,
not a substantial new reconstruction advance. The same selected source-validation
population selected the retained predictors and the new branch checkpoints;
these intervals describe development results, not independent confirmation.

Keep the complete current predictor as the released model. Retain the expanded
river branches as completed research candidates. The primary remains the
declared dynamic real-upstream model; do not replace it with the numerically
better uniform arm after this comparison.

## What changed

Nine packages, partitions 142/143/144 and seeds 42/43/44, completed **36 new fits**.
The existing ecological encoder, causal observation-aware GRU, source-similarity
model, environmental trees and observed-DOC river branch remain fixed. Only the
eligible environmental donor pool expands: 71 additional ST357 nodes per
partition can provide temperature/discharge, ecology, calendar and daily
hydrology, with every water-quality value, mask and support feature removed.

The whole receiving fold/validation role is excluded from that pool. Other
known-atlas nodes may supply hydro covariates even if they were outside the
original labelled source role. The labelled DOC experience library does not
expand. This is transductive use of known environmental covariates, not external
basin validation. Historical all-cohort static/ecology normalization is retained;
monthly hydro statistics use the original source training cells. Saved source
environmental states replay exactly in every package.

The same zero-initialized residual operator uses two heads, dimension 32, at most
20 upstream candidates, causal monthly lags 0/1/3, 30 epochs and patience 5.
Thirty-six fits include real upstream, uniform allocation, hydro-matched real
upstream and hydro-matched non-ancestor controls. All branches add to the same
fixed observed-DOC prediction. Complete model training anchors are fitted,
not complete-neural OOF predictions. No new hydrology was downloaded.

## Accuracy and controls

| Model | MAE mg/L | Q90 MAE mg/L | log1p MAE |
|---|---:|---:|---:|
| Complete current model | 1.723002 | 8.895024 | 0.248207 |
| Observed upstream DOC | 1.717226 | 8.852561 | 0.247090 |
| Previous upstream environmental state | 1.715815 | 8.848287 | 0.246815 |
| Expanded dynamic upstream state, primary | 1.714729 | 8.841157 | 0.246694 |
| Expanded uniform upstream state | 1.713772 | 8.834304 | 0.246560 |
| Expanded matched real upstream | 1.714690 | 8.843898 | 0.246699 |
| Expanded matched non-upstream | 1.715839 | 8.852472 | 0.246881 |
| Strong environmental trees | 1.866362 | 9.394458 | 0.275922 |

The primary improves over the complete model in 8/9 packages and all three
partition means. Its **additional** gain over the previous state version occurs
in 4/9 packages and two partition means. Against the observed-DOC-only branch,
MAE decreases **0.145% [0.016, 0.302%]**. Station-equal improvement is **0.321%**
versus complete and **0.091%** versus previous state. Its 8.125% advantage over
the strong tree baseline includes the already-existing complete model advantage
and is not the contribution of this upgrade.

Dynamic allocation is **0.056% worse** than uniform allocation
[-0.108, -0.008% reduction]. The matched real-upstream advantage is only
**0.067% [-0.192, 0.299%]** over matched non-ancestors. The additional covariates
do not establish a distinct connectivity advantage or a benefit of learned
allocation. Uniform attention includes a zero-message slot; it is not an
unconditional average with forced messages.

Primary Q90 error decreases **0.606% [0.002, 1.339%]** versus complete, but only
**0.081% [-0.040, 0.203%]** versus previous state. The tail comprises 91 unique
stations and 774 unique cells across partitions. Seed repetitions do not add
ecological samples. Four of nine primary fits select epoch zero, preserving
the observational anchor rather than forcing an environmental correction.

## Coverage and where the increment occurs

| Partition | Original state coverage | Expanded coverage | Newly supported cells | Lost cells |
|---|---:|---:|---:|---:|
| 142 | 28.997% | 31.888% | 66 | 0 |
| 143 | 32.236% | 36.147% | 120 | 0 |
| 144 | 63.463% | 64.347% | 31 | 0 |

Equal-partition coverage rises **41.565 to 44.127%**, or **2.562 percentage
points**. The 217 newly supported cells belong to only seven receiving stations.
Their MAE improves **0.049% [-0.104, 0.191%]**, an unstable descriptive estimate.
Previously supported cells improve **0.176% [-0.218, 0.582%]** relative to the
previous state version; 96 stations/4284 unique cells without either usable
river channel remain exactly unchanged.

The >200-km group contains only four stations. Its narrow conditional bootstrap
interval requires many empty-partition redraws and is not evidence of broadly
reliable long-distance transfer. Tables retain the numerical intervals and
redraw counts; plots omit intervals for groups with fewer than 20 stations.
The 140 unique receiving stations contain 7897 unique station-month cells;
there are 8857 partition-cell occurrences because roles overlap across partitions.

## Next scientific experiment

Coverage expansion alone does not explain the remaining graph limitation.
The next change should give message **values** an explicit upstream-to-local
contrast: upstream environmental state minus the receiving site's state, with
the same pool, path descriptors, capacity and fixed anchors. The present value
projection receives absolute upstream state; hydro differences currently enter
only the attention/path features. A contrast tests whether the river branch can
supply a condition difference that the local environment model has not already
explained. Compare real paths, matched non-ancestors and uniform allocation.
Do not add depth or attention heads on the basis of this small gain.

This is a hypothesis for the next version. Monthly lag weights are information
allocation, not observed travel-time coefficients. The field studies of arrivals
and buffering remain separate evidence; these model outputs do not establish a
causal effect of whole-network morphology on DOC.

## Reproduction and verification

Run with the uv-managed environment, using `scripts/run_ladder.py --experiment
doc-hydro-river-expansion-v1`. Coverage audit, full input/checkpoint verification,
5000-draw analysis, independent arithmetic and plotting use the corresponding
`audit_`, `verify_`, `analyze_`, `check_` and `plot_doc_hydro_river_expansion_v1.py`
scripts. All nine input banks were rebuilt, all 36 checkpoints replayed bitwise,
and missing-state predictions equal the fixed observational anchor exactly.
Primary MAE/Q90/log arithmetic and station bootstrap agree independently for
all three anchors. Full pytest: **1450 passed, 3 skipped**; Ruff passes;
historical provenance audit exits 0.

Two PNG/PDF/SVG figures were visually inspected. A mixed-comparator axis label
was corrected, and sparse-stratum interval display was repaired. These are
analysis/presentation changes only; executed training sources and weights are
unchanged. Large fitting caches and checkpoint files remain local. Old results,
geographical tests and independent-basin evaluations are preserved.
