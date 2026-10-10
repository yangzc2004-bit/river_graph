# Upstream-to-local message contrasts for DOC reconstruction

## Result and decision

**Do not adopt this contrast-value branch.** Its MAE is **1.715889 mg/L**, compared
with **1.714729 mg/L** for the preceding absolute-state primary. The additional
MAE reduction is **-0.068%** (95% paired station interval **-0.213 to +0.068%**).
Only 1/9 packages improves, and none of the three partition means improves.
Source-Q90 error rises from **8.841157 to 8.853289 mg/L**, a **0.137%** increase;
the corresponding gain interval is **-0.268 to -0.032%**. This is a completed
negative mechanism test, not a reason to replace the preceding model.

The new branch remains **0.413%** better than the complete current predictor
[0.020, 0.778%], but that accumulated advantage includes the retained observed-DOC
branch. The contrast adds only **0.078% [-0.009, 0.166%]** to that observed branch.
Do not attribute existing performance to this unsuccessful upgrade. These are
source-validation development results; the same population selected the retained
predictors and new checkpoints, rather than an independent confirmation sample.

Keep the released complete predictor and all completed river candidates. Retain
the preceding absolute-state river representation for subsequent research; do
not promote a different control arm simply because its point estimate is lower.

## The experiment

Nine packages, partitions 142/143/144 and seeds 42/43/44, completed **36 new fits**
in approximately 237 summed package-seconds. For each genuine upstream candidate
and allowed lag, the message changes from the absolute upstream environmental
state to:

`environmental_state(upstream, t-lag) - environmental_state(receiver, t)`.

The receiving state is explicitly at the current query month, not the source's
lagged month. Both states use the same frozen environmental encoder/GRU, with
receiving DOC/pH/conductance values, masks, histories and support removed.
Neither state is a DOC concentration. This operation is a feature contrast, not
a physical gradient or transport equation.

Candidate nodes, real paths, lag eligibility, attention/path descriptors,
query features, matching and nominal capacity are identical to the preceding
expanded-pool experiment. Keep 20 candidate slots, lags 0/1/3, 2 heads x32,
12098 trainable parameters, 30 epochs, patience 5, Adam 0.001, batch 512 and
source-Q90 tail weight 2. Each branch adds to the same frozen observed-DOC anchor,
not to the selected absolute-state correction. Uniform allocation retains the
same nominal capacity but leaves allocation/reliability parameters inactive.

The environmental bank uses known ST357 covariates except the whole receiving
fold/validation role. The labelled DOC experience bank remains source-only.
This is known-atlas transductive covariate use, not external basin validation.
Historical preprocessing is retained; complete-neural source training anchors
are fitted predictions, not complete-neural OOF predictions. Daily features are
retrospective within-month descriptors. Monthly lag weights express information
allocation, not measured river travel times.

## Accuracy and mechanism comparisons

| Model | MAE mg/L | Q90 MAE mg/L | log1p MAE |
|---|---:|---:|---:|
| Complete current model | 1.723002 | 8.895024 | 0.248207 |
| Observed upstream DOC anchor | 1.717226 | 8.852561 | 0.247090 |
| Previous absolute upstream state, primary | 1.714729 | 8.841157 | 0.246694 |
| Previous absolute uniform state | 1.713772 | 8.834304 | 0.246560 |
| Upstream-to-local contrast, primary | 1.715889 | 8.853289 | 0.246856 |
| Uniform contrast | 1.715912 | 8.851760 | 0.246846 |
| Matched real-upstream contrast | 1.715867 | 8.852491 | 0.246824 |
| Matched non-upstream contrast | 1.716690 | 8.854154 | 0.246987 |
| Strong environmental trees | 1.866362 | 9.394458 | 0.275922 |

Dynamic versus uniform contrast has a **0.001% [-0.042, 0.046%]** MAE difference.
Matched real versus non-ancestor contrast has a **0.048% [-0.089, 0.168%]**
advantage. Neither establishes an allocation or connectivity benefit. Assigned
control-path descriptors are fictitious matched slots, not actual river paths.
The matched real contrast also worsens Q90 relative to its absolute counterpart
by **0.097%**, with the gain interval **-0.212 to -0.002%**.

Primary log1p error worsens **0.065% [-0.266, 0.121%]** relative to the preceding
absolute state. Station-equal MAE worsens **0.072%**. Five of nine primary fits
select epoch zero; all three partition-143 primary fits keep the observed anchor.
Epoch zero counts are 5/9 for uniform and matched upstream, and 8/9 for matched
non-upstream. No environmental correction is forced when validation prefers
its fixed observational anchor.

## Availability and error strata

Upstream environmental coverage remains exactly **31.888%, 36.147% and 64.347%**
in the three partitions, or **44.127%** under equal-partition averaging. No cell
gains or loses support relative to the absolute-state version. The comparison
therefore isolates message representation rather than increased data coverage.

There are 140 unique receiving stations and 7897 unique station-month cells,
with 8857 partition-cell occurrences. The Q90 subset contains 91 unique stations
and 774 unique cells. Seed repetitions are not additional ecological samples.
At cells with usable state messages, the extra MAE gain is **-0.151%**
[-0.534, 0.234%]. Environmental-state cells without usable upstream DOC have
**-0.673%** [-2.158, 0.615%] gain. Neither reveals a useful condition-specific
contrast improvement. Cells without either river channel remain exactly unchanged.

Station/path strata overlap and are descriptive. The >200-km band contains four
stations and requires 8338 empty-partition redraws; its conditional interval is
retained in tables but omitted in figures. Do not present it as evidence of
reliable long-distance deterioration.

## What the result contributes and the next step

An environmental difference is not automatically a more informative river
message. Within this fixed architecture, replacing absolute state with a
contrast sacrifices some tail performance and does not reveal a distinct real-
connectivity advantage. This does not establish why the contrast failed or that
river transport is irrelevant. It does establish that this particular algebraic
change is not a useful performance upgrade.

Pause further absolute/contrast/attention variants of this same environmental
state family. The next useful model experiment should address the **training
residual** before adding capacity: simulate wholly unmonitored receiving sites
with their local anchor fitted without that receiving fold, then learn the river
correction on those held-station errors. Current environmental river corrections
learn against fitted complete-model source anchors, while evaluation uses unseen
receiving sites. The possible residual distribution mismatch deserves a source-
only comparison; it is a hypothesis, not an explanation proved by this result.

Retain genuine upstream paths, observed-DOC innovation messages and non-ancestor
controls. Use the existing junction/storage descriptors to test an explicit
propagation/attenuation operator after the training task is aligned, rather than
merely appending those already-present descriptors again. Any monthly operator
must remain an information-transport model; separate field arrival/buffering
studies cannot turn these monthly weights into measured travel times or causal
whole-network morphology effects. New development remains in source roles;
this round does not retune on geographical or external test outcomes.

## Reproduction and checks

- Training: `uv run python scripts/run_ladder.py --experiment doc-river-state-contrast-v1`.
- Full input/checkpoint replay: `uv run python scripts/verify_doc_river_state_contrast_v1.py --rebuild-inputs`.
- Analysis: `uv run python scripts/analyze_doc_river_state_contrast_v1.py --bootstrap-draws 5000`.
- Independent arithmetic: `uv run python scripts/check_doc_river_state_contrast_calculations_v1.py`.
- Figures: `uv run python scripts/plot_doc_river_state_contrast_v1.py`.

All nine full input banks rebuild exactly. Independent subtraction agrees for
all source/validation real/matched/control arrays; path features, queries and
validity remain bitwise identical to the preceding absolute-state version.
All 36 checkpoints replay bitwise, and absent-state predictions preserve the
observed anchor exactly. Independent MAE/Q90/log arithmetic and 5000-draw station
CIs match for four comparators. Full pytest: **1454 passed, 3 skipped**;
Ruff passes; historical provenance audit exits 0.

Two PNG/PDF/SVG figures were visually inspected. Crowded MAE tick labels were
repaired by reducing tick count; data and executed training code are unchanged.
Large fitting caches and checkpoints remain local; small results and immutable
execution sources are retained with this version. Previous experiment files
and frozen paper endpoints are unchanged.
