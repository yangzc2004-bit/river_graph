# Upstream environmental states: completed research decision

## Answer

Upstream environmental states expand the river branch beyond months with usable
upstream DOC, but contribute little additional predictive accuracy in this
experiment. The combined observed-DOC/state model reduces MAE by **0.417%**
against the complete current model and **0.082%** against the preceding observed
river model. The latter is the actual increment of this upgrade. Its paired
station interval crosses zero. Matched non-upstream states perform at least as
well at the point estimate; these results do not establish a distinct native-MAE
advantage of true upstream connectivity.

Keep the complete current model as the released predictor, the observed-river
branch as a candidate, and this state extension as a completed mechanism result.
Do not promote the numerically best non-upstream arm into the primary model
after seeing the comparison. This version does not justify more attention
heads, deeper GCNs or longer lag grids.

## What was fitted

Nine retained source-role packages (partitions 142/143/144, seeds 42/43/44), each
with five new neural state branches: **45 fits completed**. Original complete
DOC, matched environmental trees and observed-river outputs are reused bitwise.
Receiving stations have no DOC, pH or conductance input. The new state branch
reuses the frozen ecological/self encoder and twelve-month GRU. It removes all
source water-quality input channels and sends environmental hidden states,
ecology, monthly hydro/masks and daily-derived flow descriptors over genuine
upstream paths and causal 0/1/3-month history. The learned output starts at zero.

The combined model applies the already fixed observed correction and a separate
state correction sequentially in log1p space. There is no per-cell model switch.
Hydro-matched upstream/non-upstream arms share path slots, operator, observed
anchor and common hydro validity. Training libraries exclude receiving station
folds; controls exclude complete ancestry and COMID aliases. Source validation
selects checkpoints, with epoch0 retained as a candidate. Complete and observed
training predictions are fitted, not complete-neural-model OOF outputs.

## Fixed comparison

Losses first average seeds within partition, then weight the three partitions
equally. Native values are mg/L. Q90 thresholds come from source training labels.

| Model | MAE | Q90 MAE | log1p MAE |
|---|---:|---:|---:|
| Complete current model | 1.723002 | 8.895024 | 0.248207 |
| Observed upstream DOC | 1.717226 | 8.852561 | 0.247090 |
| Environmental state only | 1.720945 | 8.895400 | 0.247863 |
| **Observed DOC + environmental state (primary)** | **1.715815** | **8.848287** | **0.246815** |
| Uniform upstream state | 1.718931 | 8.890391 | 0.247477 |
| Hydro-matched upstream state | 1.715799 | 8.853490 | 0.246827 |
| Hydro-matched non-upstream state | 1.715176 | 8.856303 | 0.246832 |
| Matched environmental trees | 1.866362 | 9.394458 | 0.275922 |

| Contrast | MAE reduction % [95% paired station interval] |
|---|---:|
| Combined / complete | 0.417 [−0.012, 0.816] |
| Combined / observed river | **0.082 [−0.046, 0.226]** |
| State only / complete | 0.119 [0.012, 0.260] |
| State attention / uniform state | −0.117 [−0.356, 0.099] |
| Matched upstream / matched non-upstream | **−0.036 [−0.199, 0.124]** |

Combined-model native MAE improves in 8/9 packages versus complete and 4/9 versus
the observed anchor; all three partition means improve in both comparisons.
Its log1p MAE improves 0.561% versus complete (interval 0.135–1.007%) and 0.111%
versus observed (−0.058–0.298%). Q90 gains are 0.525% and 0.048%, respectively,
with intervals crossing zero. Station-equal gains are 0.231% versus complete
and 0.026% versus observed. This prevents the accumulated improvement from being
presented as the contribution of the new state branch alone.

The aggregate experiment contains 140 unique receivers and 7,897 unique DOC
station-months (8,857 partition-cell occurrences). Three repeated training seeds
do not create additional ecological observations. All intervals use 5,000 paired
whole-station draws jointly across the three partitions. Source validation is
also the checkpoint-selection population; these are development estimates,
not independent geographical or external validation.

## Coverage and mechanism

| Partition | Usable upstream DOC | Usable upstream state | State without DOC |
|---|---:|---:|---:|
| 142 | 22.73% | 29.00% | 6.26% |
| 143 | 21.74% | 32.24% | 10.50% |
| 144 | 52.48% | 63.46% | 10.98% |

Equal-partition usable-message coverage rises from 32.32% to 41.57%, an increase
of 9.25 percentage points. In the state-without-DOC group (33 stations, 692 unique
cells), the combined model improves 1.121% versus either anchor, but its interval
is −0.411–2.419%. Across all cells lacking observed DOC, improvement is 0.094%
(−0.039–0.311%). Where both channels are absent, every prediction is unchanged.
The positive subgroup estimate is a useful direction, not an established gain.

The state-only dynamic branch selects epoch 0 in 7/9 packages; combined does so
in 5/9. Uniform state pooling is numerically better than learned state allocation,
and matched non-upstream state correction is slightly better than matched real
upstream correction. Thus a more expressive allocator is not the current
demonstrated source of performance. Similar environments can provide much of
the same information as these coarse monthly upstream latent states.

This experiment uses environmental states from retained **source-training
stations**. It opens source months without DOC, not an entirely new library of
all hydro-only stations along every reach. Many receiver-months still lack any
eligible supported upstream source. That restricted footprint is an explicit
remaining bottleneck. Latent states are not imputed DOC observations, and the
monthly lag slots are not measured travel times.

## Next research step

Return the improvement effort to information content and spatial coverage.
First measure how much additional upstream coverage genuinely hydro-only nodes
could provide, without opening their DOC/pH/conductance or adding receiving-fold
observations to the source residual library. Use source training/validation roles
for that development. Only if coverage increases substantially, build a new
version that distinguishes measured upstream DOC departures from environmental
states and compares true upstream with matched non-ancestor sources again.

The existing monthly hidden states mostly repeat local ecological/hydro signals.
A productive future river branch should represent source-versus-receiver flow
changes, branch arrivals and shared-channel processing at the available temporal
resolution. It should preserve a local-only control. The channel/tracer studies
motivate those inputs; they do not supply transport coefficients for ST357.
Do not claim structural transport benefit until the matched river comparison
improves, and do not enlarge the architecture to compensate for missing data.

## Reproduction and completed checks

- `run_ladder.py --experiment doc-dynamic-river-state-v1`: fixed 45-fit execution.
- `verify_doc_dynamic_river_state_v1.py --rebuild-inputs`: all nine source states,
  banks and products reconstruct exactly; every checkpoint prediction replays
  bitwise; original anchors and absent-state fallbacks are exact.
- `analyze_doc_dynamic_river_state_v1.py --bootstrap-draws 5000`: complete and
  stratified comparisons, native/log/tail losses, station-equal estimates.
- `check_doc_dynamic_river_state_calculations_v1.py`: independent raw-parquet
  arithmetic and both primary 5,000-draw station intervals match.
- `plot_doc_dynamic_river_state_v1.py`: two PNG/PDF/SVG figures, visually checked.
- Full suite: **1,446 passed, 3 skipped**; additional strengthened unsorted-fold
  test passed; Ruff green; historical artifact audit exit 0. Existing warnings
  are retained in the local test log.

Executed source copies and configurations are retained. New fitted weights and
large input caches stay local; small products and analysis are versioned. No
historical experiment, endpoint, paper figure or fitted model was overwritten.
