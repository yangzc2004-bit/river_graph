# Source-validation diagnostic: native-DOC encoder tuning

## Scope

This diagnostic reads only the 27 completed `runs/split*_seed*/{frozen,last_self,last_self_ecology}.json` training summaries. It does not read outer-test labels, prediction files, endpoint tables, or the main experiment analysis. The accompanying CSV lists every source summary and trace-derived quantity. No model was fitted or changed for this diagnostic.

Each score is the selected unweighted, pooled source-validation query MAE in mg/L. Seeds are averaged within each station partition, then the three partitions receive equal weight. The same validation data selected epoch and residual scale; these are development comparisons, not independent confirmation estimates. Repeated seeds do not provide additional ecological samples.

## Does encoder tuning improve validation?

| Station partition | Frozen | Last self | Last self + ecology | Gain: self | Gain: self + ecology |
|---|---:|---:|---:|---:|---:|
| 142 | 1.997370 | 1.980798 | 1.973959 | 0.016572 | 0.023411 |
| 143 | 1.892201 | 1.891869 | 1.887525 | 0.000332 | 0.004676 |
| 144 | 1.819910 | 1.818762 | 1.817604 | 0.001148 | 0.002305 |
| Equal-partition mean | 1.903160 | 1.897143 | 1.893030 | 0.006017 | 0.010131 |

Positive gain means lower MAE than the matched frozen encoder. Last-self tuning improves 8/9 seed–partition runs, with a mean gain of 0.006017 mg/L (0.316%). Adding ecology tuning also improves 8/9 runs, with a mean gain of 0.010131 mg/L (0.532%). These are small numerical benefits, concentrated in partition 142; they do not by themselves establish a statistically precise benefit across new station populations.

The encoder parameters are updating: selected last-self weight distances range from 0.0506 to 0.3190. Ecology weights also change in the ecology arm. Absolute parameter distances are recorded for reproducibility and are not comparable effect sizes across parameter groups.

## Is the 30-epoch ceiling binding?

| Mode | Reached epoch 30 | Reached 30 before patience exhausted | Selected epoch 30 | Selected epoch 26–30 |
|---|---:|---:|---:|---:|
| frozen | 5/9 | 5/9 | 3/9 | 5/9 |
| last_self | 5/9 | 4/9 | 3/9 | 4/9 |
| last_self_ecology | 6/9 | 5/9 | 4/9 | 5/9 |

Across all modes, 16/27 fits reach epoch 30; 14/27 reach it with fewer than five stale epochs, and 10/27 establish a new validation best at epoch 30. A fit reaching30 with five stale epochs is not counted as clearly budget-limited. Last-self and ecology tuning are therefore not the only modes that may benefit from more optimization: the frozen control also frequently reaches the ceiling while improving.

The pattern differs by partition:

- **Partition 142:** all nine fits select epoch 29 or 30. Validation MAE decreases by about 0.020–0.034 mg/L from epoch 25 to 30. Encoder tuning helps throughout the final stretch, and the ecology arm selects 30 for all three seeds. This supports incomplete optimization at the present budget.
- **Partition 143:** seeds 42/43 select epoch 5 and stop at 10 in all modes, using scale 0.5. Their encoder gains are only about 0.0002–0.0005 mg/L. For seed 44, frozen/self select 8 and stop 13, whereas the ecology arm continues to 30 and selects29, improving by 0.01350 mg/L over frozen. The early plateau and half-strength correction suggest station-dependent response or mismatch between weighted training improvement and pooled validation benefit; they do not prove that a larger encoder step will help.
- **Partition 144:** behavior is mixed. Seed 42 selects epochs25–28; seed 43 stops at25–27 and slightly favors frozen; seed 44 selects 30 in all modes and favors tuning. Its mean gains are much smaller than partition 142.

Training losses and validation losses have different weighting and different stations. Their numerical levels should not be compared directly. Their trends can identify a training–validation divergence, but cannot isolate its cause.

## What this says about the next bottleneck

There is evidence of **under-adaptation for some station partitions**, rather than evidence that the current encoder learning rate is universally too small. Partition 142 still improves at the budget boundary; partitions 143/144 include early plateaus or slight reversals. Increasing the encoder learning rate or unfreezing more layers would mix a new optimization choice with the unresolved duration effect.

These summaries contain only overall validation MAE and weighted source training loss. They cannot determine whether a tuning gain fixes high-DOC underprediction, introduces ordinary-concentration false positives, or simply shifts all predictions. A claim about improved selectivity needs source-validation tail/ordinary diagnostics, not the weight-distance or overall-MAE traces alone.

## Next focused experiment

**Change only the maximum training budget from 30 to 60 epochs.** Keep encoder learning rate 1e-5, GRU/head learning rates, patience 5, tail weight 2, three encoder modes, source OOF bases, validation queries, seed order, and checkpoint/scale selection unchanged. Start from the same original expert weights so the first 30 epochs reproduce the current training schedule; do not resume only selected weights without the original optimizer state.

Run the matched frozen control under the same 60-epoch budget. This isolates whether the incremental encoder benefit grows after the current ceiling, while allowing partition 143's early-stopping cases to remain unchanged. Keep the full validation trace and report the additional selected-MAE gain beyond each arm's own first 30-epoch best. If longer training helps all modes equally, it is an optimization-duration gain; if self/ecology pulls further ahead, it supports task-specific representation adaptation. No new layer or higher encoder learning rate is justified before this distinction is tested.
