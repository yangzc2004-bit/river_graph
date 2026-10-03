# Source-validation diagnostic: matched 60-epoch extension

## Scope and comparison

Only the 27 model JSON summaries and their training/source-validation traces were read from `doc_encoder_budget_v1`, together with the matching summaries from `doc_encoder_residual_v1`. No outer-test labels, predictions, endpoint tables or analyses were read. No extra model fit was performed. Every run is included in the accompanying CSV.

**All 27 prior trace prefixes reproduce exactly**, including validation candidates, chosen scales, training losses and parameter-distance records. For an early-stopped prior run, its complete trace is the prefix; otherwise epochs 0–30 are compared. Each extended run's best score through epoch 30 exactly equals its prior selected validation score. Thus the comparison isolates extra optimization after the original stopping horizon.

Scores below are pooled native-unit MAE within each source-validation run, averaged over seeds within partition and then equally over the three partitions. A positive duration gain is the run's best-through-30 MAE minus its final selected MAE. A positive change in tuning advantage means that a tuned arm gains more from the extension than its matched frozen arm. Checkpoint/scale selection uses these same validation data, so these comparisons diagnose model development; they are not independent performance confirmation.

## Overall duration effect

| Encoder mode | Best within 30 | Selected with cap 60 | Additional gain (mg/L) | Best epoch >30 | Total executed epochs |
|---|---:|---:|---:|---:|---:|
| frozen | 1.903160 | 1.886054 | 0.017106 | 5/9 | 287 |
| last_self | 1.897143 | 1.883320 | 0.013824 | 4/9 | 268 |
| last_self_ecology | 1.893030 | 1.869505 | 0.023525 | 5/9 | 307 |

The prior panel executed 645 epochs; the extension executed 862 epochs, adding 217 effective epochs. Fourteen of 27 fits select a new best after epoch 30. **No fit reaches epoch 60:** all stop by epoch 55 with five stale epochs; the latest selected checkpoint is epoch 50. The new cap is therefore not binding under the unchanged patience rule.

Additional optimization improves all three modes on average. It does not uniformly amplify encoder tuning: the last-self advantage over frozen shrinks from 0.006017 to 0.002734 mg/L, while the self-plus-ecology advantage grows from 0.010131 to 0.016549 mg/L. Both tuned modes beat frozen in 6/9 runs at the larger budget, compared with 8/9 at the original budget.

## Partition-level decomposition

| Partition | Mode | Best within 30 | Selected with cap 60 | Additional gain | Change in advantage over frozen |
|---|---|---:|---:|---:|---:|
| 142 | frozen | 1.997370 | 1.947714 | 0.049656 | +0.000000 |
| 142 | last_self | 1.980798 | 1.940408 | 0.040390 | -0.009266 |
| 142 | last_self_ecology | 1.973959 | 1.939105 | 0.034854 | -0.014802 |
| 143 | frozen | 1.892201 | 1.892201 | 0.000000 | +0.000000 |
| 143 | last_self | 1.891869 | 1.891869 | 0.000000 | +0.000000 |
| 143 | last_self_ecology | 1.887525 | 1.853430 | 0.034096 | +0.034096 |
| 144 | frozen | 1.819910 | 1.818247 | 0.001663 | +0.000000 |
| 144 | last_self | 1.818762 | 1.817682 | 0.001081 | -0.000582 |
| 144 | last_self_ecology | 1.817604 | 1.815979 | 0.001625 | -0.000038 |

### Partition 142: mainly a generic optimization benefit

All nine fits improve after epoch 30. Frozen gains 0.049656 mg/L, more than last-self (0.040390) or self-plus-ecology (0.034854). Tuning remains better on the partition average, but its advantage narrows: self-plus-ecology versus frozen falls from 0.023411 to 0.008609 mg/L. For seed 43, frozen slightly overtakes both tuned modes. This is evidence that part of the earlier tuning advantage reflected faster progress under a short budget.

### Partition 143: one delayed ecological adaptation, not a uniform effect

Seeds 42/43 reproduce their early stop at epoch 10 and selected epoch 5 in every mode; frozen/self seed 44 also remain unchanged at a stop of 13 and selected epoch 8. Only seed 44 self-plus-ecology continues: it improves from 1.854232 to 1.751946 mg/L, selects epoch 49 and stops at 54. Its selected residual scale rises from 0.5 to 1.0. This is a joint benefit of the learned residual trajectory and validation-selected correction strength; it should not be attributed to encoder weights alone.

That one run supplies the entire partition-143 duration gain and more than the net panel-wide increase in the ecology arm's advantage. As a sensitivity description, the ecology arm's mean advantage at cap 60 is 0.004144 mg/L across the other eight runs, compared with 0.016549 across all nine. The run remains in every main average; this calculation identifies concentration of the gain, not a reason to exclude it.

### Partition 144: little additional gain and no broad tuning amplification

Duration gains are only 0.0011–0.0017 mg/L per mode. The ecology advantage is essentially unchanged, while last-self loses some of its small lead. Seed 42 tuning already stopped exactly at epoch 30 under the five-stale-epoch rule and does not continue; the frozen arm improves slightly at epoch 32. Seed 44 improves in all modes. This partition gives little evidence that an enlarged budget especially benefits encoder tuning.

## Interpretation

**The duration gain is largely shared optimization, with a substantial but isolated delayed benefit from ecology tuning.** The extension resolves the obvious 30-epoch ceiling, rather than establishing that a broadly more adaptable encoder consistently needs longer training. Repeated seeds are optimization replications on the same validation station set, not independent ecological cohorts.

These overall-MAE traces do not determine whether the gains repair high-DOC errors or improve ordinary-concentration selectivity. They also cannot establish that a higher encoder learning rate or station-balanced training would help. Those would be distinct experiments.

A further increase above 60 epochs is not supported by the present stopping trajectories. If another optimization-only experiment is pursued, the remaining question is whether patience 5 stops some runs during an early shallow plateau before a delayed benefit can develop. This is suggested by the contrast within partition 143, but remains untested; a minimum training window or a patience change should be isolated from architecture and learning-rate changes. No new experiment is launched by this diagnostic.
