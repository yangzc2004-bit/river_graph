# Recurrent clock comparison for spatial DOC transfer

The encoder/head architecture, inputs, original initialization, trainable scope, loss and
budget are held constant. Each new clock refits the last spatial self layer, ecology encoder,
GRU, decay layer and native residual head. Forests, the ecological profile and support-basis
weights remain fixed. The changed design input is the scalar supplied to recurrent decay.
Legacy is the completed parent off expert;
unseen-neutral and flow-window are two new matched-budget fits. No forest is retrained.

Unseen-neutral sets only the decay-layer age scalar to zero until a local DOC observation
has been visible. It does not disable decay or change the raw M1 age feature. Flow-window
starts its clock at twelve months within each queried causal window, resets on visible
monthly discharge, otherwise increments to a cap of twelve, and skips padded dates.
The flow clock uses monthly discharge visibility, not daily descriptor validity. It encodes
shared-state decay rather than river travel time, residence time, or a physical causal effect.

Checkpoint and global residual scale use source-validation K0 MAE, including the exact
zero-residual fallback. The same source-validation support grids are used independently
for every arm. Ecological gamma is selected for each arm; integrated contrasts therefore
include those mixing choices in addition to the recurrent-clock change.

All fourteen model curves and the eight prespecified clock-versus-legacy comparisons are
reported. Constant-only support adaptation remains a diagnostic; bootstrap comparisons use
the unchanged GRU support basis at K0 and K5. These are reused development partitions,
with no model, K, route or endpoint selected from target outcomes.

Intervals use 5,000 paired whole-station bootstrap draws. Repeated station identities
are jointly resampled across partitions; seed means average within partition and partitions
receive equal weight. Q90 includes ties at the source-training threshold. Negative MAE/FPR
deltas favor the candidate; recall is reported alongside false-positive rate.

The loader verifies 54 exact copied-context or legacy control comparisons.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_constant | 1.902823 | 1.866563 | 1.643669 | 1.609381 | 3.614595 | 0.562102 |
| context_gru_tuned_anchor | 1.902823 | 1.866563 | 1.634818 | 1.596001 | 3.591068 | 0.567520 |
| legacy_constant | 1.806996 | 1.758826 | 1.641713 | 1.591869 | 3.611671 | 0.563919 |
| legacy_gru_tuned_anchor | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| unseen_neutral_constant | 1.820109 | 1.776430 | 1.649756 | 1.591954 | 3.576141 | 0.570778 |
| unseen_neutral_gru_tuned_anchor | 1.820109 | 1.776430 | 1.637231 | 1.576967 | 3.537784 | 0.578676 |
| flow_window_constant | 1.816609 | 1.770284 | 1.647455 | 1.587807 | 3.573703 | 0.571477 |
| flow_window_gru_tuned_anchor | 1.816609 | 1.770284 | 1.620974 | 1.572724 | 3.535629 | 0.579349 |
| legacy_integrated_constant | 1.803538 | 1.755368 | 1.632934 | 1.574914 | 3.572022 | 0.571864 |
| legacy_integrated_gru_tuned_anchor | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| unseen_neutral_integrated_constant | 1.815765 | 1.771027 | 1.641373 | 1.578807 | 3.570857 | 0.571839 |
| unseen_neutral_integrated_gru_tuned_anchor | 1.815765 | 1.771027 | 1.626570 | 1.570798 | 3.559818 | 0.574182 |
| flow_window_integrated_constant | 1.811108 | 1.764629 | 1.639740 | 1.579000 | 3.568037 | 0.572653 |
| flow_window_integrated_gru_tuned_anchor | 1.811108 | 1.764629 | 1.610432 | 1.568986 | 3.559236 | 0.574389 |

## Direct recurrent-clock contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| unseen_neutral_vs_legacy_gru_tuned_anchor_k0 | +0.013113 [-0.006404, +0.032840] | +0.079114 [+0.011918, +0.156578] | +0.005709 [-0.015183, +0.025001] | -1.526850 [-2.632213, -0.621724] | -0.120009 [-0.249544, -0.020759] |
| unseen_neutral_vs_legacy_gru_tuned_anchor_k5 | -0.012420 [-0.037814, +0.006255] | -0.093581 [-0.245504, +0.039454] | -0.004603 [-0.019642, +0.009298] | -0.035458 [-0.533101, +0.665676] | -0.007943 [-0.075324, +0.060394] |
| flow_window_vs_legacy_gru_tuned_anchor_k0 | +0.009614 [-0.013205, +0.033315] | +0.070666 [+0.007696, +0.145921] | +0.002678 [-0.022777, +0.027802] | -1.451070 [-2.507522, -0.632712] | -0.112894 [-0.252574, -0.006524] |
| flow_window_vs_legacy_gru_tuned_anchor_k5 | -0.016663 [-0.041728, +0.002856] | -0.098013 [-0.245268, +0.028596] | -0.008612 [-0.024298, +0.006602] | -0.091778 [-0.710302, +0.567956] | -0.000548 [-0.085066, +0.097329] |

## Integrated recurrent-clock contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| unseen_neutral_vs_legacy_integrated_gru_tuned_anchor_k0 | +0.012227 [-0.006296, +0.030415] | +0.079449 [+0.015407, +0.152861] | +0.004670 [-0.015315, +0.022760] | -1.427288 [-2.500815, -0.542612] | -0.102432 [-0.227144, -0.007992] |
| unseen_neutral_vs_legacy_integrated_gru_tuned_anchor_k5 | +0.005814 [+0.000946, +0.010517] | +0.021230 [+0.005404, +0.041212] | +0.004236 [-0.001267, +0.009211] | -0.019912 [-0.347531, +0.251178] | -0.024393 [-0.071064, +0.027789] |
| flow_window_vs_legacy_integrated_gru_tuned_anchor_k0 | +0.007570 [-0.012527, +0.028023] | +0.075794 [+0.012962, +0.149316] | -0.000270 [-0.022574, +0.020477] | -1.451070 [-2.463736, -0.695514] | -0.085273 [-0.223040, +0.019036] |
| flow_window_vs_legacy_integrated_gru_tuned_anchor_k5 | +0.004002 [-0.001616, +0.009594] | +0.020420 [+0.004840, +0.038314] | +0.002385 [-0.003823, +0.008400] | -0.003394 [-0.428226, +0.390976] | -0.015460 [-0.070364, +0.053083] |

## High and ordinary DOC with detection tradeoffs

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.558635 | -6.606236 | 1.257359 | +0.482039 | 58.898% | 78.465% | 1.909% |
| context_gru_tuned_anchor | 5 | 6.694451 | -5.165304 | 1.008430 | +0.231465 | 65.013% | 77.677% | 2.208% |
| legacy_gru_tuned_anchor | 0 | 7.167345 | -5.532017 | 1.194119 | +0.213345 | 64.363% | 77.424% | 2.195% |
| legacy_gru_tuned_anchor | 5 | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| unseen_neutral_gru_tuned_anchor | 0 | 7.246459 | -5.704454 | 1.199828 | +0.174261 | 62.836% | 77.921% | 2.075% |
| unseen_neutral_gru_tuned_anchor | 5 | 6.548695 | -4.881216 | 1.002945 | +0.173953 | 65.510% | 77.865% | 2.195% |
| flow_window_gru_tuned_anchor | 0 | 7.238011 | -5.695718 | 1.196797 | +0.204773 | 62.912% | 77.911% | 2.082% |
| flow_window_gru_tuned_anchor | 5 | 6.544264 | -4.868971 | 0.998936 | +0.182350 | 65.453% | 77.786% | 2.203% |
| legacy_integrated_gru_tuned_anchor | 0 | 7.179187 | -5.591957 | 1.188732 | +0.203547 | 64.064% | 77.636% | 2.157% |
| legacy_integrated_gru_tuned_anchor | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| unseen_neutral_integrated_gru_tuned_anchor | 0 | 7.258636 | -5.751752 | 1.193401 | +0.166631 | 62.637% | 78.026% | 2.055% |
| unseen_neutral_integrated_gru_tuned_anchor | 5 | 6.608224 | -5.035774 | 0.989876 | +0.174607 | 65.114% | 78.008% | 2.163% |
| flow_window_integrated_gru_tuned_anchor | 0 | 7.254980 | -5.770497 | 1.188462 | +0.188996 | 62.613% | 77.902% | 2.072% |
| flow_window_integrated_gru_tuned_anchor | 5 | 6.607414 | -5.034667 | 0.988025 | +0.180842 | 65.131% | 77.962% | 2.172% |

## Source-validation checkpoint selection

| Arm | Selected epoch range | Epochs executed | Mean initial MAE | Mean selected MAE |
|---|---:|---:|---:|---:|
| legacy | 27–72 | 473 | 1.990984 | 1.804428 |
| unseen_neutral | 27–81 | 454 | 1.990984 | 1.822566 |
| flow_window | 28–89 | 474 | 1.990984 | 1.818777 |

Source-validation scores describe model selection, separately from target performance.
All selected scales, support alpha/ridge values, ecological gamma choices and candidate
scores are retained. Partition/seed directions, transformed errors and station gain/harm
concentration accompany the aggregate means. Intervals crossing zero do not establish
equivalence or absence of a clock effect.
