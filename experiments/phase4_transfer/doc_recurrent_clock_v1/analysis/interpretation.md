# Interpretation: recurrent clocks for DOC station transfer

## Decision

Retain the legacy daily-head model with ecological integration and the unchanged GRU support basis. Neither new clock supports replacing it. The unseen-neutral integrated model has a small but resolved K5 MAE disadvantage; the flow-window integrated model also has an unfavorable point estimate, with its overall interval crossing zero. Both new integrated clocks worsen Q90 error at K0 and K5. Their direct K5 gains do not carry through to a better complete model.

The design intervention changes the scalar entering the recurrent decay layer. Raw M1 inputs, encoder/head architecture, original initialization, trainable scope, source forest OOF targets, loss and budget are matched. Each new clock refits the last spatial self layer, ecology encoder, GRU, decay layer and native residual head. Forests, the ecological profile and support-basis weights stay fixed. Legacy is reused from the completed parent rather than retrained. All eight planned contrasts and fourteen model curves are retained; no K-specific or station-specific route is constructed from these outcomes.

## Complete principal curves

MAE in mg/L; all rows use the unchanged GRU support basis.

| Model | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Context forest | 1.902823 | 1.866563 | 1.634818 | 1.596001 |
| Legacy direct | 1.806996 | 1.758826 | 1.631317 | 1.589387 |
| Unseen-neutral direct | 1.820109 | 1.776430 | 1.637231 | 1.576967 |
| Flow-window direct | 1.816609 | 1.770284 | 1.620974 | 1.572724 |
| Legacy integrated | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| Unseen-neutral integrated | 1.815765 | 1.771027 | 1.626570 | 1.570798 |
| Flow-window integrated | 1.811108 | 1.764629 | 1.610432 | 1.568986 |

The two new direct models improve their K5 point estimates by 0.78% and 1.05% relative to legacy direct, but their paired intervals cross zero. Integrated K5 performance is 0.37% worse for unseen-neutral and 0.26% worse for flow-window. The integrated contrast includes the source-validation-selected ecological mixture and support calibration, so a direct improvement does not guarantee an integrated improvement.

## Eight specified comparisons

Delta is new clock minus legacy; negative favors the new clock. Intervals use 5,000 paired whole-station bootstrap draws, with seed means within partition and equal partition weighting. Repeated station identities are sampled jointly across partitions.

| Clock | Stage | K | Delta MAE | 95% interval |
|---|---|---:|---:|---|
| Unseen-neutral | Direct | 0 | +0.013113 | [-0.006404, +0.032840] |
| Unseen-neutral | Direct | 5 | -0.012420 | [-0.037814, +0.006255] |
| Flow-window | Direct | 0 | +0.009614 | [-0.013205, +0.033315] |
| Flow-window | Direct | 5 | -0.016663 | [-0.041728, +0.002856] |
| Unseen-neutral | Integrated | 0 | +0.012227 | [-0.006296, +0.030415] |
| Unseen-neutral | Integrated | 5 | +0.005814 | [+0.000946, +0.010517] |
| Flow-window | Integrated | 0 | +0.007570 | [-0.012527, +0.028023] |
| Flow-window | Integrated | 5 | +0.004002 | [-0.001616, +0.009594] |

## High DOC and classification

All four K0 Q90 error contrasts favor legacy: new-clock MAE increases by 0.071–0.079 mg/L and every interval excludes zero. Integrated K5 Q90 MAE also increases: unseen-neutral +0.021230 [0.005404, 0.041212], flow-window +0.020420 [0.004840, 0.038314]. Direct K5 Q90 point estimates improve by about 0.094–0.098, but their intervals cross zero.

At K0, integrated Q90 recall declines from 64.064% for legacy to 62.637% unseen-neutral and 62.613% flow-window. The recall changes are -1.427 percentage points [-2.501, -0.543] and -1.451 [-2.464, -0.696]. False-positive rates also decline, from 2.157% to 2.055% and 2.072%; the latter interval includes zero. Fewer false high predictions therefore accompany poorer detection of genuinely high DOC, rather than establishing improved selectivity.

This tradeoff is consistent with stronger underprediction in the tail. Integrated K0 Q90 signed bias changes from -5.591957 mg/L to -5.751752 and -5.770497. At K5, the integrated biases are -4.965678, -5.035774 and -5.034667, respectively. K5 recall/FPR contrasts are all uncertain. Full ordinary-DOC errors, signed biases, precision and transformed errors remain in the CSVs and findings report.

## Heterogeneity and source-validation selection

Direct K5 improves in 7/9 seed-partition fits for each new clock, but most of the mean gain comes from partition 142: neutral deltas across partitions 142/143/144 are -0.034560, -0.006698, +0.003998; flow deltas are -0.044441, -0.009506, +0.003958. The top five positive station contributions supply 59.0% and 56.3% of positive station-gain mass.

Integrated K5 neutral worsens in all three partitions (+0.014138, +0.000509, +0.002796), with 3/9 fits improving. Flow improves only in partition 143 (+0.010301, -0.001001, +0.002706), with 4/9 fits improving. This does not support choosing a new clock globally or constructing a target-informed regional route.

Source-validation K0 already favors legacy: selected direct MAE is 1.804428 legacy, 1.822566 neutral and 1.818777 flow. Integrated K5 source-validation MAE is 1.604602, 1.606435 and 1.606664. New fits stop by epochs 86 and 94 under the 120-epoch ceiling; longer optimization is not indicated by an active cap. All training and calibration choices were made before target evaluation.

## What the neutral intervention actually did

The independent source-validation diagnostic replays the selected decay tensors on every valid step of every fixed validation query window, without reading DOC query labels. Its sampled means reproduce the runner diagnostics exactly. The result is stronger than merely silencing the 64 observation-age weights:

- All three seed44 models have gamma exactly one in every one of the 64 channels throughout all validation query windows. Their entire external decay operation is an identity on those inputs.
- For partitions 143 and 144 at seed44, all decay tensors are unchanged from initialization. The partition142 seed44 tensors moved by 0.001297, but still produce identity decay over the validation windows.
- In the other six models, 62 or 63 of 64 channels are identically one across all validation windows. Between 98.13% and 99.83% of valid channel-step values equal one; only one or two channels ever attenuate the recurrent state.
- Across the runner's fixed validation samples, mean external gamma is 0.8734 legacy, 0.99993 neutral and 0.9797 flow. The corresponding mean eleven-step products are 0.2930, 0.9993 and 0.8579.

Thus unseen-neutral nearly removes the extra shared-state decay in this cohort. The GRU's internal gates remain active. These products describe the explicit multiplicative decay factors, not complete end-to-end memory retention, hydraulic residence time or physical transport. Identity behavior is established for the inspected validation inputs, not every possible input.

This capacity change must accompany the interpretation: the neutral arm holds allocated parameter count fixed but changes which decay channels are effectively active. Its result does not isolate a pure semantic relabeling of observation age. The flow arm also changes the amount of attenuation substantially. The data do not establish one unique cause of the performance differences.

## Next implication and reproducibility

The chosen shared-state clock is not the main remaining performance bottleneck demonstrated by this experiment. Keep the successful legacy model and direct subsequent development toward high-DOC residual underprediction, rather than extending these clock fits or claiming longer effective storage automatically helps spatial transfer. The direct K5 behavior remains a useful conditional observation, not a new retained route.

These station partitions have been used in development, so all intervals are pointwise within the existing cohort. The executed runner is preserved in `code_snapshot`; subsequent whitespace/import wrapping in the working runner does not replace that execution record. Analysis bindings preserve the archived runtime and the source-validation-only capacity calculation alongside all favorable and unfavorable results.
