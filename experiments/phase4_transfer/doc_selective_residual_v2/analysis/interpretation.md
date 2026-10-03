# Selective residual study: performance interpretation

Both the 60-epoch and 120-epoch panels were analyzed only after all nine packages of the extended panel were complete. Model, checkpoint, scale, support calibration, and ecological mixing choices used source validation. These results describe development on the same outer cohort; no new model was substituted into the fixed panel after target inspection.

## Main result

The selective-loss integrated model offers a small supported-sampling improvement over its matched tail-weighted control. It does not establish better zero-shot spatial transfer or a clear replacement for the retained ecological-affine model. Equal-station loss weighting worsens the current K=5 performance objective. Longer optimization alone does not resolve this tradeoff.

## Overall MAE (mg/L)

All entries use the same GRU support basis. K=0 reserves the same five support cells from evaluation, but consumes none of their labels. Direct models are the context forest plus native temporal residual; integrated models additionally use the frozen ecological memory with validation-selected mixing.

| Model | 60 cap K0 | 120 cap K0 | 60 cap K5 | 120 cap K5 |
|---|---:|---:|---:|---:|
| Context forest | 1.902823 | 1.902823 | 1.596001 | 1.596001 |
| tail2, direct | 1.823984 | 1.823984 | 1.594694 | 1.594694 |
| tail2, integrated | 1.810792 | 1.810792 | 1.588681 | 1.588681 |
| mae, direct | 1.821485 | 1.821473 | 1.581239 | 1.581539 |
| mae, integrated | 1.808602 | 1.808591 | 1.576456 | 1.576756 |
| selective, direct | 1.822768 | 1.823102 | 1.581764 | 1.581675 |
| selective, integrated | 1.813604 | 1.813938 | 1.574045 | 1.572667 |
| station, direct | 1.845401 | 1.844852 | 1.605161 | 1.609395 |
| station, integrated | 1.825026 | 1.824477 | 1.604613 | 1.604281 |
| Retained ecological affine | 1.809217 | 1.809217 | 1.579905 | 1.579905 |

## Matched-loss contrasts at the 120-epoch cap

Delta is candidate minus reference MAE; negative values favor the candidate. Brackets give pointwise 95% paired station-bootstrap intervals (5,000 draws). Cells are pooled within each seed, seeds averaged within partition, and the three partitions equally weighted. Station identities are resampled jointly across partitions. The overall query set contains 172 distinct stations and 10,520 distinct station-months; the 12,865 partition-cell occurrences and repeated training seeds do not create independent ecological samples.

| Candidate vs tail2 | K | Overall delta [CI] | Q90 delta [CI] | Non-tail delta [CI] | Better partitions |
|---|---:|---|---|---|---:|
| mae direct | 0 | -0.002511 [-0.043803, +0.035897] | +0.298802 [+0.183571, +0.426463] | -0.037284 [-0.084525, +0.002030] | 1/3 |
| mae direct | 5 | -0.013155 [-0.035042, +0.005299] | +0.011533 [-0.058882, +0.086740] | -0.016833 [-0.040511, +0.001225] | 2/3 |
| mae integrated | 0 | -0.002202 [-0.034872, +0.028947] | +0.261966 [+0.154096, +0.377233] | -0.032066 [-0.068381, -0.002229] | 1/3 |
| mae integrated | 5 | -0.011926 [-0.028970, +0.002452] | +0.026501 [-0.014464, +0.073479] | -0.016685 [-0.037040, -0.001212] | 2/3 |
| selective direct | 0 | -0.000882 [-0.043848, +0.037976] | +0.261093 [+0.175324, +0.358444] | -0.030797 [-0.076438, +0.010510] | 2/3 |
| selective direct | 5 | -0.013020 [-0.033249, +0.004309] | +0.054720 [+0.004638, +0.114949] | -0.021089 [-0.043956, -0.001802] | 2/3 |
| selective integrated | 0 | +0.003145 [-0.031115, +0.034783] | +0.227542 [+0.145777, +0.313602] | -0.021975 [-0.057858, +0.010841] | 2/3 |
| selective integrated | 5 | -0.016015 [-0.034476, -0.000360] | +0.027912 [-0.027886, +0.085584] | -0.021413 [-0.042557, -0.005135] | 3/3 |
| station direct | 0 | +0.020868 [-0.006328, +0.057471] | -0.003534 [-0.047583, +0.037666] | +0.023145 [-0.007156, +0.066347] | 0/3 |
| station direct | 5 | +0.014701 [+0.002800, +0.029081] | +0.037716 [-0.007360, +0.080347] | +0.012270 [-0.000213, +0.027814] | 0/3 |
| station integrated | 0 | +0.013685 [-0.012861, +0.049512] | +0.005767 [-0.035844, +0.047549] | +0.013869 [-0.015584, +0.055414] | 1/3 |
| station integrated | 5 | +0.015600 [+0.004977, +0.028728] | +0.034541 [-0.008724, +0.076039] | +0.013639 [+0.002824, +0.028025] | 0/3 |

## Selective integrated versus retained ecological affine

- K=0: MAE 1.813938 versus 1.809217; delta +0.004721 [−0.040551, +0.045592]. Partition deltas are −0.045768, +0.072872, and −0.012942. There is no established overall improvement.
- K=5: MAE 1.572667 versus 1.579905; delta −0.007238 [−0.018232, +0.001849], a 0.458% point reduction. All partition means improve (−0.013206, −0.006414, −0.002095); 7/9 seed-partition fits improve. The interval does not establish superiority over the retained reference.
- Against matched tail2 integrated at K=5, the reduction is 1.008% (95% relative-gain interval 0.0245–2.0933%). Its MAE delta is −0.016015 [−0.034476, −0.000360]. The gain is primarily ordinary-concentration error reduction: non-tail delta −0.021413 [−0.042557, −0.005135]; Q90 delta +0.027912 [−0.027886, +0.085584]. This narrow nominal interval should be read with its small effect size and the full fixed comparison panel.
- K=5 versus retained reference: Q90 delta +0.006449 [−0.010190, +0.030157]; non-tail delta −0.008554 [−0.022166, +0.002081]. This is not a newly solved high-DOC error problem.

## Detection and error tradeoff

Q90 is defined from source observations. Recall, precision and false-positive rate below are equal-partition summaries after seed averaging; precision is descriptive (no precision bootstrap interval was specified).

| Integrated model/reference | K | Q90 MAE | Non-tail MAE | Recall (%) | Precision (%) | False-positive rate (%) |
|---|---:|---:|---:|---:|---:|---:|
| tail2, integrated | 0 | 7.251297 | 1.187692 | 61.991 | 76.772 | 2.182 |
| tail2, integrated | 5 | 6.659295 | 1.004130 | 64.877 | 77.418 | 2.224 |
| mae, integrated | 0 | 7.513263 | 1.155625 | 58.605 | 78.892 | 1.844 |
| mae, integrated | 5 | 6.685797 | 0.987445 | 64.280 | 78.019 | 2.141 |
| selective, integrated | 0 | 7.478838 | 1.165716 | 60.346 | 78.300 | 1.955 |
| selective, integrated | 5 | 6.687207 | 0.982717 | 64.446 | 77.669 | 2.183 |
| station, integrated | 0 | 7.257064 | 1.201560 | 62.015 | 76.001 | 2.273 |
| station, integrated | 5 | 6.693837 | 1.017769 | 64.903 | 77.118 | 2.256 |
| Retained ecological affine | 0 | 7.431761 | 1.165287 | 60.076 | 77.998 | 1.989 |
| Retained ecological affine | 5 | 6.680758 | 0.991271 | 64.262 | 78.039 | 2.135 |

- At K=0, selective integrated versus tail2 lowers false positives by 0.227 percentage points [−0.396, −0.104], but reduces recall by 1.644 points [−3.224, −0.171] and increases Q90 MAE by 0.227542 [0.145777, 0.313602]. Its ordinary-concentration mean bias moves from +0.231482 to +0.006694 mg/L. Lower false-positive rate alone is therefore not proof of better high-DOC discrimination.
- Unweighted MAE shows the same broad tradeoff more strongly: integrated K=0 Q90 MAE rises by 0.261966 [0.154096, 0.377233], and recall drops by 3.385 points [−5.471, −1.759], relative to tail2.
- Selective integrated K=5 versus retained ecological affine has recall change +0.184 points [−0.308, +0.862], but false-positive rate increases by 0.048 points [+0.012, +0.102]. Its precision is 77.669%, versus 78.039%. A uniform improvement in detection is not established.
- Station balancing is not retained as a performance improvement: integrated K=5 delta versus tail2 is +0.015600 [0.004977, 0.028728], and versus ecological affine is +0.024376 [0.004942, 0.047415]. All three partitions worsen. At K=0 it does reduce tail error versus ecological affine, but raises non-tail error and false positives; it does not solve both regimes jointly.

## What extending 60 to 120 epochs changed

All 36 extended training traces reproduce their 60-cap prefixes exactly. The tail2 control reproduces the earlier encoder model and predictions exactly in both panels. Only partition 142 changes target predictions under the duration extension; partitions 143 and 144 remain identical.

- Selective integrated K=5 changes from 1.574045 to 1.572667: delta −0.001378 [−0.003134, +0.000113]. This is a 0.0876% point improvement, not an established duration benefit.
- Its K=0 MAE changes from 1.813604 to 1.813938: delta +0.000334 [−0.000697, +0.001445].
- Unweighted-MAE integrated K=5 changes by +0.000300 [−0.000567, +0.001360]; station direct K=5 changes by +0.004235 [−0.001765, +0.012338]. Every reported overall duration contrast includes zero.
- The 60-cap selective integrated-versus-tail2 K=5 interval already nearly excluded zero: delta −0.014636 [−0.032375, +0.000418]. Crossing the nominal significance boundary after the matched extension does not constitute a large new gain.

## Retention and scientific reading

1. Retain the existing ecological-affine model as the established overall reference. Keep the matched tail2 control to show the tail/ordinary tradeoff.
2. Retain selective integrated as a small-gain, support-available development candidate. Do not replace the zero-shot model or introduce test-selected K-specific deployment routing from this panel.
3. Stop advancing station-balanced loss and stop increasing training duration for this architecture on the basis of these results. The station-balancing hypothesis was plausible from source count concentration, but this experiment did not support its performance benefit.
4. The remaining bottleneck is assigning the correction to the right high-DOC stations/months without shifting ordinary concentrations. Global loss reweighting alone mainly trades those errors. Any next experiment should target conditional tail discrimination using source-validation diagnostics, rather than another unrestricted loss-weight sweep.
5. Integrated gains include ecological blending and support-adapter choices made separately on source validation; they cannot be attributed uniquely to the neural loss change. No causal explanation is established by these comparisons.

## Artifact checks

Both analyzer invocations completed with exit 0. The stable analyzer hash is `06b92bf37f98574085dd08799a707aa086fa9f89c550d1dce32c92f1da524685`. The 60-cap manifest binds 333 input records and 21 outputs; the 120-cap manifest binds 666 input records and 30 outputs. Every recorded hash was rechecked successfully. `analysis_binding_check.json` records these checks without modifying the bound manifests. All target comparisons, including negative contrasts, remain in the generated CSVs.
