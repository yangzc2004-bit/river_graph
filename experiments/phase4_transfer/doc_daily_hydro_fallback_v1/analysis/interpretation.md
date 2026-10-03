# Availability fallback: completed experiment interpretation

## Conclusion

The fixed availability router restores the monthly native prediction wherever all three daily descriptors are invalid. This improves the ordinary-concentration portion of K=0 target error, but does not establish an overall K=0 improvement over the daily pipeline. After ecological mixing and support adaptation, K=5 performance is slightly worse than the existing daily pipeline. Retain the router as a robustness variant; it does not replace daily integration as the main support-assisted performance candidate.

The route was proposed after the preceding daily experiment and before these nine new packages. It uses any valid numeric descriptor, not all three. No experts were retrained. Source validation alone fitted the downstream support and ecological choices; all fixed target contrasts were then reported without selecting a new route or model.

## Source-validation evidence came first

- Integrated K0 mean validation MAE: daily 1.804067, hybrid 1.803249. The gain was small.
- Integrated K5 mean validation MAE: daily 1.604602, hybrid 1.604714. These point estimates were essentially unchanged, with a small disadvantage for hybrid; no equivalence test was specified.
- Thus validation did not promise improved K5 performance. Target results below are separate evidence, not a reinterpretation of validation as held-out performance.

## Target performance (mg/L)

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE |
|---|---:|---:|---:|---:|---:|
| monthly_gru_tuned_anchor | 1.822772 | 1.786383 | 1.646213 | 1.590928 | 3.591736 |
| daily_gru_tuned_anchor | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 |
| hybrid_gru_tuned_anchor | 1.797611 | 1.752151 | 1.630645 | 1.588207 | 3.603739 |
| monthly_integrated_gru_tuned_anchor | 1.809588 | 1.777090 | 1.626636 | 1.588987 | 3.594587 |
| daily_integrated_gru_tuned_anchor | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 |
| hybrid_integrated_gru_tuned_anchor | 1.794386 | 1.748926 | 1.619420 | 1.571403 | 3.574318 |
| prior_ecological_affine_gru_tuned_anchor | 1.809217 | 1.791696 | 1.622073 | 1.579905 | 3.581369 |

## Fixed overall bootstrap contrasts

Negative delta favors hybrid. Pointwise 95% intervals use 5,000 paired whole-station draws, joint station identities across partitions, seed means within partition and equal partition weights. Neither repeated training seeds nor repeated partition appearances increase ecological sample size.

| Comparison | Delta MAE [95% CI] | Relative reduction (%) | Better partition means | Better seed-partition fits |
|---|---|---:|---:|---:|
| hybrid_vs_daily_gru_tuned_anchor_k0 | -0.009384 [-0.020678, +0.000780] | 0.519 | 3/3 | 7/9 |
| hybrid_vs_monthly_gru_tuned_anchor_k0 | -0.025161 [-0.068256, +0.012772] | 1.380 | 2/3 | 6/9 |
| hybrid_vs_prior_ecological_gru_tuned_anchor_k0 | -0.011606 [-0.070081, +0.043250] | 0.641 | 2/3 | 4/9 |
| hybrid_vs_daily_integrated_gru_tuned_anchor_k0 | -0.009152 [-0.020062, +0.000582] | 0.507 | 3/3 | 7/9 |
| hybrid_vs_monthly_integrated_gru_tuned_anchor_k0 | -0.015201 [-0.053796, +0.020253] | 0.840 | 2/3 | 5/9 |
| hybrid_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.014831 [-0.068989, +0.036685] | 0.820 | 2/3 | 4/9 |
| hybrid_vs_daily_gru_tuned_anchor_k5 | -0.001180 [-0.004019, +0.001855] | 0.074 | 2/3 | 6/9 |
| hybrid_vs_monthly_gru_tuned_anchor_k5 | -0.002721 [-0.025043, +0.020122] | 0.171 | 1/3 | 5/9 |
| hybrid_vs_prior_ecological_gru_tuned_anchor_k5 | +0.008301 [-0.017719, +0.036634] | -0.525 | 1/3 | 2/9 |
| hybrid_vs_daily_integrated_gru_tuned_anchor_k5 | +0.006419 [+0.000291, +0.014868] | -0.410 | 1/3 | 6/9 |
| hybrid_vs_monthly_integrated_gru_tuned_anchor_k5 | -0.017584 [-0.031265, -0.005455] | 1.107 | 3/3 | 9/9 |
| hybrid_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.008502 [-0.022937, +0.005623] | 0.538 | 3/3 | 7/9 |

Integrated K0 versus daily improves by 0.507% on the point estimate, with delta −0.009152 [−0.020062, +0.000582]. All three partition means improve, but the interval spans zero. Ordinary MAE improves by −0.010751 [−0.022409, −0.001230]; Q90 MAE changes by +0.002756 [−0.024249, +0.042046].

Integrated K5 versus daily worsens by 0.410%: delta +0.006419 [+0.000291, +0.014868]. Ordinary MAE worsens by +0.004385 [+0.000186, +0.009224]; Q90 MAE change +0.028119 [−0.016810, +0.083428] is uncertain. This does not erase the preceding daily-information result: hybrid K5 still improves over monthly by 1.107%, delta −0.017584 [−0.031265, −0.005455].

Against the old ecological-affine reference, hybrid integrated K5 has a smaller uncertain overall improvement (−0.008502 [−0.022937, +0.005623]), although Q90 MAE improves (−0.065645 [−0.131447, −0.008334]).

## What happens where daily descriptors are absent

Of 10,520 distinct target station-months, 8,141 have all three descriptors valid and 2,379 have none valid. There are no partially valid target cells. The any-valid versus all-valid distinction therefore has no empirical test in this target population; the empty partial stratum is retained as undefined.

| No-valid-descriptor cells, K0 | Monthly | Daily | Hybrid |
|---|---:|---:|---:|
| Native/direct MAE | 1.179763 | 1.222110 | 1.179763 |
| Integrated MAE | 1.155796 | 1.205614 | 1.164704 |

Native and direct K0 hybrid predictions equal monthly bitwise on these cells. The maximum absolute difference is exactly zero. Integration does not preserve this identity: the hybrid uses its separately selected global ecological mixture. Its mean absolute prediction difference from monthly on zero-valid cells is 0.026411 mg/L. This is expected downstream behavior, not a routing defect.

At K0, the all-valid group remains exactly the daily pipeline in this run: integrated MAE 1.957774. At K5, however, the all-valid integrated group worsens from 1.716752 to 1.724214 and the zero-valid group changes from 0.978589 to 0.979968. Positive-K support and mixture choices act beyond the cells at which the native route changes. These strata are descriptive and did not add new primary bootstrap tests.

## Why K5 can change after a correct native fallback

The net K5 integrated harm is concentrated in split142/seed44: target MAE changes from 1.505133 to 1.564444 (delta +0.059310). The three partition deltas are +0.019308, +0.001419, and −0.001470. Six of nine individual fits improve despite two of three partition means worsening.

That package also changes validation-selected gamma from 0.50 to 0.25, alpha from 0.75 to 0.50, and ridge from 1 to 10. Its own selected validation MAE changes only from 1.751567 to 1.750854. Other K5 integrated packages retain their parameter choices; all direct K5 adapters retain the daily alpha/ridge choices.

This correspondence points to sensitivity in downstream validation selection when candidate scores are close. It does not uniquely isolate which changed parameter causes the target error: native/support predictions also changed, and no additional counterfactual refit was performed. The unfavorable package remains in every result.

## Detection and tail behavior

| Integrated model | K | Q90 MAE | Ordinary MAE | Recall (%) | Precision (%) | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|
| monthly_integrated_gru_tuned_anchor | 0 | 7.254986 | 1.185938 | 61.669 | 76.916 | 2.154 |
| monthly_integrated_gru_tuned_anchor | 5 | 6.660718 | 1.004322 | 64.845 | 77.440 | 2.221 |
| daily_integrated_gru_tuned_anchor | 0 | 7.179187 | 1.188732 | 64.064 | 77.636 | 2.157 |
| daily_integrated_gru_tuned_anchor | 5 | 6.586994 | 0.985640 | 65.134 | 77.807 | 2.188 |
| hybrid_integrated_gru_tuned_anchor | 0 | 7.181943 | 1.177981 | 63.689 | 77.598 | 2.147 |
| hybrid_integrated_gru_tuned_anchor | 5 | 6.615113 | 0.990025 | 65.214 | 77.760 | 2.199 |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.431761 | 1.165287 | 60.076 | 77.998 | 1.989 |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.680758 | 0.991271 | 64.262 | 78.039 | 2.135 |

Hybrid-minus-daily integrated recall changes are −0.376 percentage points [−1.724, +0.350] at K0 and +0.080 points [−0.140, +0.321] at K5. Corresponding FPR changes are −0.0099 points [−0.0530, +0.0170] and +0.0109 points [−0.0176, +0.0409]. The fallback has not established an incremental detection benefit over daily. Precision remains descriptive; no precision interval was added.

## Candidate status and next implication

1. Keep daily integrated as the existing main support-assisted candidate. The fallback does not merit blanket promotion: source-validation K5 was nearly unchanged and target K5 worsens slightly.
2. Keep the fallback as an interpretable missing-hydrology robustness variant, with its exact native/direct K0 recovery property. Its K0 overall improvement is a point trend with an interval crossing zero, not a demonstrated universal improvement.
3. Do not splice hybrid K0 and daily K5 into a new test-selected winner. Both complete fixed pipelines and every K remain reported.
4. If further optimization is pursued, the observed issue motivates examining stability of source-validation support/mixing selection. The correct fixed route alone does not guarantee stable downstream adaptation. Any change should be specified from source-validation evidence rather than tailored to split142/seed44 target outcomes.

## Completion and bindings

No new backbone was trained, and no previous experiment files were changed. All 20 models and 36 fixed contrasts are present. All 144 copied-control/reference comparisons reproduce exactly; monthly/daily adapters and mixers also match their parent serialized states. All 332 source records and 28 outputs match the analysis manifest. The analyzer hash is `30a0aeb8fd5e5220952c19920e034f4395cad89b47df3f32ed567241899421ce`. The binding check is saved independently of the unchanged generated manifest.
