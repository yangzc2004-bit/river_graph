# Daily hydrology: interpretation of the completed target analysis

All 27 fits and all nine packages were complete before this analysis. Source-validation diagnostics were examined first; target analysis then used the unchanged analyzer, fixed 45 comparisons, and 5,000 paired whole-station bootstrap draws. Checkpoint, residual scale, support adaptation and ecological blending were selected on source validation, never on these target results.

## Main finding

Within-month daily-flow information is worth retaining in the integrated residual pipeline. At K=5, the daily arm improves overall, high-DOC and ordinary-concentration MAE relative to both matched monthly and availability-only controls. Its overall improvement over the retained ecological-affine reference is smaller but consistently positive across partitions. The isolated direct residual gain is uncertain, and zero-support overall spatial-transfer improvement remains unresolved.

This is a gain from the completed prediction pipeline: daily inputs, the trained residual, and validation-selected ecological/support integration. It should not be attributed uniquely to the recurrent module or interpreted as a river-transport mechanism; these arms retain the empty-edge/self path.

## Main MAE and RMSE (mg/L)

| Model | K0 MAE | K5 MAE | K5 RMSE |
|---|---:|---:|---:|
| context_gru_tuned_anchor | 1.902823 | 1.596001 | 3.591068 |
| monthly_gru_tuned_anchor | 1.822772 | 1.590928 | 3.591736 |
| availability_gru_tuned_anchor | 1.821916 | 1.594748 | 3.612496 |
| daily_gru_tuned_anchor | 1.806996 | 1.589387 | 3.604470 |
| monthly_integrated_gru_tuned_anchor | 1.809588 | 1.588987 | 3.594587 |
| availability_integrated_gru_tuned_anchor | 1.808665 | 1.588716 | 3.594336 |
| daily_integrated_gru_tuned_anchor | 1.803538 | 1.564984 | 3.552260 |
| prior_ecological_affine_gru_tuned_anchor | 1.809217 | 1.579905 | 3.581369 |

## Fixed paired overall comparisons

Delta means daily minus its named reference; negative MAE favors daily. Intervals are pointwise 95% station-bootstrap intervals using seed means within each partition and equal weights for the three partitions. Repeated seeds predict the same ecological cells.

| Comparison | Delta MAE [95% CI] | Relative reduction (%) | Better partitions | Better seed-partition fits |
|---|---|---:|---:|---:|
| daily_vs_monthly_gru_tuned_anchor_k0 | -0.015776 [-0.060345, +0.022394] | 0.866 | 2/3 | 5/9 |
| daily_vs_availability_gru_tuned_anchor_k0 | -0.014920 [-0.058604, +0.022248] | 0.819 | 2/3 | 6/9 |
| daily_vs_monthly_integrated_gru_tuned_anchor_k0 | -0.006050 [-0.046083, +0.030308] | 0.334 | 2/3 | 5/9 |
| daily_vs_availability_integrated_gru_tuned_anchor_k0 | -0.005127 [-0.044469, +0.030195] | 0.283 | 2/3 | 6/9 |
| daily_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.005680 [-0.061615, +0.047049] | 0.314 | 2/3 | 4/9 |
| daily_vs_monthly_gru_tuned_anchor_k5 | -0.001542 [-0.024070, +0.021868] | 0.097 | 1/3 | 5/9 |
| daily_vs_availability_gru_tuned_anchor_k5 | -0.005361 [-0.025949, +0.014580] | 0.336 | 1/3 | 5/9 |
| daily_vs_monthly_integrated_gru_tuned_anchor_k5 | -0.024003 [-0.039741, -0.010160] | 1.511 | 3/3 | 9/9 |
| daily_vs_availability_integrated_gru_tuned_anchor_k5 | -0.023732 [-0.039241, -0.010066] | 1.494 | 3/3 | 9/9 |
| daily_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.014921 [-0.027809, -0.003073] | 0.944 | 3/3 | 8/9 |

- Integrated K5 versus monthly: 1.511% reduction, 95% relative-gain interval [0.664%, 2.331%]. All three partitions and all nine fits improve.
- Integrated K5 versus availability: 1.494% reduction, interval [0.657%, 2.301%]. All three partitions and all nine fits improve.
- Integrated K5 versus retained reference: 0.944% reduction, interval [0.194%, 1.711%]. All three partitions and eight of nine fits improve.
- The constant-only support path also improves after integration: versus monthly, delta −0.022258 [−0.037387, −0.008938]; versus retained reference, −0.018661 [−0.034275, −0.004879]. The result is not confined to the learned GRU support basis.
- Availability-only versus monthly yields no established overall benefit. Its K5 integrated delta is −0.000271 [−0.001270, +0.000550]. The full daily comparison is therefore not explained by an established availability-only improvement.
- Direct K5 daily versus monthly is only −0.001542 [−0.024070, +0.021868], with one of three partition means improving. Do not present the full integrated improvement as the isolated neural residual effect.

## High DOC, ordinary concentrations and detection

| Integrated model/reference | K | Q90 MAE | Ordinary MAE | Q90 bias | Ordinary bias | Recall (%) | Precision (%) | FPR (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| monthly_integrated_gru_tuned_anchor | 0 | 7.254986 | 1.185938 | -5.917423 | +0.246071 | 61.669 | 76.916 | 2.154 |
| monthly_integrated_gru_tuned_anchor | 5 | 6.660718 | 1.004322 | -5.090796 | +0.193223 | 64.845 | 77.440 | 2.221 |
| availability_integrated_gru_tuned_anchor | 0 | 7.256117 | 1.184843 | -5.931637 | +0.242557 | 61.835 | 77.001 | 2.150 |
| availability_integrated_gru_tuned_anchor | 5 | 6.659488 | 1.004160 | -5.088258 | +0.192708 | 64.877 | 77.418 | 2.224 |
| daily_integrated_gru_tuned_anchor | 0 | 7.179187 | 1.188732 | -5.591957 | +0.203547 | 64.064 | 77.636 | 2.157 |
| daily_integrated_gru_tuned_anchor | 5 | 6.586994 | 0.985640 | -4.965678 | +0.171136 | 65.134 | 77.807 | 2.188 |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.431761 | 1.165287 | -6.454223 | +0.196022 | 60.076 | 77.998 | 1.989 |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.680758 | 0.991271 | -5.132380 | +0.178223 | 64.262 | 78.039 | 2.135 |

At K=5 the integrated daily model improves both concentration regimes relative to its matched controls:

- Versus monthly: Q90 delta −0.073723 [−0.138865, −0.013241]; ordinary delta −0.018682 [−0.034521, −0.005131].
- Versus availability: Q90 delta −0.072494 [−0.136743, −0.012206]; ordinary delta −0.018520 [−0.033931, −0.005340].
- Versus retained ecological affine: Q90 delta −0.093764 [−0.130266, −0.058583], with all nine fits improving. Ordinary delta −0.005631 [−0.018801, +0.007698] remains uncertain. Tail mean bias improves from −5.132380 to −4.965678 mg/L; bias change +0.166702 [0.100080, 0.230496].

At K=0, daily information increases high-DOC sensitivity without an established FPR increase relative to matched monthly:

- Integrated recall change +2.396 percentage points [0.402, 4.890]; FPR change +0.0034 points [−0.1621, +0.1725].
- Direct recall change +2.807 points [0.568, 5.633]; FPR change +0.0513 points [−0.1100, +0.2246].
- Integrated Q90 native-MAE delta versus monthly is −0.075799 [−0.202078, +0.030906]; improved sensitivity does not establish an overall/native-tail MAE improvement at K=0.
- Versus retained reference, integrated K0 Q90 MAE is 7.179187 versus 7.431761 (3.399% reduction), with delta −0.252574 [−0.427841, −0.089908]. Recall increases by 3.988 points [1.668, 7.075]. Some of this advantage already exists in the updated monthly control, so it is not wholly incremental daily-feature gain.

At K=5, integrated recall changes are small and uncertain: +0.288 points [−0.460, 1.173] versus monthly, and +0.872 points [−0.119, 2.116] versus retained reference. FPR changes likewise span zero. Precision is descriptive; no precision bootstrap endpoint was added.

## Partition and station heterogeneity

- K5 integrated daily-minus-monthly partition deltas (142,143,144): −0.038693, −0.009383, −0.023933 mg/L.
- K5 integrated daily-minus-retained-reference partition deltas: −0.023721, −0.012723, −0.008320 mg/L.
- K0 integrated daily-minus-monthly partition deltas: −0.004960, +0.047268, −0.060457. The zero-support result does not generalize uniformly across partitions.
- Against monthly at K5, 105/172 distinct stations improve and 67 worsen; the five largest benefiting stations account for 38.2% of positive gain mass. Against the retained reference, 107 stations improve and 65 worsen. Partition consistency does not mean every station improves.

## Prespecified availability appendix

The three descriptor-validity flags are all one in 8,141 of 10,520 distinct target station-months, across 142 stations. The complementary group has 2,379 distinct cells across 57 stations; some stations appear in both groups. These are descriptive strata, not new bootstrap tests or model-selection rules.

| Model | K | All three valid MAE | Other cells MAE |
|---|---:|---:|---:|
| monthly_integrated_gru_tuned_anchor | 0 | 1.977595 | 1.155796 |
| monthly_integrated_gru_tuned_anchor | 5 | 1.746384 | 0.979177 |
| availability_integrated_gru_tuned_anchor | 0 | 1.976676 | 1.154841 |
| availability_integrated_gru_tuned_anchor | 5 | 1.745982 | 0.979391 |
| daily_integrated_gru_tuned_anchor | 0 | 1.957774 | 1.205614 |
| daily_integrated_gru_tuned_anchor | 5 | 1.716752 | 0.978589 |
| prior_ecological_affine_gru_tuned_anchor | 0 | 1.983006 | 1.129772 |
| prior_ecological_affine_gru_tuned_anchor | 5 | 1.736585 | 0.970315 |

The K5 benefit is concentrated where daily descriptors are available: integrated MAE changes from 1.746384 to 1.716752 versus monthly, while the complementary group is nearly unchanged (0.979177 to 0.978589). At K0 the valid group improves (1.977595 to 1.957774), but the complementary group worsens (1.155796 to 1.205614). These patterns are consistent with information-dependent value, not a proof that availability caused the difference; stations and DOC distributions differ between strata.

## Retention recommendation

1. Retain the daily integrated model as the leading candidate for subsequent support-assisted spatial-transfer work. Its K5 improvement is larger and more consistent than the preceding loss-reweighting change, and improves high-DOC and ordinary error together against matched controls.
2. Keep monthly, availability-only and the ecological-affine reference alongside it. Do not discard the zero-support counterexample in partition143 or the poor-daily-coverage subgroup.
3. Do not extend the epoch ceiling: all arms stopped by epoch77, below120. Strong source-validation improvement is a development signal, separate from the smaller measured target improvement.
4. Daily summaries currently enter the residual readout, while the recurrent history still uses the original monthly inputs. This study establishes the value of richer hydrological inputs in the integrated pipeline; it does not yet isolate a new recurrent-memory mechanism, graph advantage, or superiority over a forest supplied the same daily descriptors.
5. Current-month full daily summaries are appropriate for retrospective month-end reconstruction. They do not support a month-start forecast or a prediction before all within-month flow observations have occurred.

## Completion checks

All 45 fixed comparisons, both support paths, all K curves, unfavorable comparisons and source-validation choices are retained. The stable analyzer hash is `e4d5ae76cad4855b478468837b63258b42b318090538d5edd4085207cbc51066`. All 341 source records and 25 generated outputs match their bound hashes. The 72 carried-reference model/run comparisons reproduce exactly. No model, loss, threshold or execution source was changed during analysis.
