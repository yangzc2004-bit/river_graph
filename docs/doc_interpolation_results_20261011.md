# Traditional DOC interpolation comparison

Date: 2026-10-11. Completed, independently audited internal Mississippi development experiment.

## Outcome and information boundary

Lowest full-population MAE point estimates: 12-month time gaps **Environmental MLP + GRU + graph Transformer**; whole-station geographical gaps **Environmental MLP + GRU + graph Transformer**. Rankings refer to these fixed recipes and populations, not globally optimized method families. Adverse results for the fusion model are retained.

Time tasks retrain the unchanged environmental MLP + GRU + graph Transformer nine times, on the exact visible DOC labels available to interpolation. The neural model does not explicitly read DOC context as a feature. Linear/PCHIP and space-time methods may use future visible DOC; this is retrospective smoothing. Hydrological neural inputs remain the twelve past/current months of temperature/discharge, masks, ages and season. Spatial tasks reuse the previously audited fifteen source-trained fusion models under the exact same five HUC4 masks: target stations have no DOC support; validation DOC never enters interpolator neighbours.

All fourteen validation selections were completed before any new target scoring. The primary endpoints remain native MAE and log1p MAE. Native concentration errors are mg C/L; R² is predictive 1−SSE/SST. Metrics are calculated separately per seed/mask then averaged, not from a prediction ensemble. Deterministic spatial predictions repeat under nominal seed rows and are not independent fits.

## Interpretation

The fusion model's spatial MAE is 2.1839, versus 3.0079 for the lowest-MAE traditional spatial recipe, bounded space-time IDW: a 27.40% point-estimate reduction. Fusion also has the lowest spatial log1p MAE/RMSE and highest spatial R² in this comparison; the ranking is retained on the common covered subset.

Temporal advantages are smaller and endpoint-dependent. At three months, fusion native MAE is 2.2807, versus linear interpolation 2.2917, only a 0.48% point-estimate reduction; linear interpolation has better log1p MAE, RMSE and R². At twelve months, fusion native MAE is 2.3329, versus seasonal median 2.3633, only a 1.29% reduction; seasonal median has lower log1p MAE, and bounded space-time IDW has better RMSE/R². These results support further evaluation of spatial reconstruction, but do not establish universal temporal or joint-endpoint superiority. No test-driven architecture or endpoint change is made here.

## Temporal population

| seed | test_queries | validation_queries | eligible_stations |
| --- | --- | --- | --- |
| 42 | 565 | 578 | 266 |
| 43 | 569 | 564 | 266 |
| 44 | 579 | 573 | 266 |

For each seed, the query/validation three-month cores are identical across 3/6/12-month gaps; larger gaps hide additional surrounding observations without changing the scored truth. Eligibility is based on dates, requires largest-gap brackets within twelve months outside the gap and at least six retained training observations, and excludes naturally unbracketed sparse stations. The counts are query occurrences across masks, not independent stations. There is no accuracy claim on naturally missing DOC cells.

## Temporal native MAE versus gap length

| Method | MAE:3 months | MAE:6 months | MAE:12 months |
| --- | --- | --- | --- |
| Environmental MLP + GRU + graph Transformer | 2.2807 | 2.3083 | 2.3329 |
| Station seasonal/source seasonal median | 2.4021 | 2.3852 | 2.3633 |
| Bounded space-time IDW | 2.4095 | 2.5648 | 2.6145 |
| Within-station linear time interpolation | 2.2917 | 2.4632 | 2.6166 |
| Within-station PCHIP | 2.3640 | 2.5419 | 2.7798 |
| Exponential space-time ordinary kriging | 2.3359 | 2.5882 | 2.7820 |
| Exponential ordinary kriging | 2.7504 | 2.7810 | 2.8136 |
| Same-month Euclidean IDW | 2.8028 | 2.8499 | 2.8629 |
| Same-month NHD river-distance IDW | 2.8885 | 2.9284 | 2.9344 |
| Same-month spatial nearest neighbour | 3.1486 | 3.1936 | 3.2517 |

## Full temporal metrics at twelve months

| Method | mae | log1p_mae | rmse | r2 | intrinsic_coverage |
| --- | --- | --- | --- | --- | --- |
| Environmental MLP + GRU + graph Transformer | 2.3329 | 0.2600 | 6.1618 | 0.4229 | 1.0000 |
| Station seasonal/source seasonal median | 2.3633 | 0.2565 | 6.1646 | 0.4097 | 1.0000 |
| Bounded space-time IDW | 2.6145 | 0.3174 | 5.9215 | 0.4663 | 1.0000 |
| Within-station linear time interpolation | 2.6166 | 0.2898 | 7.4166 | -0.0848 | 1.0000 |
| Within-station PCHIP | 2.7798 | 0.3038 | 8.1872 | -0.4270 | 1.0000 |
| Exponential space-time ordinary kriging | 2.7820 | 0.3320 | 6.1435 | 0.4208 | 1.0000 |
| Exponential ordinary kriging | 2.8136 | 0.3336 | 6.3780 | 0.3790 | 0.9907 |
| Same-month Euclidean IDW | 2.8629 | 0.3394 | 6.3311 | 0.3884 | 1.0000 |
| Same-month NHD river-distance IDW | 2.9344 | 0.3506 | 6.3942 | 0.3773 | 1.0000 |
| Same-month spatial nearest neighbour | 3.2517 | 0.3755 | 7.0536 | 0.2383 | 1.0000 |

## Full spatial population: 7,262 queries at 104 stations

| Method | mae | log1p_mae | rmse | r2 | intrinsic_coverage |
| --- | --- | --- | --- | --- | --- |
| Environmental MLP + GRU + graph Transformer | 2.1839 | 0.3085 | 4.6991 | 0.4972 | 1.0000 |
| Bounded space-time IDW | 3.0079 | 0.4796 | 5.1318 | 0.4004 | 0.9992 |
| Exponential ordinary kriging | 3.0433 | 0.4761 | 5.5570 | 0.2969 | 0.9960 |
| Same-month Euclidean IDW | 3.0454 | 0.4838 | 5.2819 | 0.3648 | 0.9986 |
| Exponential space-time ordinary kriging | 3.1120 | 0.4832 | 5.7340 | 0.2515 | 0.9992 |
| Same-month NHD river-distance IDW | 3.1275 | 0.4922 | 5.4553 | 0.3224 | 0.9986 |
| Station seasonal/source seasonal median | 3.5574 | 0.5475 | 6.8609 | -0.0717 | 1.0000 |
| Same-month spatial nearest neighbour | 3.6069 | 0.5568 | 6.0572 | 0.1647 | 0.9986 |

**Coverage meaning:** every full-population row has 100% final finite coverage. Intrinsic coverage is the fraction actually answered by the selected interpolator before fallback. Any unsupported prediction uses the declared training station-season median, station median, source-season median, then global median. Thus these are complete operational recipes with a declared fallback; a kriging+fallback row must not be called pure kriging accuracy over every cell. Seasonal median and fusion have full intrinsic coverage by construction.

Spatial raw time interpolation is excluded because target stations have no observed DOC. Same-month methods can lack donors in sparse historical months. Space-time methods use source observations within predefined 6/12-month windows. River IDW uses physical cached NHD mainstem/shared-downstream distance; Euclidean methods use Earth-centered chord km. Ordinary kriging uses a positive exponential covariance, with validation-selected range/nugget and up to30 neighbours. Space-time ordinary kriging uses up to20 observations and the Euclidean norm of scaled coordinates/time. Native outputs share the inherited clamp [0,expm1(15)]; raw values and interpolation/fallback indicators are saved.

## Raw-method comparison on the exact common covered subset

All methods in each track/seed are scored on the same intersection of intrinsic coverage. The following counts and scores exclude fallback and retain fusion on the identical cells. These populations differ from the full-population tables and must not be combined in one ranking.

| track | gap_months | Method | mae | log1p_mae | rmse | r2 | query_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| space | 0 | Environmental MLP + GRU + graph Transformer | 2.1729 | 0.3082 | 4.6789 | 0.4965 | 7233.0000 |
| space | 0 | Bounded space-time IDW | 2.9933 | 0.4789 | 5.1062 | 0.4004 | 7233.0000 |
| space | 0 | Exponential ordinary kriging | 3.0171 | 0.4739 | 5.5035 | 0.3035 | 7233.0000 |
| space | 0 | Same-month Euclidean IDW | 3.0320 | 0.4835 | 5.2505 | 0.3660 | 7233.0000 |
| space | 0 | Exponential space-time ordinary kriging | 3.0941 | 0.4821 | 5.7058 | 0.2513 | 7233.0000 |
| space | 0 | Same-month NHD river-distance IDW | 3.1146 | 0.4919 | 5.4259 | 0.3230 | 7233.0000 |
| space | 0 | Station seasonal/source seasonal median | 3.5333 | 0.5456 | 6.8223 | -0.0704 | 7233.0000 |
| space | 0 | Same-month spatial nearest neighbour | 3.5958 | 0.5568 | 6.0326 | 0.1631 | 7233.0000 |
| time | 3 | Environmental MLP + GRU + graph Transformer | 2.2644 | 0.2504 | 6.1315 | 0.4259 | 569.3333 |
| time | 3 | Within-station linear time interpolation | 2.2720 | 0.2494 | 5.6667 | 0.5099 | 569.3333 |
| time | 3 | Exponential space-time ordinary kriging | 2.3164 | 0.2575 | 5.9006 | 0.4659 | 569.3333 |
| time | 3 | Within-station PCHIP | 2.3436 | 0.2549 | 5.8664 | 0.4749 | 569.3333 |
| time | 3 | Station seasonal/source seasonal median | 2.3843 | 0.2544 | 6.3444 | 0.3775 | 569.3333 |
| time | 3 | Bounded space-time IDW | 2.3900 | 0.2728 | 5.9784 | 0.4522 | 569.3333 |
| time | 3 | Exponential ordinary kriging | 2.7338 | 0.3163 | 6.5404 | 0.3460 | 569.3333 |
| time | 3 | Same-month Euclidean IDW | 2.7674 | 0.3202 | 6.6031 | 0.3264 | 569.3333 |
| time | 3 | Same-month NHD river-distance IDW | 2.8543 | 0.3288 | 6.6577 | 0.3176 | 569.3333 |
| time | 3 | Same-month spatial nearest neighbour | 3.1140 | 0.3525 | 7.2523 | 0.1724 | 569.3333 |
| time | 6 | Environmental MLP + GRU + graph Transformer | 2.2948 | 0.2538 | 6.1512 | 0.4231 | 568.3333 |
| time | 6 | Station seasonal/source seasonal median | 2.3709 | 0.2558 | 6.2625 | 0.3979 | 568.3333 |
| time | 6 | Within-station linear time interpolation | 2.4501 | 0.2797 | 6.4246 | 0.3350 | 568.3333 |
| time | 6 | Within-station PCHIP | 2.5278 | 0.2889 | 6.2722 | 0.3881 | 568.3333 |
| time | 6 | Bounded space-time IDW | 2.5474 | 0.2963 | 6.1898 | 0.4127 | 568.3333 |
| time | 6 | Exponential space-time ordinary kriging | 2.5715 | 0.2935 | 6.2274 | 0.4062 | 568.3333 |
| time | 6 | Exponential ordinary kriging | 2.7686 | 0.3242 | 6.5022 | 0.3510 | 568.3333 |
| time | 6 | Same-month Euclidean IDW | 2.8180 | 0.3262 | 6.6996 | 0.3059 | 568.3333 |
| time | 6 | Same-month NHD river-distance IDW | 2.8975 | 0.3361 | 6.7395 | 0.2910 | 568.3333 |
| time | 6 | Same-month spatial nearest neighbour | 3.1632 | 0.3574 | 7.4375 | 0.1359 | 568.3333 |
| time | 12 | Environmental MLP + GRU + graph Transformer | 2.3173 | 0.2583 | 6.1579 | 0.4244 | 565.6667 |
| time | 12 | Station seasonal/source seasonal median | 2.3527 | 0.2552 | 6.1667 | 0.4098 | 565.6667 |
| time | 12 | Bounded space-time IDW | 2.5887 | 0.3152 | 5.9051 | 0.4701 | 565.6667 |
| time | 12 | Within-station linear time interpolation | 2.6104 | 0.2897 | 7.4289 | -0.0920 | 565.6667 |
| time | 12 | Exponential space-time ordinary kriging | 2.7472 | 0.3287 | 6.1110 | 0.4279 | 565.6667 |
| time | 12 | Within-station PCHIP | 2.7720 | 0.3037 | 8.2009 | -0.4381 | 565.6667 |
| time | 12 | Exponential ordinary kriging | 2.8071 | 0.3330 | 6.3811 | 0.3792 | 565.6667 |
| time | 12 | Same-month Euclidean IDW | 2.8205 | 0.3344 | 6.2707 | 0.4008 | 565.6667 |
| time | 12 | Same-month NHD river-distance IDW | 2.8919 | 0.3454 | 6.3341 | 0.3899 | 565.6667 |
| time | 12 | Same-month spatial nearest neighbour | 3.2145 | 0.3714 | 7.0084 | 0.2489 | 565.6667 |

## Verification and scope

Independent audit replayed all **363** candidate validation outputs and selected test predictions, including all nine neural checkpoints. Maximum validation replay difference: **0**. Independent full/common-subset metric reconciliation passed, maximum main-metric discrepancy **1.78e-15**. Aggregation independently matched all **38** summary rows with maximum discrepancy **1.78e-15**. Query truth, source-support separation, selected settings, immutable dependencies, model normalization, masks, hashes and explicit fallback coverage were checked.

Seed repetition alone does not establish population significance. These previously examined river-system data are development evidence; results do not demonstrate independent-river transfer, arbitrary virtual-site accuracy, or a fully observed continuous DOC field. The native interpolation/observation-assisted task is distinct from a future deployment without available DOC support.

V1 failed before neural training or target scoring on a shared metrics-interface mismatch. V2 corrected the frame contract, added an integration guard, re-froze unchanged tasks/models/budgets/endpoints and restarted all nine final fits. The failed v1 receipt is retained; no partial fit is mixed into these results.

The representative process-model feasibility investigation is reported separately in [INCA-C feasibility](doc_process_model_feasibility_20261011.md). A native supplied example ran successfully; no physical-model score on a different catchment is inserted here.

Evidence: [design](doc_interpolation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_interpolation_v2/protocol.json), [summary](../experiments/phase4_transfer/doc_interpolation_v2/analysis/summary.csv), [seed/mask metrics](../experiments/phase4_transfer/doc_interpolation_v2/analysis/by_seed.csv), [common-subset metrics](../experiments/phase4_transfer/doc_interpolation_v2/analysis/common_subset_by_seed.csv), [selected settings](../experiments/phase4_transfer/doc_interpolation_v2/analysis/selected_recipes.csv), [audit](../experiments/phase4_transfer/doc_interpolation_v2/analysis/independent_audit.json).
