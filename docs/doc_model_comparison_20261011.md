# Consolidated DOC model comparison

Date: 2026-10-11. The environmental MLP + GRU + graph Transformer has the lowest spatial native MAE and log1p MAE point estimates among the tested methods. The hydrological/environmental GAM has the lowest spatial RMSE and highest predictive R². These are fixed-budget internal development results, not global SOTA or universal endpoint superiority.

The spatial columns use identical 7,262 observed queries at 104 whole-DOC-held-out stations in five Mississippi HUC4 regions. Native errors are mg C/L; R² is 1−SSE/SST. Scores are computed per seed then averaged, not from an ensemble. The proposed model's duplicate predictions were checked for exact equality across the two parent comparisons.

The time column separately uses 266 eligible stations and three masking seeds, with the same scored cores across 3/6/12-month gaps within each seed. It reports native MAE only; it must not be ranked against spatial scores. A dash denotes unavailable or inapplicable results: reference models have not run this temporal protocol; within-station time interpolation cannot support a station with no DOC. All methods from both rounds are included exactly once. Bold marks the best full-precision point estimate in each scored column; coverage is descriptive.

| Category | Method | Spatial MAE ↓ | Spatial log1p MAE ↓ | Spatial RMSE ↓ | Spatial R² ↑ | Intrinsic spatial coverage | Time MAE:3 / 6 / 12 months ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed | Environmental MLP + GRU + graph Transformer | **2.1839** | **0.3085** | 4.6991 | 0.4972 | 100.00% | **2.2807** / **2.3083** / **2.3329** |
| Reference | XGBoost with 12-month history | 2.2526 | 0.3245 | 4.6955 | 0.4980 | 100.00% | — / — / — |
| Reference | SSN covariance adaptation | 2.3265 | 0.3578 | 4.6762 | 0.5022 | 100.00% | — / — / — |
| Reference | Hydrological/environmental GAM adaptation | 2.4512 | 0.4096 | **4.5142** | **0.5361** | 100.00% | — / — / — |
| Reference/control | RF with 12-month history | 2.4637 | 0.3560 | 4.8091 | 0.4735 | 100.00% | — / — / — |
| Reference | Environmental/current-driver RF adaptation | 2.9912 | 0.3671 | 7.8566 | -0.4067 | 100.00% | — / — / — |
| Reference | PhyST-HC official architecture adaptation | 3.1652 | 0.4478 | 6.3018 | 0.0851 | 100.00% | — / — / — |
| Traditional | Bounded space-time IDW | 3.0079 | 0.4796 | 5.1318 | 0.4004 | 99.92% | 2.4095 / 2.5648 / 2.6145 |
| Traditional | Exponential ordinary kriging | 3.0433 | 0.4761 | 5.5570 | 0.2969 | 99.60% | 2.7504 / 2.7810 / 2.8136 |
| Traditional | Same-month Euclidean IDW | 3.0454 | 0.4838 | 5.2819 | 0.3648 | 99.86% | 2.8028 / 2.8499 / 2.8629 |
| Traditional | Exponential space-time ordinary kriging | 3.1120 | 0.4832 | 5.7340 | 0.2515 | 99.92% | 2.3359 / 2.5882 / 2.7820 |
| Traditional | Same-month NHD river-distance IDW | 3.1275 | 0.4922 | 5.4553 | 0.3224 | 99.86% | 2.8885 / 2.9284 / 2.9344 |
| Traditional | Station/source seasonal median | 3.5574 | 0.5475 | 6.8609 | -0.0717 | 100.00% | 2.4021 / 2.3852 / 2.3633 |
| Traditional | Same-month spatial nearest neighbour | 3.6069 | 0.5568 | 6.0572 | 0.1647 | 99.86% | 3.1486 / 3.1936 / 3.2517 |
| Traditional | Within-station linear time interpolation | — | — | — | — | — | 2.2917 / 2.4632 / 2.6166 |
| Traditional | Within-station PCHIP time interpolation | — | — | — | — | — | 2.3640 / 2.5419 / 2.7798 |

Every spatial score includes 100% final prediction coverage. Unsupported interpolation outputs use the declared training-only station/seasonal/source median fallback. Intrinsic coverage describes direct output before fallback. The independent common-covered-subset comparison in the interpolation report retained the fusion versus traditional spatial ranking.

The spatial MAE point estimate is 27.40% lower than bounded space-time IDW, 3.05% lower than historical XGBoost, and 6.13% lower than SSN. Existing exploratory five-basin confidence intervals for native-MAE differences against XGBoost, SSN and GAM include zero; they are unadjusted and use previously examined development regions. The small numerical advantages do not establish population significance. No new uncertainty estimate is inferred from this consolidation.

Temporal native MAE is lowest for fusion at all three tested gap lengths, but the advantage is only 0.48% over linear interpolation at three months and 1.29% over seasonal median at twelve months. At three months linear interpolation has lower log1p MAE/RMSE and higher R²; at twelve months seasonal median has lower log1p MAE and bounded space-time IDW has lower RMSE/higher R². Therefore temporal joint-endpoint superiority is not established.

SSN, GAM, ecological RF and PhyST-HC rows retain their adaptation labels. PhyST-HC executes the author's architecture with our inputs/target/training and is not a complete reproduction of the original process-state workflow. Input organization and budgets differ between families; this comparison alone cannot attribute gains to additional input sources or architectural components. Matched-input ablations and external-river evaluation address those separate claims. INCA-C is excluded because only its public native example, not this matched Mississippi comparison, has been executed.

Sources: [related-model comparison](doc_network_work_results_20261011.md), [traditional comparison](doc_interpolation_results_20261011.md), [combined metrics](../experiments/phase4_transfer/doc_model_comparison_v1/summary.csv), [matched-query aggregation checks](../experiments/phase4_transfer/doc_model_comparison_v1/aggregation_audit.json). The combined checks independently recomputed all four spatial metrics for both parent comparisons, reconciled matching masks/dataset/seeds, and verified identical query/truth membership for every spatial model and seed.
