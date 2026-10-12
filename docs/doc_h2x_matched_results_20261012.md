# Historical H2X under matched current DOC inputs

All 24 new H2X fits and the 96 immutable current-model/control packages completed. All 120 selected checkpoints were replayed independently on validation and test queries. The masks, query truth, normalization and local information budget are matched.

## Important identity of the H2X row

This is **H2X matched-input adaptation**, not a literal reproduction of the historical experiment. The released environmental encoder and two transport GCN layers are retained unmodified (hidden 64, environmental embedding 32). The current twelve-month driver history is flattened. Observed DOC inputs are absent; static edge gates read only the existing environmental features of their two endpoints instead of the old additional NHD edge attributes. No GRU, Transformer, new measurements or post-test tuning is introduced.

H2X has 41,317 parameters, versus 12,853 for original fusion and 17,617 for the explicit-history update. The optimizer, log loss, monthly batch schedule and 60-epoch maximum match the current fusion protocol. Native validation MAE alone chooses checkpoints. This tests the adapted architecture and capacity under the fixed information/budget contract, not one isolated component.

## Temporal reconstruction

| Model | mae | log1p_mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 2.1753 | 0.2397 | 5.7731 | 0.4904 |
| RiverDOCNet: original three-part fusion | 2.2214 | 0.2444 | 5.8936 | 0.4721 |
| RiverDOCNet: explicit-history/direct-readout fusion | 2.1677 | 0.2376 | 5.8128 | 0.4865 |
| History random forest | **2.1050** | **0.2271** | **5.6775** | **0.5081** |
| History XGBoost | 2.1387 | 0.2346 | 5.7343 | 0.4988 |

| Model | station_centered_r2 | log1p_station_centered_r2 | monthly_delta_mae | monthly_delta_correlation |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 0.0368 | 0.0906 | 2.1839 | 0.1587 |
| RiverDOCNet: original three-part fusion | 0.0103 | 0.0655 | 2.1666 | 0.0940 |
| RiverDOCNet: explicit-history/direct-readout fusion | 0.0043 | 0.0730 | 2.1904 | 0.0989 |
| History random forest | 0.0449 | 0.0711 | **2.1328** | 0.1778 |
| History XGBoost | **0.0666** | **0.0972** | 2.1459 | **0.2269** |

Overview scores are the equal mean across 3/6/12-month gaps and three fixed repetitions. The gap lengths reuse the same scored cores, so there are not nine independent ecological repetitions.

## 3-month gaps

| Model | mae | log1p_mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 2.1250 | 0.2322 | 5.6774 | 0.5086 |
| RiverDOCNet: original three-part fusion | 2.2049 | 0.2432 | 5.8627 | 0.4783 |
| RiverDOCNet: explicit-history/direct-readout fusion | 2.1498 | 0.2361 | 5.8092 | 0.4879 |
| History random forest | **2.0728** | **0.2243** | **5.6152** | **0.5190** |
| History XGBoost | 2.1099 | 0.2318 | 5.7688 | 0.4940 |

## 6-month gaps

| Model | mae | log1p_mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 2.1705 | 0.2392 | 5.7847 | 0.4882 |
| RiverDOCNet: original three-part fusion | 2.2128 | 0.2435 | 5.8667 | 0.4763 |
| RiverDOCNet: explicit-history/direct-readout fusion | 2.1825 | 0.2397 | 5.8421 | 0.4820 |
| History random forest | **2.1169** | **0.2281** | **5.6820** | **0.5078** |
| History XGBoost | 2.1429 | 0.2350 | 5.7708 | 0.4934 |

## 12-month gaps

| Model | mae | log1p_mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 2.2303 | 0.2478 | 5.8573 | 0.4746 |
| RiverDOCNet: original three-part fusion | 2.2465 | 0.2465 | 5.9515 | 0.4617 |
| RiverDOCNet: explicit-history/direct-readout fusion | 2.1708 | 0.2371 | 5.7872 | 0.4895 |
| History random forest | **2.1253** | **0.2290** | 5.7354 | 0.4974 |
| History XGBoost | 2.1634 | 0.2370 | **5.6634** | **0.5089** |

## Spatial reconstruction

| Model | mae | log1p_mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| H2X: matched-input adaptation | 2.2221 | 0.3269 | **4.5496** | **0.5285** |
| RiverDOCNet: original three-part fusion | **2.1958** | **0.3113** | 4.7403 | 0.4884 |
| RiverDOCNet: explicit-history/direct-readout fusion | 2.2038 | 0.3234 | 4.6078 | 0.5159 |
| History random forest | 2.4625 | 0.3558 | 4.8088 | 0.4735 |
| History XGBoost | 2.3157 | 0.3420 | 4.7135 | 0.4942 |

Spatial scores pool 7,262 observed queries at 104 fully held-out DOC stations across five HUC4 regions within seed, then average three seeds. All models have complete coverage on these queries.

## Point-estimate conclusions

Temporal native-MAE ranking: History random forest < History XGBoost < RiverDOCNet: explicit-history/direct-readout fusion < H2X: matched-input adaptation < RiverDOCNet: original three-part fusion.
Spatial native-MAE ranking: RiverDOCNet: original three-part fusion < RiverDOCNet: explicit-history/direct-readout fusion < H2X: matched-input adaptation < History XGBoost < History random forest.

Under the predeclared native/log-MAE primary endpoints, matched-input H2X does not displace the current fusion architecture or solve its temporal gap to the tree baselines. It improves temporal point MAE relative to original fusion, but has higher overall primary errors than the explicit-history update and both trees. It has lower native/log-MAE than the update at three and six months, but higher errors at twelve months. This is not a uniform advantage over gap lengths.

H2X has the lowest spatial RMSE and highest predictive R-squared in this five-model comparison, while original fusion retains the lowest spatial native/log-MAE. This metric tradeoff does not authorize switching the primary endpoint after seeing results. Overall H2X-versus-fusion primary difference intervals include zero in both tasks; H2X-versus-random-forest spatial primary intervals are below zero. Five spatial clusters and repeated development use limit generalization.

H2X temporal monthly-change MAE is 2.1839 mg/L, close to the zero-change reference (2.1840), and higher than original fusion (2.1666), random forest (2.1328) and XGBoost (2.1459). Its native station-centered R-squared is 0.0368, versus 0.0666 for XGBoost. Lower concentration error relative to original fusion therefore does not establish a comprehensive improvement in temporal change reconstruction.

- time, H2X versus RiverDOCNet: original three-part fusion: native MAE error reduction 2.08%; log-MAE error reduction 1.91% (negative means deterioration).
- time, H2X versus RiverDOCNet: explicit-history/direct-readout fusion: native MAE error reduction -0.35%; log-MAE error reduction -0.89% (negative means deterioration).
- time, H2X versus History random forest: native MAE error reduction -3.34%; log-MAE error reduction -5.54% (negative means deterioration).
- time, H2X versus History XGBoost: native MAE error reduction -1.71%; log-MAE error reduction -2.19% (negative means deterioration).
- space, H2X versus RiverDOCNet: original three-part fusion: native MAE error reduction -1.20%; log-MAE error reduction -5.01% (negative means deterioration).
- space, H2X versus RiverDOCNet: explicit-history/direct-readout fusion: native MAE error reduction -0.83%; log-MAE error reduction -1.08% (negative means deterioration).
- space, H2X versus History random forest: native MAE error reduction 9.76%; log-MAE error reduction 8.12% (negative means deterioration).
- space, H2X versus History XGBoost: native MAE error reduction 4.04%; log-MAE error reduction 4.39% (negative means deterioration).

## Paired uncertainty

| Task | Reference | Metric | H2X minus reference | Lower 95% | Upper 95% |
| --- | --- | --- | --- | --- | --- |
| space | RiverDOCNet: original three-part fusion | mae | 0.02630 | -0.11302 | 0.09904 |
| space | RiverDOCNet: original three-part fusion | log1p_mae | 0.01559 | -0.00214 | 0.02977 |
| space | RiverDOCNet: explicit-history/direct-readout fusion | mae | 0.01831 | -0.07338 | 0.10240 |
| space | RiverDOCNet: explicit-history/direct-readout fusion | log1p_mae | 0.00348 | -0.01243 | 0.01753 |
| space | History random forest | mae | -0.24042 | -0.55696 | -0.07551 |
| space | History random forest | log1p_mae | -0.02888 | -0.05353 | -0.00877 |
| space | History XGBoost | mae | -0.09365 | -0.34376 | 0.09918 |
| space | History XGBoost | log1p_mae | -0.01503 | -0.05066 | 0.01779 |
| time | RiverDOCNet: original three-part fusion | mae | -0.04617 | -0.15111 | 0.07410 |
| time | RiverDOCNet: original three-part fusion | log1p_mae | -0.00468 | -0.01519 | 0.00710 |
| time | RiverDOCNet: explicit-history/direct-readout fusion | mae | 0.00758 | -0.08915 | 0.11737 |
| time | RiverDOCNet: explicit-history/direct-readout fusion | log1p_mae | 0.00212 | -0.00714 | 0.01273 |
| time | History random forest | mae | 0.07026 | -0.00722 | 0.14774 |
| time | History random forest | log1p_mae | 0.01259 | 0.00381 | 0.02153 |
| time | History XGBoost | mae | 0.03654 | -0.01465 | 0.09616 |
| time | History XGBoost | log1p_mae | 0.00513 | -0.00113 | 0.01180 |

Intervals use 10,000 paired HUC4-block resamples with the same weights across models, seeds and gap lengths. Spatial inference has five clusters; temporal inference has 61 clusters. These are exploratory, unadjusted comparisons on repeatedly examined development tasks. An interval spanning zero does not establish a stable population lead, and a point-score win is not global SOTA.

The input adaptations, cohort, masking and training schedule prevent transferring the old mixed-571 ranking directly. This experiment cannot identify which historical protocol/input change caused any ranking reversal.

The target is the mean of available DOC samples within each calendar month. Observed-query evaluation does not validate natural missing-cell truth, continuous daily DOC, arbitrary river reaches or cross-river use.

Evidence: protocol.json, manifest.json, per-fit prediction sidecars, summary.csv, by_seed.csv, paired_bootstrap.csv and independent_audit.json in experiments/phase4_transfer/doc_h2x_matched_v1. All previous results are unchanged.
