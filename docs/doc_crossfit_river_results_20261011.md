# DOC whole-station cross-fitting and matched relationship controls

Protocol `9ac26c78c01cec6df9895030f41ce0cda4ebd78ed58335508180a440dee29a19`. Nine fixed packages: original source-station partitions 142/143/144 by seeds 42/43/44. All 54 local fits and 54 correction fits completed and bound before evaluation. Checkpoints were replayed with zero numerical difference. 20259 real/negative-control matching records were independently verified, including every source fold, against uncapped directed paths. All six correction heads have identical seeded initial tensors and parameter counts. Full query coverage is 100%; the support column denotes the subset with a usable matched message.

This experiment isolates two correction-training strategies on a **new, clean common backbone**: covariate-only ecological ExtraTrees plus MLP/12-month GRU, then environmental-state attention. It does not reproduce the entire historical observed-DOC/retrieval chain. No historical supervised weights, chemistry, DOC-valued inputs or station embeddings are used. Input data and outer partitions match the selected branch; the absolute errors must not be described as a direct upgrade to a different historical pipeline.

## Findings and research decision

**Do not adopt whole-station crossfit correction training in this controlled backbone/budget.** On test stations, its native/log1p errors worsen versus fitted-anchor training by 6.19%/7.17% for directed attention, 4.74%/5.41% for generic attention and 15.85%/14.01% for nonconnected controls. Each comparison worsens in 8 of 9 paired runs on both primary errors, and all three split averages worsen. The source-anchor error mismatch is measurable, but this attempted remedy does not improve deployment performance. The validation population has the same average ordering between training strategies.

**There is no demonstrated unique directed-river gain.** With fitted-anchor training, directed attention has higher native/log1p errors than generic attention (0.56%/0.57%) and nonconnected donors (0.62%/0.86%); their supplementary station-resampling intervals include zero. With crossfit training, real paths outperform nonconnected donors (7.77% native, 5.20% log1p), but trail generic attention (1.94% native, 2.25% log1p). Thus this conditional real-versus-null advantage does not establish superiority over generic spatial attention.

The numerically best primary row is fitted/nonconnected, not a genuine-river model. Its improvement over the local baseline is only 0.54% native and 0.26% log1p; the exploratory paired native difference is -0.0102 mg/L with station-resampling interval [-0.0335, 0.0106], and the log1p interval also includes zero. Do not promote that tiny result into a main model claim. The generic fitted row has the lowest Q90 error; secondary metrics do not override the jointly fixed primary errors. Matched usable-message coverage averages 86.75% on test and 89.58% on validation; output coverage is 100% of the identical observed-cell queries.

Keep the previously released model unchanged. The next focused diagnosis should examine compatibility between crossfit local latent states/anchors during correction training and full-source states/anchors at deployment, plus limits on correction magnitude. Representation mismatch is a **hypothesis**, not a mechanism established by these six comparisons. Re-tuning against these test results would require a prospectively new protocol/evaluation; this report retains the negative findings and does not re-select epochs or endpoints.

The fitted strategy trains corrections on the full-source local predictor's source predictions and states. Crossfit uses five local models trained without any receiving-fold DOC, including all supervised encoders; both its anchor and latent state are fold isolated. At deployment every correction uses the **same full-source local model**, with no fold ensemble. Source-anchor training MAE averaged 1.0725 for fitted vs 1.8307 for whole-station-held-out predictions; this quantifies the distribution mismatch, not test performance.

## Outer-test stations excluded from this experiment's training and selection

| Training | Relationship | MAE | log1p MAE | RMSE | R2 | Q90 MAE |
|---|---|---:|---:|---:|---:|---:|
| Local only | No message | 1.8912 | 0.2767 | 3.9968 | 0.4767 | 7.4174 |
| fitted | directed | 1.8928 | 0.2783 | 3.9607 | 0.4865 | 7.2793 |
| fitted | generic | 1.8823 | 0.2767 | 3.9521 | 0.4887 | **7.2620** |
| fitted | nonconnected | **1.8810** | **0.2759** | **3.9505** | **0.4890** | 7.2848 |
| crossfit | directed | 2.0098 | 0.2983 | 4.1402 | 0.4386 | 7.6654 |
| crossfit | generic | 1.9715 | 0.2917 | 4.0746 | 0.4558 | 7.5975 |
| crossfit | nonconnected | 2.1793 | 0.3146 | 4.2923 | 0.3929 | 7.8250 |

All numbers are equal means of nine complete per-run metrics, including per-run R2; no seed ensemble. MAE/RMSE/Q90 MAE have the dataset's DOC concentration units (mg/L); log1p MAE and R2 are dimensionless. Q90 uses each run's source-training threshold. Independent observed-cell queries only, not scoring an unobserved continuous field.

## Source-validation stations: diagnostic evaluation

| Training | Relationship | MAE | log1p MAE | RMSE | R2 | Q90 MAE |
|---|---|---:|---:|---:|---:|---:|
| Local only | No message | 1.9743 | 0.2944 | 6.5884 | 0.3235 | 9.8480 |
| fitted | directed | 1.9890 | 0.2965 | 6.5702 | 0.3276 | 9.7400 |
| fitted | generic | 1.9771 | 0.2950 | 6.5638 | 0.3292 | 9.7454 |
| fitted | nonconnected | **1.9460** | **0.2915** | **6.5343** | **0.3364** | **9.6724** |
| crossfit | directed | 2.0538 | 0.3087 | 6.6571 | 0.3100 | 10.1074 |
| crossfit | generic | 2.0380 | 0.3070 | 6.6247 | 0.3150 | 10.0456 |
| crossfit | nonconnected | 2.1214 | 0.3180 | 6.7149 | 0.2906 | 10.1757 |

These labels selected no epoch or arm. Every neural component uses the pre-fixed final epoch (30). The outer validation query population is retained from the original branch, but the local predictor is freshly fitted; historical development metrics belong to their own pipeline.

## Paired effects, support and limits

`paired_comparisons.csv` contains crossfit-minus-fitted within all three relationships, genuine-minus-control within each training strategy, and every correction-minus-local effect. Negative difference means lower error. `training_relationship_interactions.csv` reports whether crossfit changes the directed relationship advantage. `support_strata.csv` retains all predictions and separates cells with/without a common usable candidate. Unsupported predictions reproduce the shared local baseline exactly; no model is evaluated only on supported cells.

`station_resampling_intervals.csv` provides supplementary 5000-draw paired station-resampling intervals for both primary errors. Repeated split/seed occurrences of a station travel together; run means are recomputed within each draw. These are conditional exploratory intervals for the fixed models and masks. They do not propagate refitting uncertainty, station-network dependence, or establish population-level significance. Three overlapping partitions and three seeds are not nine independent rivers. Inspect paired run signs and split-average ranges as well.

Real attention uses upstream and downstream paths as statistical context, including 0/1/3-month lagged environmental states. Downstream context does not imply backward physical DOC transport. Generic attention remains learned but has no direction/path input; donor count and date support are matched. Nonconnected attention keeps pseudo-path slot attributes while replacing donors with unrelated matched nodes; those copied attributes are a negative-control device, not asserted physical links. Generic sparse attention is not a dense all-node Transformer. The known 357-node atlas supplies only covariates and paths outside training roles; this is internal K0 transfer, not a genuinely unseen river or virtual-site validation.

Native and log1p MAE are jointly primary. Improvements in one error, RMSE or R2 alone do not justify an overall superiority claim. River-specific benefit requires consistent directed advantages over **both** generic and nonconnected controls. All six outcomes are retained. No released architecture, paper endpoint or default was changed based on this experiment.

Evidence: `experiments/phase4_transfer/doc_crossfit_river_v2/` protocol, manifest, scoring gate, run completion records, prediction/provenance pairs, independent audit and analysis CSVs. Runnable entry: `uv run python scripts/start_doc_crossfit_river_v2.py`; audit: `uv run python scripts/verify_doc_crossfit_river_v3.py`; supplementary report: `uv run python scripts/analyze_doc_crossfit_river_v3.py`.

## Execution repairs and verification

V1 is retained as an aborted pre-scoring archive: 48 local checkpoints, zero river correction checkpoints and zero scored prediction products. A missing optional station-storage attribute was replaced by complete label-free hydro availability in V2; all 63 source/evaluation relation banks passed finite-feature preflight. Every V2 fit is fresh. The original frozen V2 audit script's pandas local-scope lookup failed before aggregation; V3 uses explicit boolean row selection and records its own hash. Prediction files, trained weights, source snapshots, endpoints and formulas were not changed by this audit-only repair. The supplementary V3 report includes all correction-versus-local intervals as well as the prospective factorial comparisons.

Full suite: 1603 passed, 3 skipped with explicit local-data conditions. Ruff passed. All 126 model/role/run metric populations checked; all saved predictions replayed exactly; all 20,259 source/evaluation matching records checked against uncapped mapped paths. Train/validation/test stations are disjoint at 232/54/71 per partition. No prior paper charter, endpoint, mask or released model was modified.
