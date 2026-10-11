# DOC reconstruction benchmark and evaluation metrics

Date: 11 October 2026. Status: evaluation design for the standalone environmental MLP, GRU and graph Transformer candidate. No new comparative fits or SOTA result accompany this plan.

The next comparison should establish whether the fixed fusion candidate improves on strong DOC predictors and applicable missing-data methods under the same permitted information. Native DOC MAE and log1p DOC MAE are the joint primary endpoints. The five HUC4 regions already examined during component selection remain development evidence. Confirmation requires new, reserved geographical tasks, with independent river systems needed for a claim about unseen rivers.

This plan belongs to the architecture branch described in [the component results](doc_fusion_component_results_20261011.md). It preserves [the existing paper endpoints](paper/primary_endpoints.json), paper charter, masks and result freezes. Its machine-readable companion is [the benchmark design](../configs/doc_fusion_benchmark_design_v1.json). Exact confirmation data, masks, implementation revisions and tuning budgets must be recorded in a new run freeze before fitting; this design alone is not an executable protocol.

## Related work and experimental comparators

Related work explains how the research differs from prior work; benchmark comparators supply predictions on the same hidden observations. An article's published score cannot substitute for a rerun on the common dataset. DOC, total organic carbon, biodegradable DOC and optical CDOM are different targets and must not share a numerical leaderboard.

| Comparison family | Relevant evidence | Role and eligibility |
|---|---|---|
| River DOC retrieval | [Aqua-OC, Tian et al. 2025](https://authors.library.caltech.edu/records/b74cg-58e81) combines Landsat, environmental inputs and optical water types; it evaluates RF, XGBoost, SVR and DNN and includes a Mississippi case study | Direct scientific comparator for reach-scale DOC reconstruction. RF and boosted trees with our permitted inputs are priority runnable baselines. These are not reproductions of the complete Aqua-OC framework. Its optical inputs require an explicitly separate, shared-input experiment |
| Regional DOC reconstruction | [Gong et al. 2026](https://www.sciencedirect.com/science/article/pii/S0957582026011201) uses RF, Landsat, watershed characteristics and water-quality variables for six Arctic rivers | Relevant current DOC study. The available publisher abstract does not establish that its inputs and splits match ours; chemistry access and code/data availability need checking before reproduction |
| Process-based river DOC | [ELM-MOSART-DOC, 2026](https://www.pnnl.gov/publications/elm-mosart-doc-large-scale-riverine-dissolved-organic-carbon-model-and-its-application) simulates DOC transport and transformation in headwaters and river networks | Scientific positioning and possible process-model comparison. Its long-term-average validation is not a station-month reconstruction score; a matched run requires its own forcing and calibration audit |
| Temporal imputation | [SAITS](https://arxiv.org/abs/2202.08516) uses attention for multivariate imputation; [CSDI](https://papers.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html) supplies conditional probabilistic imputations | Established temporal competitors. Audit conditioning variables, target DOC visibility, window direction and fixed variable dimensions before assigning them to a track |
| Graph imputation | [GRIN](https://arxiv.org/abs/2108.00298) uses recurrent graph message passing; [PriSTI](https://github.com/LMZZML/PriSTI) combines diffusion with spatiotemporal and geographical conditioning | Important spatial and temporal comparators. Use river relationships rather than silently replacing them with Euclidean distance; document any changes to the published methods |
| Attention and low-rank priors | [ImputeFormer, KDD 2024](https://arxiv.org/html/2312.01728v3) combines low-rank priors and attention | Strong within-network reconstruction comparator. Its original formulation assigns learned embeddings to series; changing missing rates in its zero-shot experiment does not demonstrate unseen-river transfer. Any inductive replacement must be labelled an adaptation |
| Recent spatial and temporal methods | [GSLI, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32081), [CoFILL, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0320.pdf), [CoSTI, 2025](https://arxiv.org/abs/2501.19364), and [STFF, 2026](https://www.sciencedirect.com/science/article/pii/S0925231226001505) extend graph learning, diffusion, consistency modelling or feature fusion | Review recent competitors before closing the roster. Reproduce at least one applicable recent method; disclose exclusions and adapter changes. A paper's own SOTA statement on traffic or air quality does not establish its ranking for DOC |

The first runnable comparison should include a source seasonal-median baseline, linear/ridge regression, RF, boosted trees, the environmental MLP + GRU without cross-node messages, and the fixed fusion candidate. Tabular competitors should receive the same permitted environmental variables, observation masks, ages, season and historical driver window; otherwise a weak current-only RF would not be a strong test of the fusion proposal. Existing RF/EcoRF artifacts use different feature protocols and cannot be inserted into this table without checking and rerunning their adapters.

The deep-method comparison follows input-eligibility checks for SAITS, GRIN and recent spatial and temporal methods. This is a targeted literature selection, not a complete survey or a declaration of the current best algorithm. Code availability and compatibility determine reproducibility, while exclusions must be recorded before inspecting their DOC test performance.

## Permitted information and evaluation tracks

The primary deployment track is **covariates only**: source DOC is supervision, while no DOC value enters the model as an input at source or target nodes. This matches the current candidate. Environmental and hydrological covariates, their missingness, and river topology are permitted. A target river supplies no DOC labels for fitting, normalization or model selection. All methods must be able to process target nodes without fitting new station-specific parameters.

A separate **observation-assisted reconstruction** track may expose a frozen support subset of DOC observations. Its hidden query is fixed, and every method has the same permitted support. Interpolation, kriging, matrix completion and native imputers can exploit that support. The current fusion model may be evaluated unchanged, but it does not consume DOC support. Results in this track must not be presented as zero-DOC deployment performance.

If a method needs satellite or laboratory inputs beyond the common contract, use a separately declared shared-input comparison or discuss it as related work. Never provide hidden query DOC, derived query statistics, target labels for normalization, or future observations outside the allowed window. Fixed-node models can remain within-network comparators even when they cannot support inductive new-river deployment.

The common reconstruction window ends in the target month and contains 12 months of available drivers. It includes current-month covariates and therefore represents retrospective reconstruction, not an advance forecast. Bidirectional processing within this bounded window is permissible; using months after the query is a separate smoothing task.

## Metrics and aggregation

Let `Q` contain only hidden cells with measured DOC, `y` denote concentration in mg/L, and `p` denote the prediction after inverse transformation. Apply a common nonnegative concentration rule before validation and test scoring, retaining the raw negative-prediction count. Declare any additional upper caps before fitting. Compute metrics independently for each fitted seed before aggregating. Missing predictions are not zeros and must not silently shrink `Q`.

| Metric | Definition | Purpose |
|---|---|---|
| Native MAE, joint primary | `mean_Q(abs(p - y))`, mg/L; lower is better | Interpretable concentration error; retains continuity with the component study |
| Log1p MAE, joint primary | `mean_Q(abs(log1p(p) - log1p(y)))`; concentrations first expressed numerically in mg/L | Complementary assessment with reduced leverage from very high concentrations; always report beside native MAE |
| RMSE, secondary | `sqrt(mean_Q((p - y)^2))`, mg/L | Sensitivity to large errors; not a substitute when primary MAE loses |
| R2, secondary | `1 - sum_Q((p-y)^2) / sum_Q((y-mean_Q(y))^2)` | Explained variation relative to the query mean; may be negative and is undefined when query variance is zero |
| Basin-macro MAE, secondary | Arithmetic mean of the native MAE of each held-out basin | Prevents densely sampled basins from concealing poor geographical transfer |
| High-DOC tail MAE, secondary | Native MAE for query truths above the 90th percentile of source training DOC in the outer fold | Assesses high-concentration performance using a training-defined threshold; report tail sample size |
| Prediction coverage, required | Finite predictions divided by the common eligible query count | Prevents a method that abstains on difficult cells from appearing artificially accurate |
| Cost, required | Parameter count, training cost, and serial inference latency on the same node/time grid and hardware | Assesses accuracy against practicality; record graph/input preparation separately |

Primary MAE is cell weighted within each task. Report every task and the explicitly pooled query result; give basin-macro MAE alongside it. Seeds repeat fits, not rivers. Averaging seed errors is different from scoring an ensemble of averaged predictions. Pooled R2 differs from a mean of per-task R2; both require an explicit label if both are shown.

Do not use MAPE as the headline metric because near-zero DOC makes percentage denominators unstable. A smooth map is not an accuracy metric. Spatial contrasts and seasonal patterns can be assessed on observed pairs or sufficiently sampled periods, with eligible counts, as secondary diagnostics. Naturally missing DOC cells have no known truth and cannot enter an error calculation. Prediction-interval coverage and CRPS become additional metrics only when a probabilistic product is actually constructed and validated; training-seed spread alone is not such a product.

## Missingness and geographical evaluation

Use distinct, declared scenarios: scattered hidden observations, contiguous temporal DOC gaps, completely hidden stations, and completely held-out basins. An independent river supplies the final transfer test. Artificial missing rates refer to the fraction removed from existing observed DOC cells, rather than the entire naturally sparse station-month grid. The original hydrological missingness remains visible to all methods.

The previously inspected five HUC4 tasks are development comparisons. Reserve new tasks before tuning benchmark adapters. New masks and confirmation artifacts belong in a new directory; preserve all existing paper and architecture-study freezes. Exact node populations, query indices, source/validation roles and hashes must be set before fitting. The confirmation river has not been selected by this design.

Include a driver-availability stress test: the component study found current temperature available in 95.6% of observed query months but only 32.6% of naturally missing DOC months within the evaluated station spans. Score observed queries under prespecified masked-input interventions. This tests robustness without claiming direct accuracy in unknown DOC gaps. Missingness interventions must be declared before test outcomes, with observation ages and later windows recomputed consistently.

## Rules for performance claims

Use identical query populations, units, permitted information, preprocessing boundaries and audited model-selection rules. Select hyperparameters by the lowest native validation MAE averaged over fitted seeds; resolve an exact tie by log1p validation MAE, then by a fixed configuration identifier. Both primary endpoints remain in the final test report. Tune each model family using source validation under a recorded comparable search budget; identical epoch counts are not a meaningful common budget for forests and diffusion models. Freeze final choices before evaluating confirmation queries. Any adapter needed for inductive or label-free operation is part of the reported method.

Report paired improvement against each declared comparator as `100 * (MAE_baseline - MAE_fusion) / MAE_baseline`. Use basin-clustered paired bootstrap intervals for transfer tasks, with station-clustered intervals for within-river station tasks. Keep training-seed spread separate. Five geographical clusters offer limited precision. A validation-selected primary challenger can support a prespecified pairwise test; simultaneous superiority claims over multiple methods require a declared multiplicity procedure before the new experiment.

A lowest point estimate establishes a leader in the declared table. Stronger wording requires paired uncertainty, agreement of the joint primary endpoints, full coverage, and performance on reserved geographical tasks. Divergence between native and log MAE is a tradeoff, not unconditional superiority. Preserve every primary result even when another metric is more favorable.

The appropriate claim, if supported, is improvement over the tested methods **under the specified DOC reconstruction benchmark and input conditions**. The 165 component fits establish neither that claim nor SOTA against related work. Results on one major basin cannot establish general unseen-river performance. A global SOTA statement would require a substantially broader, current and reproducible comparison.
