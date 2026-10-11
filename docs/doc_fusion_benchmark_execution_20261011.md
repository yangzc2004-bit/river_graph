# DOC fusion experimental execution plan

Date: 11 October 2026. This is the next execution stage of the [benchmark design](doc_fusion_benchmark_plan_v1_20261011.md), not a replacement for existing paper endpoints. The objective is to test the fixed environmental MLP + 12-month GRU + graph Transformer + concatenation candidate under sparse DOC supervision. A claim about a continuous river field needs evidence beyond the currently observed station-month evaluation grid.

Execution status: the first batch was frozen and started on 11 October 2026. Initial Ridge fits completed with model and prediction provenance; the worker proceeded to random-forest fitting. No new target ranking is available at this launch handoff. The full suite passed 1,537 tests with four explicit skips and eight pre-existing warnings; the subsequent freeze-dependent check passed all 18 selected tests. Ruff passed. Current progress is available through the status command below; the final completion marker and analysis receipt are written by the worker only after the full batch succeeds.

## Experiments and the questions they address

| Experiment | Design | Question and permitted conclusion |
|---|---|---|
| Matched-input baselines | Source seasonal median, Ridge, RF, histogram boosted trees, local neural control and fixed fusion; common query cells and permitted inputs | Is fusion better than practical strong predictors? This establishes a ranking among tested recipes, not SOTA |
| Missingness scenarios | Independently frozen scattered hidden observations, temporal blocks and wholly hidden stations | Does accuracy hold across missingness geometries? Artificial missingness is measured against observed DOC, never against all naturally missing cells |
| Component information controls | Environmental constant branch, current-only temporal branch and self-only spatial branch with otherwise fixed parents | Does each information package supply reproducible predictive gains? Earlier stage-selected comparisons cannot replace a fully matched ablation |
| Observed temporal and spatial structure | Seasonal centered profiles, within-station correlation/amplitude, consecutive-month increments and simultaneous connected-station contrasts | Does concentration accuracy also preserve observed variation? No complete curve or arbitrary reach ground truth is implied |
| Geographical transfer | Entire HUC4 target withheld from training and selection; subsequently an independent reserved river system | Can the model reconstruct without target DOC labels? Previously inspected internal and external regions are development evidence; fresh confirmation needs a reserved dataset |
| Covariate availability | Frozen deletion schedules for temperature/discharge, with masks and ages recalculated | How sensitive are predictions to incomplete dynamic inputs? A deletion stress test alone does not validate a cheap globally available forcing product |
| Published deep comparators | Eligibility review and documented shared-input adapters for temporal/graph imputers and at least one applicable recent method | Does fusion outperform reproducible related methods on this benchmark? Published scores on different data are not comparable; covariate-only and DOC-assisted tracks stay separate |

The work proceeds in stages. Only the first matched-input baseline batch and its structural diagnostics are launched by the present runner. Missingness masks, fully matched component controls, input-stress schedules and published deep adapters require their own run freeze before execution. Existing component and input-robustness studies remain relevant development evidence and are not silently rerun.

## First executable batch

Output directory: `experiments/phase4_transfer/doc_fusion_benchmark_development_v1/`.

Use ST357, the existing five HUC4 tasks {1013, 1019, 0708, 1030, 1101}, and seeds 42/43/44. The query union contains 7,262 observed station-month cells. These regions were inspected during architecture development and are not fresh confirmation. The earlier five-seed confirmation proposal remains a design; this three-seed batch deliberately reuses the exact completed development folds.

Tabular models receive 113 columns: 15 source-scaled environment descriptors, all 12 months of eight hydrological/mask/age/season channels, and two current-season fields. No DOC history, auxiliary chemistry or station embedding is allowed. The graph Transformer also aggregates available source/held-out-node covariates; the local neural control isolates that cross-node processing. All inference is retrospective because current-month drivers are available.

Ridge has three fixed target/regularization recipes and is deterministic: 15 unique fits across five folds. RF and histogram boosted trees each have three fixed recipes and three seeds: 45 fits per family. This gives **105 new unique candidate fits**, five source seasonal summaries, and **30 reused neural checkpoints** for the fixed fusion and local control. Each tabular family has three recipes; compute budgets differ by family and are disclosed. This is not an equal-compute exhaustive optimization study. No satellite-enabled Aqua-OC reproduction is claimed for the RF implementation.

All candidate fits use source training labels. Per fold, choose each family's recipe by mean source-validation native MAE; exact ties use log1p MAE and then a fixed recipe ID. Ridge is fitted once per recipe/fold and evaluated under the same three seed labels for table alignment; this creates no additional independent replicates. All five regional selection receipts are bound before any new target scoring. Reused neural checkpoints keep their original early-stopping decisions, fixed parents and provenance.

Native MAE and log1p MAE are joint primary. Secondary outputs include RMSE, predictive R², signed bias, station/basin macro MAE, and DOC-tail MAE above source-training Q90. Predictions share the inherited physical lower clamp and log-space numerical upper cap: `[0, expm1(15)]` mg/L. Raw clipping counts are recorded for new tabular models. Every reported model must cover all query cells. Calculate metrics separately per seed, then average errors rather than implicitly ensembling predictions. Report basin-macro and pooled metrics separately.

Structural diagnostics use the same hidden observed dates for predictions and truth:

- Seasonal centered calendar-month profiles require at least 24 observations and six occupied calendar-month bins per station. This is an observed-date seasonal diagnostic, not a verified complete annual cycle.
- Temporal correlation and amplitude ratios require at least 12 observations. Correlation is undefined for constant series; report eligible counts.
- Temporal increment errors use consecutive calendar months with both observations present.
- Spatial contrast errors require both direct-edge endpoints to have a query observation in the same month; empty eligible sets are reported as unavailable.

Paired uncertainty uses 5,000 basin-cluster bootstrap draws after averaging seed errors. Only five geographical clusters are available; intervals are exploratory and unadjusted. Do not make simultaneous statistical-superiority claims, declare a universal model, or replace primary endpoints with a favorable secondary metric.

## Execution and verification

The historical runner is included in older freezes, so it is not edited. `scripts/start_doc_fusion_benchmark_development_v1.py` prepares a versioned workspace under `cache/`, snapshots the new code, and starts its isolated `scripts/run_ladder.py` through `uv run`. The isolated dispatcher and the original dispatcher's hash are both recorded. The worker uses the existing managed environment and a copied source tree. New predictions carry config, data, mask, source and model hashes; changed completed packages are rejected on resume.

```bash
uv run python scripts/start_doc_fusion_benchmark_development_v1.py
uv run python scripts/status_doc_fusion_benchmark_development_v1.py
```

Tests cover independent metric examples, historical-window alignment, target-label poisoning, provenance refusal, whole-region separation, and preservation of old freezes. Every reused neural checkpoint is replayed against its saved query predictions. Final native MAE is independently recomputed from the original truth grid. The runner writes an analysis receipt and a batch-complete marker only after these checks pass.

No K-shot, Phase-2 or historical architecture runtime file is changed by this standalone branch. Their existing contracts remain in force; the new branch has its own freeze and contract test. No existing primary-endpoint document, mask or prediction is overwritten.
