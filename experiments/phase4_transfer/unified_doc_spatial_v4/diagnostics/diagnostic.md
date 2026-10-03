# Where DOC reconstruction error remains

This is a **reused source-validation selection diagnostic**. It replays saved adapters on validation support/query cells using held-station fusion coefficients. These validation labels previously selected models, checkpoints and adapter settings. No outer-test query label or outer-test prediction table is used, and no model is fitted or selected here.

The panel contains nine runs, three station partitions and 7,197 distinct validation station-months at 140 distinct stations. Repeated seeds/partitions are not independent samples. Run metrics average cells, then seeds within partition, then the three partitions equally.

## Exact residual decomposition

For residual e = prediction − observation, MSE = cell-weighted mean(station mean(e)²) + cell-weighted mean((e − station mean(e))²). This is an exact accounting identity, not a statement about causes or irreducible error.

| Predictor | MAE (mg/L) | Raw MSE: station bias / within station | Log MSE: station bias / within station | Q90 share of raw squared / absolute error |
|---|---:|---:|---:|---:|
| Unadapted fusion (K=0) | 1.9805 | 19.3% / 80.7% | 44.8% / 55.2% | 95.7% / 42.6% |
| Constant correction (K=5) | 1.6567 | 14.3% / 85.7% | 13.3% / 86.7% | 96.5% / 44.9% |
| Updated GRU shape (K=5) | 1.6419 | 13.8% / 86.2% | 12.3% / 87.7% | 96.4% / 45.0% |
| Tree shape (K=5) | 1.6492 | 13.9% / 86.1% | 13.0% / 87.0% | 96.3% / 44.8% |

The remaining raw squared error in the updated-GRU adapter is dominated by **within-station temporal variation**. Its within-station fraction is 86.2% in raw space and 87.7% in log space. Source-training-Q90 cells make up 9.5% of validation queries but account for 96.4% of raw squared error (39.6% of log squared error). They contribute 45.0% of raw absolute error and 20.3% of log absolute error. Their signed mean residual is -8.149 mg/L (-0.463 in log1p space); negative means underprediction. Absolute-error shares are directly relevant to MAE, whereas squared-error shares give greater weight to extreme residuals.

## What adaptation changes on the selection set

| Candidate versus constant correction | Raw MAE reduction | Station-bias MSE reduction | Within-station MSE reduction |
|---|---:|---:|---:|
| Updated GRU shape (K=5) | 0.89% | 4.37% | 0.55% |
| Tree shape (K=5) | 0.45% | 4.10% | 1.03% |

Positive reductions favor the candidate. These differences describe the reused validation set; they do not replace the held-out comparison and do not justify another hyperparameter choice.

## Calendar pattern and observation coverage

A further descriptive split assigns 29.8% of total raw MSE to differences among each station's calendar-month residual means, and 56.4% to variation remaining within those groups. However, 13.7% of station×calendar-month groups are singletons (3.2% of cells). This in-sample grouping is descriptive; it cannot identify a learnable seasonal fraction or an irreducible ceiling.

Station diagnostics retain the five-support date span, represented calendar months and meteorological seasons (DJF/MAM/JJA/SON), nearest-support gaps, query/timeline hydro visibility and ecological novelty. Correlations average seeds within each partition/station and use Spearman ranks without significance tests.

| Covariate versus updated-GRU log MAE | Split 142 | Split 143 | Split 144 |
|---|---:|---:|---:|
| nearest support gap mean months | 0.14 | -0.22 | -0.23 |
| support span months | 0.08 | -0.28 | -0.16 |
| support unique seasons | -0.17 | -0.29 | -0.10 |
| hydro observed fraction | 0.16 | 0.14 | 0.23 |
| ecological novelty | 0.21 | 0.13 | 0.04 |
| n query | 0.01 | -0.29 | -0.32 |

These associations are descriptive and may reflect record length, station DOC variability or other differences; they do not establish that a missing covariate caused an error.

## Research implication

Prioritize the dominant component and its high-value records when reviewing the existing data: inspect timing and hydro coverage of large residuals, then assess whether the present monthly inputs represent those events. This diagnostic does not establish that a larger network, more observations or higher-frequency drivers would necessarily solve the remaining error. It provides a concrete target for the next scientific question while preserving the current model.

## Reproduction

Run `uv run python scripts/diagnose_unified_doc_spatial_v4.py`. `validation_queries.parquet` contains only source-validation queries. `run_decomposition.csv` and `decomposition.csv` retain exact masses/shares; `station_diagnostics.csv`, `paired_effects.csv` and `covariate_correlations.csv` retain the descriptive evidence. Source hashes are in `sources.json`.
