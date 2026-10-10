# DOC component comparison: experimental results

This is a sequential internal development study, not proof of a universally optimal architecture.

Completed: 165 unique fits, five whole-HUC4 regions, three seeds, 7,262 observed query cells at 104 stations; 100% identical query coverage in every candidate.

MAE is in mg/L. Seed errors are averaged; predictions are not ensembled. Parent modules and final pipelines were selected solely by source-validation MAE averaged over three seeds per outer region.

## Stage comparisons

| model_name                |     mae |   basin_macro_mae |    rmse |   q90_mae |   min_parameters |   max_parameters |
|:--------------------------|--------:|------------------:|--------:|----------:|-----------------:|-----------------:|
| environment__constant     | 3.6746  |           3.85226 | 6.695   |  12.9992  |            11605 |            11605 |
| environment__linear       | 2.17487 |           2.29474 | 4.60445 |   7.38995 |            11965 |            11965 |
| environment__mlp          | 2.1848  |           2.3033  | 4.70285 |   7.45009 |            12565 |            12565 |
| environment__residual_mlp | 2.23325 |           2.34576 | 4.63529 |   7.37161 |            12565 |            12565 |
| fusion__concat            | 2.19432 |           2.30449 | 4.57817 |   7.21482 |            11965 |            13165 |
| fusion__conditioned       | 2.19843 |           2.30234 | 4.60913 |   7.19676 |            12565 |            13765 |
| fusion__gated             | 2.19088 |           2.29995 | 4.59238 |   7.21458 |            12541 |            13741 |
| fusion__residual_concat   | 2.20338 |           2.31353 | 4.57135 |   7.24464 |            12565 |            13765 |
| selected__environment     | 2.2083  |           2.32637 | 4.71564 |   7.48198 |            11965 |            12565 |
| selected__fusion          | 2.19033 |           2.29792 | 4.57441 |   7.21965 |            12548 |            13741 |
| selected__reference       | 2.1848  |           2.3033  | 4.70285 |   7.45009 |            12565 |            12565 |
| selected__time            | 2.19432 |           2.30449 | 4.57817 |   7.21482 |            11965 |            13165 |
| time__current             | 2.23569 |           2.34727 | 4.70788 |   7.42872 |            12004 |            12604 |
| time__gru                 | 2.2083  |           2.32637 | 4.71564 |   7.48198 |            11965 |            12565 |
| time__lag_mlp             | 2.20276 |           2.30569 | 4.5769  |   7.20816 |            11948 |            12548 |
| time__lstm                | 2.24821 |           2.35103 | 4.65214 |   7.33936 |            12229 |            12829 |
| time__transformer         | 2.22544 |           2.32624 | 4.66729 |   7.34617 |            12565 |            13165 |

Positive gain means lower error. Basin intervals resample the five regions and are exploratory:

- environment__constant vs environment__mlp: -68.19% gain; 95% basin interval [-96.58, -27.11]%; improved in 0/5 regions.
- environment__linear vs environment__mlp: 0.45% gain; 95% basin interval [-1.50, 4.82]%; improved in 3/5 regions.
- environment__residual_mlp vs environment__mlp: -2.22% gain; 95% basin interval [-5.05, -0.31]%; improved in 1/5 regions.
- time__current vs time__gru: -1.24% gain; 95% basin interval [-8.59, 2.56]%; improved in 3/5 regions.
- time__lag_mlp vs time__gru: 0.25% gain; 95% basin interval [-3.63, 3.10]%; improved in 3/5 regions.
- time__lstm vs time__gru: -1.81% gain; 95% basin interval [-5.73, 0.75]%; improved in 2/5 regions.
- time__transformer vs time__gru: -0.78% gain; 95% basin interval [-4.86, 2.12]%; improved in 3/5 regions.
- fusion__residual_concat vs fusion__concat: -0.41% gain; 95% basin interval [-1.09, 0.90]%; improved in 2/5 regions.
- fusion__conditioned vs fusion__concat: -0.19% gain; 95% basin interval [-1.83, 1.04]%; improved in 3/5 regions.
- fusion__gated vs fusion__concat: 0.16% gain; 95% basin interval [-0.63, 1.01]%; improved in 3/5 regions.
- selected__environment vs selected__reference: -1.08% gain; 95% basin interval [-3.20, -0.06]%; improved in 0/5 regions.
- selected__time vs selected__reference: -0.44% gain; 95% basin interval [-4.15, 2.12]%; improved in 1/5 regions.
- selected__fusion vs selected__reference: -0.25% gain; 95% basin interval [-3.49, 2.19]%; improved in 1/5 regions.

## Validation choices

|   target_huc4 | stage       | choice       |   mean_validation_mae |
|--------------:|:------------|:-------------|----------------------:|
|          1013 | environment | mlp          |               1.41537 |
|          1013 | time        | lag_mlp      |               1.33236 |
|          1013 | fusion      | concat       |               1.33236 |
|          1019 | environment | linear       |               1.45972 |
|          1019 | time        | gru          |               1.45972 |
|          1019 | fusion      | conditioned  |               1.44744 |
|          0708 | environment | mlp          |               1.21331 |
|          0708 | time        | transformer  |               1.19656 |
|          0708 | fusion      | gated        |               1.18962 |
|          1030 | environment | residual_mlp |               1.02148 |
|          1030 | time        | gru          |               1.02148 |
|          1030 | fusion      | conditioned  |               1.00051 |
|          1101 | environment | residual_mlp |               5.6472  |
|          1101 | time        | gru          |               5.6472  |
|          1101 | fusion      | conditioned  |               5.62483 |

## Optimization budget

| model_name                |   median_best_epoch |   median_epochs_run |   max_epoch_selected_fits |
|:--------------------------|--------------------:|--------------------:|--------------------------:|
| environment__constant     |                   6 |                  16 |                         0 |
| environment__linear       |                  18 |                  28 |                         0 |
| environment__mlp          |                  19 |                  29 |                         0 |
| environment__residual_mlp |                  14 |                  24 |                         0 |
| fusion__concat            |                  23 |                  33 |                         0 |
| fusion__conditioned       |                  21 |                  31 |                         0 |
| fusion__gated             |                  25 |                  35 |                         0 |
| fusion__residual_concat   |                  21 |                  31 |                         0 |
| selected__environment     |                  21 |                  31 |                         0 |
| selected__fusion          |                  21 |                  31 |                         0 |
| selected__reference       |                  19 |                  29 |                         0 |
| selected__time            |                  23 |                  33 |                         0 |
| time__current             |                  13 |                  23 |                         0 |
| time__gru                 |                  21 |                  31 |                         0 |
| time__lag_mlp             |                  15 |                  25 |                         0 |
| time__lstm                |                  20 |                  30 |                         0 |
| time__transformer         |                  14 |                  24 |                         0 |

A checkpoint selected at the 60-epoch ceiling may benefit from a longer budget. No candidate receives extra epochs after its test outcomes are seen.

## Interpretation boundaries

- Environmental and time information ablations are reported even when they beat eligible models. Excluding them from the full-input selection was declared before fitting.
- Time comparisons are conditional on each region's validation-selected environmental module; fusion comparisons additionally condition on its selected time module. This does not explore every interaction.
- Actual capacity differs. Environmental conditioning and residual fusion share the same additional 600 parameters; conditioning modifies the final time representation rather than recurrent gates.
- The temporal Transformer includes an extra temporal attention block and dropout; its result compares this compact recipe with the other declared recipes, not all possible temporal Transformers.
- Current-only time uses a small MLP, whereas history candidates use other operators. This is an approximate-capacity recipe comparison, not a same-operator causal isolation of history.
- The previously inspected five regions share one major river basin. These data cannot demonstrate global transfer, performance on completely independent rivers or continuous all-reach reconstruction.
- Inference retains source stations as covariate-only context alongside the target graph. Target-only graph inference on an independent new river is not tested.
- Candidate comparisons are exploratory and multiple; percentile intervals have no multiplicity correction.
- Target hydrological covariates are available where observed, with masks and ages. Future prediction, a no-hydrology setting and a DOC-history assimilation setting are not tested.
- Availability differs between observed DOC query months and other months inside each station's query span. A good observed-query score does not directly establish accuracy in actual DOC gaps.
- Concurrent training time reflects early stopping and shared CPU load; it is not a controlled latency comparison.

## Audit

All fit/config/input/output hashes verified. Checkpoint replay: True; maximum native prediction difference 0. Independent source-grid arithmetic is recorded separately in independent_metrics_audit.json.

## Hydrological information availability

| population                         |   cells |   temperature_current_fraction |   discharge_current_fraction |   temperature_mean_observed_months_in_12 |   discharge_mean_observed_months_in_12 |   no_hydrology_in_12_months_fraction |
|:-----------------------------------|--------:|-------------------------------:|-----------------------------:|-----------------------------------------:|---------------------------------------:|-------------------------------------:|
| observed_query                     |    7262 |                       0.95621  |                     0.849766 |                                  9.15987 |                                9.93528 |                           0.00289177 |
| nonquery_within_station_query_span |   11076 |                       0.326201 |                     0.713886 |                                  5.20775 |                                8.55869 |                           0.119357   |

These diagnostics read covariate masks and query indices, not DOC values. They do not measure missing-cell accuracy or change any component selections. Lower input availability in non-query intervals is a concrete limitation of using observed-query errors to describe full reconstruction performance.

## Additional frozen input diagnostics

These fixed-checkpoint diagnostics were declared before pooled test scores. They do not change the original primary endpoint, any model choices, or the 165-fit training protocol.

Pipeline stress test: erase selected target channels for all months; source context remains:

| input_scenario                  |   selected__fusion |   selected__reference |
|:--------------------------------|-------------------:|----------------------:|
| available_covariates            |            2.19033 |               2.1848  |
| target_dynamic_hydrology_hidden |            2.36496 |               2.36451 |
| target_temperature_hidden       |            2.33378 |               2.30164 |

Time-recipe stress test: erase query-month current inputs, retaining non-query historical measurements. Erased months also disappear from later windows, and observation ages are recomputed:

| input_scenario                   |   time__current |   time__gru |   time__lag_mlp |   time__lstm |   time__transformer |
|:---------------------------------|----------------:|------------:|----------------:|-------------:|--------------------:|
| available_covariates             |         2.23569 |     2.2083  |         2.20276 |      2.24821 |             2.22544 |
| query_current_hydrology_hidden   |         2.32208 |     2.38837 |         2.31197 |      2.30939 |             2.4047  |
| query_current_temperature_hidden |         2.32069 |     2.31698 |         2.29418 |      2.31899 |             2.42638 |

The latter compares time candidates with the same validation-selected environmental module and concatenation per fold. Operator and capacity differ; it does not causally isolate memory alone. Both interventions are stress tests of observed DOC queries, not direct measurements of naturally missing DOC cells.

## Equal-shape serial inference

One CPU thread; 357 station nodes × 654 months; three passes after a warmup. One representative region's parent choices; feature assembly, graph building and I/O excluded. This is not an all-reach field.

| model_name                |   parameters |   median_seconds |
|:--------------------------|-------------:|-----------------:|
| environment__constant     |        11605 |         1.09603  |
| environment__linear       |        11965 |         1.06963  |
| environment__mlp          |        12565 |         1.0721   |
| environment__residual_mlp |        12565 |         1.06881  |
| selected__environment     |        12565 |         1.0721   |
| time__current             |        12604 |         0.835147 |
| time__lag_mlp             |        12548 |         0.82699  |
| time__gru                 |        12565 |         1.0721   |
| time__lstm                |        12829 |         1.12653  |
| time__transformer         |        13165 |         1.31334  |
| selected__time            |        12548 |         0.82699  |
| fusion__concat            |        12548 |         0.82699  |
| fusion__residual_concat   |        13148 |         0.829904 |
| fusion__conditioned       |        13148 |         0.834653 |
| fusion__gated             |        13124 |         0.841049 |
| selected__fusion          |        12548 |         0.82699  |
| selected__reference       |        12565 |         1.0721   |

## Final verification

Disposition: Share with caveats within the internal development scope. All 165 checkpoints replay validation and test predictions; primary and diagnostic MAE are independently recalculated against the original DOC grid. Parent paper protocols and endpoints remain unchanged.

Reproduction:

```bash
uv run python scripts/run_ladder.py --experiment doc-fusion-component-comparison-v1
uv run python scripts/analyze_doc_fusion_component_comparison_v1.py --replay
uv run python scripts/verify_doc_fusion_component_comparison_v1.py
uv run python scripts/audit_doc_fusion_input_availability_v1.py
uv run python scripts/evaluate_doc_fusion_input_robustness_v1.py
uv run python scripts/evaluate_doc_time_current_missing_v1.py
uv run python scripts/benchmark_doc_fusion_component_comparison_v1.py
uv run python scripts/finalize_doc_fusion_component_comparison_v1.py
```
