# DOC environmental, temporal and fusion comparison v1

This is an internal architecture development experiment. It extends the previous
spatial comparison while keeping its graph Transformer trunk, ST357 cohort,
source-only fitting, whole-HUC4 masks, source normalization, loss, optimiser,
epoch budget and observed query-cell endpoint. It does not alter paper endpoints
or old experimental products. The five geographical tasks have previously been
inspected; their test results are development evidence, not a fresh confirmation.

## Sequential design, fixed before fitting

Each outer target region uses three seeds, 42/43/44. Selection uses the mean
validation-region native DOC MAE over these seeds. Stage 1 fixes GRU and linear
concatenation; stage 2 fixes the environment choice from stage 1; stage 3 fixes
both choices. Choices can differ by outer region. Test outcomes never choose a
parent module or a final pipeline. Exact validation ties prefer the declared
incumbent, then candidate order. Checkpoint selection uses native validation MAE.

| Stage | Candidates | Eligible choices |
|---|---|---|
| Environment | learned global constant; linear 15→24; MLP 15→24→24; same-parameter residual MLP | linear, MLP, residual MLP |
| Time | current month MLP; ordered 12-month lag MLP; GRU; LSTM; causal single-layer temporal Transformer | lag MLP, GRU, LSTM, Transformer |
| Fusion | concatenation plus linear projection; extra residual projection; environment-conditioned temporal state; gated environment/time contributions | all four |

The constant and current-only arms diagnose information contribution. They are
excluded from selecting a model designed to use environmental background and
historical dynamics. If either diagnostic wins, report that the corresponding
information benefit was not established; do not hide that result.

Stage 2 reuses the stage-1 winning environment/GRU/concat fit. Stage 3 reuses
the stage-2 winning environment/time/concat fit. Thus 4 + 4 + 3 = 11 unique
configurations per seed and region, or 165 fits. All stage decisions and fit
hashes are saved before target DOC is accessed for scoring.

## Capacity and mechanism controls

All candidates preserve identical initial tensors in the common spatial trunk,
readout and base fusion projection. Alternative construction does not advance
the common training RNG. Actual parameter counts are reported: different
architectures cannot be called parameter-matched. The temporal lag MLP is kept
small to approximate GRU capacity; the Transformer has an extra temporal
attention block and dropout, so that comparison is between declared recipes.

Residual fusion and environmental conditioning each add exactly 600 parameters
and start with the baseline function. Conditioning applies scale and offset to
the final time representation, not the internal GRU gates. Gating starts with
equal contributions and learns relative environment/time weights. The residual
fusion control helps distinguish conditioning from added projection capacity.

Current-only time retains present masks, observation ages and season. Ages
contain historical observation timing, but this arm receives no older measured
temperature/discharge values. All models exclude DOC histories and station
identity embeddings. Source-only feature statistics are reused at inference.
The current-only arm uses an MLP of approximately GRU capacity. It compares a
current-input recipe with historical recipes; changing the operator means it
cannot causally isolate history from all architecture differences.

## Evaluation and reproducibility

Primary: native DOC MAE in mg/L, identical observed target cells and full query
coverage. Average errors across seeds rather than ensemble predictions. Report
basin/station macro MAE, RMSE, R² and source-Q90 tail MAE secondarily. Use paired
basin resampling for primary uncertainty; five basins give limited inference.
Do not treat three seeds or individual cells as independent geographical tasks.

Report conditional stage comparisons and the validation-selected cumulative
pipeline against a freshly trained MLP/GRU/concat reference. A pooled candidate
that wins the test table is descriptive evidence, not a test-selected pipeline.
Sequential comparisons do not exhaust all environment×time×fusion interactions.

The new protocol records data, node, mask, code and runtime hashes before fits.
Each fit stores its checkpoint, training trace, validation predictions and
provenance; target predictions are written after stage-selection freezing.
Completed packages are immutable, verified on resumption and independently
replayed during audit. Only observed query-cell predictions are archived, not
an all-reach or continuous space–time field. CPU training time under concurrent
workers is throughput evidence, not a controlled latency benchmark.
