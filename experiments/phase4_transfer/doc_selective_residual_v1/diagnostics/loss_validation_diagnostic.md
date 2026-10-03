# Source-validation comparison of four DOC training objectives

This diagnostic reads only bound model JSON summaries/traces and run configurations.
It does not read outer-query predictions, labels, or evaluation tables. All 36 completed
fits use the same last-self/ecology encoder, concentration head, source OOF context inputs,
60-epoch ceiling, patience 5, native output and fixed source-validation query set.

Checkpoint and residual scale are selected by unweighted pooled native validation MAE
for every objective. These are reused selection-set results, not independent transfer
estimates. No objective, coefficient or training budget is chosen by this diagnostic.

## What changes across objectives

- tail2: uniform cell sampling; ordinary error weight 1, source-Q90 tail weight 2.
- mae: uniform cell sampling and unweighted absolute error.
- selective: tail2 plus 0.5 times positive prediction error on source observations below Q90.
- station: tail2 weights divided by their source station's total tail-weighted mass; each station contributes one unit.

All arms shuffle every observed source cell once per epoch and divide batch-mean loss
by one fixed full-source mean weight. Station weighting changes the training population
being averaged, not the sampler. The selective term changes the cost of ordinary
overprediction; it is not an inference-time gate. Training-objective values therefore
must not be compared as if they were the same risk. Validation retains cell-weighted MAE.

## Selected validation performance and duration

| Loss | Validation MAE | Context MAE | Selected epochs | Epochs run | Cap-limited fits | Scale counts |
|---|---:|---:|---:|---:|---:|---|
| tail2 | 1.869505 | 1.990984 | 5–49 | 10–54 | 0/9 | 0.5: 2, 1: 7 |
| mae | 1.882749 | 1.990984 | 2–60 | 7–60 | 2/9 | 1: 9 |
| selective | 1.854275 | 1.990984 | 19–57 | 24–60 | 1/9 | 0.5: 2, 1: 7 |
| station | 1.885577 | 1.990984 | 0–60 | 5–60 | 2/9 | 0: 1, 0.25: 2, 1: 6 |

Each displayed MAE averages seeds within partition and then the three partitions equally;
the balanced nine-run design makes this identical to their arithmetic mean. Within each run,
the underlying validation selection metric pools query cells. No inferential CI is attached
to these reused selection values.

## Paired validation differences from tail2

| Loss | Partition | Delta MAE | Relative reduction | Improved seeds |
|---|---:|---:|---:|---:|
| mae | 142 | +0.039271 | -2.025% | 0/3 |
| mae | 143 | -0.015220 | +0.821% | 2/3 |
| mae | 144 | +0.015683 | -0.864% | 0/3 |
| selective | 142 | +0.016891 | -0.871% | 0/3 |
| selective | 143 | -0.090840 | +4.901% | 3/3 |
| selective | 144 | +0.028260 | -1.556% | 0/3 |
| station | 142 | -0.010127 | +0.522% | 1/3 |
| station | 143 | +0.051051 | -2.754% | 0/3 |
| station | 144 | +0.007293 | -0.402% | 1/3 |

Negative Delta MAE favors the alternative objective. The CSV retains all nine paired
runs, checkpoint/scale changes and every recorded validation candidate choice.

## Duration and numerical stability

- 5/36 fits ended at the 60-epoch ceiling before exhausting patience; 2/36 selected epoch 60.
- All recorded training objectives after epoch zero and all validation losses are finite. The model's gradient/parameter checks also completed.

| Partition | Seed | Loss | Selected epoch | Last stale epochs |
|---:|---:|---|---:|---:|
| 142 | 42 | mae | 59 | 1 |
| 142 | 42 | station | 59 | 1 |
| 142 | 44 | mae | 60 | 0 |
| 142 | 44 | selective | 57 | 3 |
| 142 | 44 | station | 60 | 0 |

A binding cap means this budget has not observed an ordinary early-stopping endpoint for
that fit. It does not establish that longer training would improve spatial transfer.
The fitted scale can absorb loss-induced changes in correction magnitude; both its value
and the selected epoch must be read alongside validation MAE.

## Interpretation

The paired table evaluates how each source objective aligns with the fixed cell-weighted
validation target. Equal-station source risk may improve sparse-station representation while
changing this alignment. Likewise, a penalty can reduce ordinary overprediction simply by
lowering predictions. Generalization, high-value recovery and false-alarm tradeoffs require
the separately retained outer evaluation; they cannot be inferred from these selected losses.
