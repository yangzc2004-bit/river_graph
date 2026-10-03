# Source-validation comparison of four DOC training objectives

This diagnostic reads only bound model JSON summaries/traces and run configurations.
It does not read outer-query predictions, labels, or evaluation tables. All 36 completed
fits use the same last-self/ecology encoder, concentration head, source OOF context inputs,
120-epoch ceiling, patience 5, native output and fixed source-validation query set.

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
| mae | 1.881015 | 1.990984 | 2–67 | 7–72 | 0/9 | 1: 9 |
| selective | 1.854002 | 1.990984 | 19–61 | 24–66 | 0/9 | 0.5: 2, 1: 7 |
| station | 1.884681 | 1.990984 | 0–69 | 5–74 | 0/9 | 0: 1, 0.25: 2, 1: 6 |

Each displayed MAE averages seeds within partition and then the three partitions equally;
the balanced nine-run design makes this identical to their arithmetic mean. Within each run,
the underlying validation selection metric pools query cells. No inferential CI is attached
to these reused selection values.

## Paired validation differences from tail2

| Loss | Partition | Delta MAE | Relative reduction | Improved seeds |
|---|---:|---:|---:|---:|
| mae | 142 | +0.034067 | -1.757% | 0/3 |
| mae | 143 | -0.015220 | +0.821% | 2/3 |
| mae | 144 | +0.015683 | -0.864% | 0/3 |
| selective | 142 | +0.016073 | -0.829% | 0/3 |
| selective | 143 | -0.090840 | +4.901% | 3/3 |
| selective | 144 | +0.028260 | -1.556% | 0/3 |
| station | 142 | -0.012815 | +0.661% | 2/3 |
| station | 143 | +0.051051 | -2.754% | 0/3 |
| station | 144 | +0.007293 | -0.402% | 1/3 |

Negative Delta MAE favors the alternative objective. The CSV retains all nine paired
runs, checkpoint/scale changes and every recorded validation candidate choice.

## Duration and numerical stability

- 0/36 fits ended at the 120-epoch ceiling before exhausting patience; 0/36 selected epoch 120.
- All recorded training objectives after epoch zero and all validation losses are finite. The model's gradient/parameter checks also completed.

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

## Paired 60-versus-120 source-validation comparison

All 36 earlier training traces are identical to the corresponding prefix of the extended run.
The model is reinitialized from the original expert, with unchanged loss, learning rates,
sampling, source labels, validation query and patience. Only the maximum epoch count differs.

| Loss | 60-cap validation MAE | 120-cap validation MAE | Delta MAE | Reduction | Changed fits |
|---|---:|---:|---:|---:|---:|
| tail2 | 1.869505 | 1.869505 | +0.000000 | +0.000% | 0/9 |
| mae | 1.882749 | 1.881015 | -0.001735 | +0.092% | 1/9 |
| selective | 1.854275 | 1.854002 | -0.000273 | +0.015% | 1/9 |
| station | 1.885577 | 1.884681 | -0.000896 | +0.048% | 1/9 |

Selected validation loss cannot worsen when an identical trace receives additional
checkpoint candidates. Any reduction is a selection-set benefit of the larger budget,
not evidence that the same benefit transfers to held-out stations. Remaining cap-limited
fits, if any, are reported without proposing an automatic additional extension.
