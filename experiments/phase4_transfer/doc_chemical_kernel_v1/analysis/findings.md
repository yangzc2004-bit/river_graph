# Bounded chemical support interpolation: source-validation pilot

This report uses source-validation observations only. No outer station-test predictions or labels are read.
The native experts, selected linear support representations, calibrators and ecological mixtures remain frozen.
The added operator transfers the centered remainder left after the complete linear support correction.
Its RBF distance unit is fitted from source-station chemical coordinates, independently of DOC labels.

## Incremental station cross-validation

Five station folds select only eta and bandwidth on the other four source-validation folds, then score the held fold.
A shared RNG(3100 + partition) station assignment is used across seeds, pipelines and coordinate modes.
The parent was previously selected using source validation: this conditional CV reduces optimism for the added
kernel but is not an out-of-fold evaluation of the complete model. The full-validation optimum below is tuning
evidence; its eta0 fallback guarantees that it cannot be worse than the parent on its selection score.
No confidence intervals or independent-confirmation claims are attached to this development pilot.

| Pipeline | K | Comparison | CV candidate MAE | CV reference MAE | ΔMAE | Relative gain | Better partitions | Better seed fits |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| neural_chemistry | 3 | chemistry vs masks | 1.662812 | 1.658717 | +0.004095 | -0.247% | 0/3 | 2/9 |
| neural_chemistry | 5 | chemistry vs masks | 1.604917 | 1.603639 | +0.001278 | -0.080% | 1/3 | 2/9 |
| neural_chemistry | 3 | chemistry vs parent | 1.662812 | 1.658708 | +0.004104 | -0.247% | 0/3 | 2/9 |
| neural_chemistry | 5 | chemistry vs parent | 1.604917 | 1.603627 | +0.001290 | -0.080% | 1/3 | 2/9 |
| neural_chemistry_integrated | 3 | chemistry vs masks | 1.656726 | 1.651361 | +0.005365 | -0.325% | 0/3 | 2/9 |
| neural_chemistry_integrated | 5 | chemistry vs masks | 1.592138 | 1.591080 | +0.001057 | -0.066% | 1/3 | 1/9 |
| neural_chemistry_integrated | 3 | chemistry vs parent | 1.656726 | 1.651356 | +0.005370 | -0.325% | 0/3 | 2/9 |
| neural_chemistry_integrated | 5 | chemistry vs parent | 1.592138 | 1.591066 | +0.001071 | -0.067% | 1/3 | 1/9 |
| neural_chemistry_integrated | 3 | masks vs parent | 1.651361 | 1.651356 | +0.000005 | -0.000% | 0/3 | 0/9 |
| neural_chemistry_integrated | 5 | masks vs parent | 1.591080 | 1.591066 | +0.000014 | -0.001% | 0/3 | 0/9 |
| neural_chemistry | 3 | masks vs parent | 1.658717 | 1.658708 | +0.000009 | -0.001% | 0/3 | 0/9 |
| neural_chemistry | 5 | masks vs parent | 1.603639 | 1.603627 | +0.000012 | -0.001% | 0/3 | 0/9 |
| tree_chemistry | 3 | chemistry vs masks | 1.678272 | 1.673123 | +0.005149 | -0.308% | 0/3 | 1/9 |
| tree_chemistry | 5 | chemistry vs masks | 1.607420 | 1.603409 | +0.004011 | -0.250% | 1/3 | 2/9 |
| tree_chemistry | 3 | chemistry vs parent | 1.678272 | 1.673123 | +0.005149 | -0.308% | 0/3 | 1/9 |
| tree_chemistry | 5 | chemistry vs parent | 1.607420 | 1.603398 | +0.004021 | -0.251% | 1/3 | 2/9 |
| tree_chemistry | 3 | masks vs parent | 1.673123 | 1.673123 | +0.000000 | -0.000% | 0/3 | 0/9 |
| tree_chemistry | 5 | masks vs parent | 1.603409 | 1.603398 | +0.000011 | -0.001% | 0/3 | 0/9 |

Metrics pool held-query cells within each fitted seed, average seeds within partition, then weight partitions equally.
Full and auxiliary-active CV scores are both saved. The full score includes the unchanged inactive fallback.

## Complete full-validation fitted curves

These scores are evaluated on the same validation observations used to choose the kernel parameters.

| Model | K | Full MAE | Active MAE | Q90 MAE | Ordinary MAE | Signed bias |
|---|---:|---:|---:|---:|---:|---:|
| neural_chemistry_chemistry | 0 | 1.769764 | 1.770041 | 9.457147 | 1.111950 | -0.633568 |
| neural_chemistry_chemistry | 1 | 1.745523 | 1.745615 | 9.471841 | 1.081750 | -0.681557 |
| neural_chemistry_chemistry | 3 | 1.653609 | 1.653480 | 9.197006 | 1.002877 | -0.687954 |
| neural_chemistry_chemistry | 5 | 1.596571 | 1.596466 | 8.970354 | 0.968692 | -0.641505 |
| neural_chemistry_integrated_chemistry | 0 | 1.769265 | 1.769551 | 9.474950 | 1.110254 | -0.642404 |
| neural_chemistry_integrated_chemistry | 1 | 1.745263 | 1.745363 | 9.485361 | 1.080664 | -0.688260 |
| neural_chemistry_integrated_chemistry | 3 | 1.645588 | 1.645481 | 9.245454 | 0.988650 | -0.715127 |
| neural_chemistry_integrated_chemistry | 5 | 1.584281 | 1.583941 | 8.989323 | 0.952970 | -0.648204 |
| neural_chemistry_integrated_masks | 0 | 1.769265 | 1.769551 | 9.474950 | 1.110254 | -0.642404 |
| neural_chemistry_integrated_masks | 1 | 1.745263 | 1.745363 | 9.485361 | 1.080664 | -0.688260 |
| neural_chemistry_integrated_masks | 3 | 1.651016 | 1.650911 | 9.263640 | 0.991742 | -0.717747 |
| neural_chemistry_integrated_masks | 5 | 1.591048 | 1.590710 | 9.020067 | 0.956560 | -0.652873 |
| neural_chemistry_integrated_parent | 0 | 1.769265 | 1.769551 | 9.474950 | 1.110254 | -0.642404 |
| neural_chemistry_integrated_parent | 1 | 1.745263 | 1.745363 | 9.485361 | 1.080664 | -0.688260 |
| neural_chemistry_integrated_parent | 3 | 1.651356 | 1.651250 | 9.266300 | 0.991832 | -0.718522 |
| neural_chemistry_integrated_parent | 5 | 1.591066 | 1.590729 | 9.020913 | 0.956492 | -0.653036 |
| neural_chemistry_masks | 0 | 1.769764 | 1.770041 | 9.457147 | 1.111950 | -0.633568 |
| neural_chemistry_masks | 1 | 1.745523 | 1.745615 | 9.471841 | 1.081750 | -0.681557 |
| neural_chemistry_masks | 3 | 1.658178 | 1.658051 | 9.200480 | 1.007064 | -0.692991 |
| neural_chemistry_masks | 5 | 1.603552 | 1.603450 | 8.997246 | 0.972300 | -0.647718 |
| neural_chemistry_parent | 0 | 1.769764 | 1.770041 | 9.457147 | 1.111950 | -0.633568 |
| neural_chemistry_parent | 1 | 1.745523 | 1.745615 | 9.471841 | 1.081750 | -0.681557 |
| neural_chemistry_parent | 3 | 1.658708 | 1.658581 | 9.203625 | 1.007326 | -0.693907 |
| neural_chemistry_parent | 5 | 1.603627 | 1.603524 | 8.998592 | 0.972223 | -0.648025 |
| tree_chemistry_chemistry | 0 | 1.910344 | 1.910617 | 9.802820 | 1.221879 | -0.522496 |
| tree_chemistry_chemistry | 1 | 1.857964 | 1.858216 | 9.639807 | 1.182894 | -0.544363 |
| tree_chemistry_chemistry | 3 | 1.666998 | 1.666628 | 9.242471 | 1.010505 | -0.615750 |
| tree_chemistry_chemistry | 5 | 1.596646 | 1.596291 | 8.934352 | 0.966406 | -0.593174 |
| tree_chemistry_masks | 0 | 1.910344 | 1.910617 | 9.802820 | 1.221879 | -0.522496 |
| tree_chemistry_masks | 1 | 1.857964 | 1.858216 | 9.639807 | 1.182894 | -0.544363 |
| tree_chemistry_masks | 3 | 1.673119 | 1.672751 | 9.246373 | 1.016357 | -0.614622 |
| tree_chemistry_masks | 5 | 1.603393 | 1.603040 | 8.951124 | 0.972006 | -0.595618 |
| tree_chemistry_parent | 0 | 1.910344 | 1.910617 | 9.802820 | 1.221879 | -0.522496 |
| tree_chemistry_parent | 1 | 1.857964 | 1.858216 | 9.639807 | 1.182894 | -0.544363 |
| tree_chemistry_parent | 3 | 1.673123 | 1.672755 | 9.246695 | 1.016305 | -0.614715 |
| tree_chemistry_parent | 5 | 1.603398 | 1.603045 | 8.951502 | 0.971945 | -0.595727 |

## Full-validation kernel settings

| Pipeline | Mode | K | eta0 / .25 / .5 / 1 | bandwidth .5 / 1 / 2 |
|---|---|---:|---|---|
| neural_chemistry | masks | 0 | 9/0/0/0 | 0/9/0 |
| neural_chemistry | masks | 1 | 9/0/0/0 | 0/9/0 |
| neural_chemistry | masks | 3 | 3/0/0/6 | 6/3/0 |
| neural_chemistry | masks | 5 | 2/0/0/7 | 5/3/1 |
| neural_chemistry | chemistry | 0 | 9/0/0/0 | 0/9/0 |
| neural_chemistry | chemistry | 1 | 9/0/0/0 | 0/9/0 |
| neural_chemistry | chemistry | 3 | 1/7/1/0 | 4/2/3 |
| neural_chemistry | chemistry | 5 | 1/5/2/1 | 3/3/3 |
| neural_chemistry_integrated | masks | 0 | 9/0/0/0 | 0/9/0 |
| neural_chemistry_integrated | masks | 1 | 9/0/0/0 | 0/9/0 |
| neural_chemistry_integrated | masks | 3 | 3/0/0/6 | 6/3/0 |
| neural_chemistry_integrated | masks | 5 | 5/0/0/4 | 2/6/1 |
| neural_chemistry_integrated | chemistry | 0 | 9/0/0/0 | 0/9/0 |
| neural_chemistry_integrated | chemistry | 1 | 9/0/0/0 | 0/9/0 |
| neural_chemistry_integrated | chemistry | 3 | 3/3/3/0 | 3/4/2 |
| neural_chemistry_integrated | chemistry | 5 | 3/4/1/1 | 3/5/1 |
| tree_chemistry | masks | 0 | 9/0/0/0 | 0/9/0 |
| tree_chemistry | masks | 1 | 9/0/0/0 | 0/9/0 |
| tree_chemistry | masks | 3 | 7/0/0/2 | 2/7/0 |
| tree_chemistry | masks | 5 | 6/0/0/3 | 3/6/0 |
| tree_chemistry | chemistry | 0 | 9/0/0/0 | 0/9/0 |
| tree_chemistry | chemistry | 1 | 9/0/0/0 | 0/9/0 |
| tree_chemistry | chemistry | 3 | 2/5/2/0 | 3/4/2 |
| tree_chemistry | chemistry | 5 | 2/4/2/1 | 2/6/1 |

The separate CV-choice table records every held-fold selection and its training/held error.
K0 and K1 are exact parent controls. Fewer than two chemically active support rows, identical donor coordinates,
inactive queries or eta0 yield exactly zero kernel correction. Availability-only coordinates control the extra
kernel information; the already-frozen parent can itself use measured chemistry.

## Population and transfer diagnostics

Validation queries contain 7,197 unique station-months at 140 unique stations;
the three partitions contribute 8,047 station-month occurrences. Seeds repeat the same ecological observations.
Source-derived Q90 thresholds include ties. Tail groups below20 observations are marked unstable.
Correction summaries distinguish all-query and chemically active populations; maxima are diagnostics, not averages.
Station gain/harm contribution tables use exactly the partition-equal estimator and are included for both
the full-validation optimum and the incremental CV, so a few influential stations remain visible.

## Scientific scope

The decision about further testing should rely on coherent incremental-CV behavior relative to the frozen parent
and the availability-only kernel. A gain at the full-validation optimum alone cannot establish useful transfer.
Neither chemical similarity nor RBF weight is interpreted as physical transport or a causal coefficient.
A later frozen fresh-partition evaluation would be required for confirmation. This pilot never opens that evaluation.
