# Source-validation support-basis comparison

Selected validation scores, seed-averaged within each partition and then partition-equal.
No target prediction file or target DOC outcome is read by this diagnostic.

| Expert | Pipeline | K | Legacy MAE | Refreshed MAE | Difference |
| --- | --- | ---: | ---: | ---: | ---: |
| current_only | direct | 0 | 1.801147 | 1.801147 | +0.000000 |
| current_only | direct | 1 | 1.772615 | 1.772615 | +0.000000 |
| current_only | direct | 3 | 1.673770 | 1.659929 | -0.013842 |
| current_only | direct | 5 | 1.614725 | 1.612743 | -0.001982 |
| current_only | integrated | 0 | 1.800971 | 1.800971 | +0.000000 |
| current_only | integrated | 1 | 1.772440 | 1.772440 | +0.000000 |
| current_only | integrated | 3 | 1.663218 | 1.648819 | -0.014399 |
| current_only | integrated | 5 | 1.604251 | 1.602396 | -0.001854 |
| full_history | direct | 0 | 1.798270 | 1.798270 | +0.000000 |
| full_history | direct | 1 | 1.768582 | 1.768582 | +0.000000 |
| full_history | direct | 3 | 1.672602 | 1.659240 | -0.013362 |
| full_history | direct | 5 | 1.614508 | 1.610879 | -0.003630 |
| full_history | integrated | 0 | 1.797794 | 1.797794 | +0.000000 |
| full_history | integrated | 1 | 1.768107 | 1.768107 | +0.000000 |
| full_history | integrated | 3 | 1.663310 | 1.649952 | -0.013358 |
| full_history | integrated | 5 | 1.604384 | 1.601889 | -0.002495 |
| off | direct | 0 | 1.804428 | 1.804428 | +0.000000 |
| off | direct | 1 | 1.774421 | 1.774421 | +0.000000 |
| off | direct | 3 | 1.675127 | 1.660738 | -0.014389 |
| off | direct | 5 | 1.616446 | 1.615358 | -0.001088 |
| off | integrated | 0 | 1.804067 | 1.804067 | +0.000000 |
| off | integrated | 1 | 1.774060 | 1.774060 | +0.000000 |
| off | integrated | 3 | 1.663327 | 1.648791 | -0.014536 |
| off | integrated | 5 | 1.604602 | 1.604835 | +0.000233 |

K0 is identical by construction. Positive-K choices use the unchanged source-validation
alpha/ridge grids and, for integrated products, positive-K ecological mixing.
Readout and native predictions are fixed; refreshed hidden coordinates may differ from
the old support-episodic coordinates. All three experts and both pipelines are retained.
