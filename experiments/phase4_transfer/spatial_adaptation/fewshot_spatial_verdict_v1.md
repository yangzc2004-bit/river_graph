# Paired few-shot spatial transfer verdict

## Scientific question

Can a source-station regional expert recover the local DOC level at an unseen
station after opening a small number of target-station observations? The
experiment evaluates this as **retrospective sparse reconstruction**: support
months are sampled across the historical record and the remaining hidden months
are queried. It is not a prospective five-month warm-start forecast.

## Paired protocol

* The outer E3 split holds out the same 43 target stations used by the strict
  spatial benchmark.
* Five candidate target-month support cells are reserved per station. K uses a
  nested prefix of this schedule: K = 0, 1, 3, 5 correspond to 0, 43, 129 and
  215 support cells.
* Every K is evaluated on the same 2,316 query cells: the outer E3 cells after
  removing the complete K=5 support set.
* Source-station regional ExtraTrees models are fitted once without target
  support. The support labels are used only for target evaluation features and
  a station-level log-space level correction.
* The correction strength is selected on a separate nested station-held-out
  validation split. Selected values are alpha = 0, 0.25, 0.5 and 0.5 for
  K = 0, 1, 3 and 5.
* Five outer seeds (42--46) use 300 trees, k = 40 source neighbours and
  minimum leaf size 4.

The station descriptors are standardized with all unlabeled station profiles,
including held-out target stations. This is label-free transductive covariate
scaling and is recorded explicitly; no held-out DOC labels are used.

## Main result

| target support K | raw context MAE | corrected MAE | reduction vs paired K=0 |
| ---: | ---: | ---: | ---: |
| 0 | 2.456 | 2.456 | 0.0% |
| 1 | 2.455 | 2.345 | 4.3% |
| 3 | 2.454 | 2.188 | 9.7% |
| 5 | 2.448 | **2.135** | **13.0%** |

The raw source model changes very little as support is added (2.456 to 2.448).
The improvement is therefore produced by the support-conditioned level
correction, rather than by simply adding support-derived context features.
The K=5 station-clustered paired bootstrap reduction is 13.0 percentage points,
with a 95% interval of 4.9--24.9 percentage points. K=1 is directionally
positive but its station-clustered interval crosses zero; K=3 and K=5 show the
clearest paired gains.

Five individual K=5 calibrated seed MAEs are 2.141, 2.146, 2.135, 2.136 and
2.152 (mean 2.142, SD 0.007). The paired-query aggregate is 2.135 because it
uses the same hidden query cells and averages errors at the cell level.

## Support-label diagnostic

Shuffling support labels across target stations removes the station-specific
signal. Mean MAE changes from 2.345 to 2.910 for K=1, from 2.188 to 3.852 for
K=3, and from 2.135 to 3.724 for K=5. The gain is therefore tied to the actual
DOC values observed at the target station, not just to the number of support
samples.

## Interpretation

The strict zero-shot benchmark remains a useful stress test, but its hardest
cases are dominated by an unobserved station-level DOC offset. A few target
observations let the same regional expert retain its cross-station seasonal
pattern while correcting that local level. This gives a concrete two-mode
spatial reconstruction result:

1. **Zero-shot:** source-similarity regional expert, paired-query MAE 2.456.
2. **Sparse target support:** the same expert plus target-level calibration,
   paired-query MAE 2.135 at K=5.

The result supports a spatial adaptation contribution for sparse water-quality
reconstruction. It does not show that deeper message passing alone solves the
zero-shot case, and it should not be described as a prospective warm-start
forecast without a separate forward-in-time support protocol.

## Reproducible outputs

* Protocol and support/query manifest:
  `fewshot_curve_paired_v1/query_manifest.json`
* Paired curve and station-bootstrap summary:
  `fewshot_curve_paired_v1/analysis/curve_summary.csv`
  and `fewshot_curve_paired_v1/analysis/paired_improvement.csv`
* Support-label diagnostic:
  `fewshot_curve_paired_v1/analysis/support_shuffle.csv`
* Paper figure:
  `fewshot_curve_paired_v1/analysis/spatial_fewshot_curve.png`
  and `.pdf`
* Runner: `scripts/run_spatial_fewshot_curve.py`
* Analysis: `scripts/analyze_spatial_fewshot_curve.py`

All outputs are versioned under `experiments/phase4_transfer/spatial_adaptation/
fewshot_curve_paired_v1/`; earlier strict zero-shot and variable-query
experiments remain unchanged.
