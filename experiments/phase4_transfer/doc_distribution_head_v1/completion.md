# DOC conditional density-head experiment: completed

## Experiment

Eighteen new heads were fitted across station partitions142/143/144 and
seeds42/43/44: a single Gaussian and an ordered two-Gaussian mixture per
package. Both use the retained expert's actual550 head features. The forest,
ecological/self encoder, observation-aware recurrent state, ecological residual
memory and legacy support basis remain fixed. The original native point model
is reproduced exactly as the reference.

The density target is log1p(truth) minus log1p(fixed native base). Mixture point
predictions use the actual conditional median. Source-validation K0 native MAE
selects checkpoint and global correction scale; existing direct support and
ecological integration rules are subsequently refitted on source validation.
Only the source forest baseline is OOF; the frozen neural expert is source-trained.

All nine packages completed in80.53 summed seconds. The18 fits ran397 epochs
in total, selecting epochs0–50 and stopping by60; none reached the100 ceiling.
Each head has one exact point fallback among the nine packages. An epoch0
source-initialized constant correction is distinct from scale0 fallback.

## Main point-prediction results

The principal integrated curves use the same retained GRU support basis.
MAE is in mg/L, pooled within each seed, seed-averaged within each partition,
then equally averaged across the three partitions.

| Head | K0 | K1 | K3 | K5 | K5 Q90 MAE |
|---|---:|---:|---:|---:|---:|
| Retained point | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 6.586994 |
| Single Gaussian | 1.805217 | 1.759075 | 1.620028 | 1.563876 | 6.593637 |
| Two-Gaussian mixture | 1.799271 | 1.753136 | 1.623813 | 1.562012 | 6.578240 |

The mixture has lower aggregate K0/K5 MAE by0.237%/0.190%. Its paired
station-bootstrap changes are:

- K0: −0.004267,95% interval[−0.028346,+0.017739].
- K5: −0.002972,95% interval[−0.011097,+0.004623].

All twelve fixed overall-MAE contrasts have intervals crossing zero. Integrated
K5 improves in3/9 seed-partition packages, worsens in5, and ties in the exact
fallback package. Its partition changes are+0.002463,+0.000341,−0.011719 for
142/143/144. The aggregate reduction comes from partition144.

K5 mixture Q90 error falls by0.133%; ordinary MAE changes from0.985640 to
0.983366. Both differences remain unresolved. High-DOC bias is still−4.921038
mg/L, versus−4.965678 for the retained model. At K0, direct mixture recall
falls1.076 percentage points, interval[−2.342,−0.014], while Q90 MAE rises
slightly. A solution to high-DOC underprediction has not emerged.

## Source-validation finding

Direct validation K0 MAE changes from1.804428 to1.786599 for the Gaussian and
1.774160 for the mixture, reductions of0.99%/1.68%. The mixture improves eight
validation packages and falls back in one. Selected source/validation NLL means
are0.1556/0.4044 for the mixture, and0.2247/0.4498 for the Gaussian.

The learned correction varies across cells and exceeds the small improvement
of an epoch0 constant correction. However, its relation to predicted
concentration varies between packages. Mixture components overlap substantially;
they do not define identified ordinary/high-DOC ecological states. The improved
validation fit does not transfer into a comparably sized held-station gain.

## Model decision

**Retain the original daily-head/legacy-support integrated model as the main
model.** Keep the mixture as a trained challenger with slightly lower aggregate
K0/K5 estimates. Do not introduce a target-selected K-specific route or treat
likelihood improvement as calibrated uncertainty or a performance breakthrough.

The next iteration should address spatial transfer or add informative covariates,
using the existing model as its base. Another head-size or tail-weight sweep is
unlikely to resolve the central high-DOC information gap.

## Verification and products

- Full suite:777 passed,2 skipped,3 existing warnings.
- Ruff: all checks passed.
- Historical `audit_artifacts.py --verify`: exit0.
- Independent replay: all nine packages and18 head states verified.
- Actual source550-feature arrays, source native base, source-only normalization,
  initialization, selected source/validation likelihood and selection trace verified.
- All504 query panels,2,161,320 rows replay bitwise;144 parent point panels exact.
- Full-grid output:9×233,478=2,101,302 rows.
- Independent SciPy median/native formula differences are at most8.88e−16 and
  9.95e−14; saved-state product replay is exact.
- All72 direct adapters,54 integrated mixers and432 source-validation panels refit.

See `analysis/findings.md`, `analysis/interpretation.md`,
`verification/replay_checks.json` and `PRODUCTS.md` for full comparisons and
reproduction. Old experiments and endpoints remain unchanged. These reused
station partitions are development data; positive-K adaptation is retrospective
reconstruction rather than prospective monitoring.
