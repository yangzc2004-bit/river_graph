# H2X-T results chapter (v1)

## Study design

We evaluated the released ecology-aware spatial graph model (H2X) and its
causal temporal extension (H2X-T) on the ST357 cohort. H2X-T applies the same
spatial encoder at each monthly snapshot and passes the resulting node
representations through a one-layer GRU with a 12-month causal window. The
three analytes were dissolved organic carbon (DOC), pH, and specific
conductance. The comparison used the same target-specific masks and query
cells for both models, five training seeds (42--46), and the matched
10-epoch/patience-3 training budget.

We considered four missingness families: random gaps (random cell masking),
unobserved-period extrapolation, observation-assisted extrapolation, and
unmonitored stations (held-out spatial stations). The primary comparison is
paired MAE on the hidden query cells. A negative delta (H2X-T minus H2X) favors
the temporal model. Station-clustered and month-clustered bootstrap intervals
use 2,000 paired replicates; training seeds are averaged before resampling and
are not treated as independent ecological observations.

## Temporal modeling improves reconstruction, with strong analyte dependence

H2X-T reduced MAE in every analyte-by-missingness family in the five-seed
comparison. The gains were large for DOC and specific conductance and smaller
for pH.

For DOC, MAE decreased by 37.5--47.2% across the four missingness scenarios.
The largest reductions occurred in unobserved-period extrapolation (47.1%) and
observation-assisted extrapolation (47.2%),
where the snapshot model had no access to the target-month temporal context.
Both station- and month-clustered intervals for the MAE delta remained below
zero in every DOC family.

For specific conductance, the corresponding reduction was 38.8--57.2%.
Unobserved-period extrapolation and observation-assisted extrapolation produced
reductions of 56.7% and 57.2%, respectively, with
negative bootstrap intervals under both clustering choices. The result shows
that the temporal extension is especially valuable for an analyte whose
variation is strongly coupled to evolving hydro-ecological conditions.

For pH, H2X-T reduced MAE by 0.5--3.8%. Random gaps, unobserved-period
extrapolation, and observation-assisted extrapolation showed small but
consistent improvements across most seeds. The unmonitored stations result was close to parity:
the month-clustered interval was slightly below zero, whereas the
station-clustered interval included zero. We therefore treat pH as a
small-effect analyte whose benefit depends on the missingness regime, rather
than pooling it with the larger DOC and conductance effects.

The improvement is not confined to one scenario. DOC and specific conductance
were better for all five seeds in all four families. The pH comparison was
better for four of five seeds in random gaps, unobserved-period extrapolation, and observation-assisted extrapolation, and three of five in unmonitored stations.

## How much temporal history is useful?

A matched three-seed window diagnostic compared lookbacks of 1, 3, 6, and 12
months in unobserved-period extrapolation and observation-assisted extrapolation. Mean MAE across the two temporal families was:

| analyte | 1 month | 3 months | 6 months | 12 months |
|---|---:|---:|---:|---:|
| DOC | 1.420 | 1.419 | 1.417 | 1.415 |
| pH | 0.362 | 0.319 | 0.305 | 0.301 |
| specific conductance | 279.12 | 255.19 | 247.72 | 245.75 |

The main gain for pH and conductance appears by six months; extending the
window from six to twelve months adds only 1.3% and 0.8% relative MAE
reductions, respectively. DOC changes little across windows. We retain
12 months as the working model because it gives the best observed point
estimates and preserves a common configuration across analytes; a six-month
version is a plausible lower-cost operational alternative.

The window result is an information-and-computation comparison: changing the
window changes both the amount of history and the GRU unroll. It does not
identify the contribution of a particular lag or prove that every month in the
window is necessary.

## History ablations and model interpretation

Two three-seed ablations were used to interpret the temporal gain. The
reverse-history arm retained the same history values but reversed the preceding
steps before the current month. The hydro-only arm removed the target-value and
target-visibility channels from every month while retaining hydro, ecology,
seasonal, and validity inputs. Both arms stayed within 0.6% of full H2X-T in
MAE for the observed analyte-by-mask families at the matched budget.

A separate current-only control used the same GRU wrapper with `lookback=1`.
Relative to H2X-T, current-only MAE was 20.9% higher for pH and 13.8% higher
for specific conductance in the two temporal extrapolation scenarios, while DOC changed by 0.5%. Together,
these results support a practical conclusion: a longer input history matters
for pH and conductance, but the present experiments do not isolate a unique
contribution from strict chronological order or from the target analyte's own
past values. We therefore describe H2X-T as an ecology- and hydro-aware
causal temporal extension, without assigning the gain to a single channel.

## Ecological and spatial patterns

The five-seed ensemble product provides a full station-month reconstruction
for each analyte and missingness family. The station-error maps show where
unobserved-period extrapolation test errors concentrate across the Mississippi graph, while the station
traces illustrate how the ensemble follows seasonal variation and where
observations depart from the reconstruction. These displays are descriptive:
station examples were chosen by a fixed data-availability rule and were not
used to select a model or endpoint.

The ensemble seed spread is retained as a model-dispersion diagnostic. It is
not a calibrated prediction interval. The earlier uncertainty experiments
showed that empirical recalibration can improve nominal coverage while
inflating interval width, and that positive error enrichment alone is
insufficient to establish stable monitoring blind spots. The present temporal
results make no active-sampling or blind-spot claim.

## Provenance and scope

All 300 source products used in the synthesis passed their frozen artifact
integrity audits. The historical sidecars for 144 products contain temporal
fields that do not fully describe the runtime arm; the prediction files and
original audits are retained unchanged, and the discrepancy is recorded in the
metadata ledger. The comparison is therefore a matched-budget performance
analysis with an explicit provenance note, not a claim of full convergence.

The resulting central claim is conditional: adding a causal temporal wrapper
to an ecology-aware spatial graph model substantially improves DOC and
specific-conductance reconstruction under the tested missingness regimes,
while pH shows smaller and regime-dependent gains. The magnitude is specific
to the ST357 cohort, masks, preprocessing, and training budget.
