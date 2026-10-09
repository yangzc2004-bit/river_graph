# Time-aligned source DOC innovations in the retained reconstruction model

## Motivation

The source-only synchrony audit finds mean nearest-ecology same-month residual
correlation0.166 versus0.042 after calendar-preserving donor shuffle. All three
source partitions have positive real-minus-shuffle differences; lag1 and lag3
signals are weaker. This motivates a prediction experiment, not a claim of
new-station accuracy improvement or physical river transmission.

## First implementation: simple information probe

Keep the complete retained DOC model and its saved predictions. Add a sparse
source-innovation correction before training another neural branch:

`prediction = max(0, retained_complete_prediction + alpha * source_innovation)`.

Use actual DOC from source training stations only, subtracting their cached
station-blocked OOF environmental predictions in native units. Fit each donor's
calendar-month mean residual on its source observations, falling back to its
station mean for missing seasons. Subtract these fitted means to form innovations.
Freeze these source statistics before inference. Receiving labels never enter
the library. No pH/conductance inputs are introduced.

For each receiver choose at most20 source candidates by the existing normalized
nine-dimensional ecology. At each calendar month, use only source innovations
observed in that same month and year. No future-month value, climatology-filled
innovation or receiving water-quality observation is read. A missing source
month gives zero correction. Source-source predictions must exclude self.

Use fixed weights `exp(-ecological_distance / sigma)`, with sigma the median
positive source nearest-five distance, fitted on source ecology only. A
zero-innovation prior of weight1 shrinks sparse/distant support:
`innovation = sum(weight * available_innovation) / (1 + sum(available_weight))`.
Report source counts, weight mass, candidates and correction magnitude.

## Matched source experiment

Partitions142/143/144 × seeds42/43/44, using each retained source OOF cache.
No forest/backbone refit is needed for this first probe. Compare:

- Current complete model.
- Current complete plus real, time-aligned source innovations.
- Current complete plus a donor's earlier same-calendar-month innovation,
  sampled with fixed seed42 from dates strictly before the prediction month.
  For this matched comparison, both real and control use the same donor cells:
  current observation available and at least one earlier same-season value.
  Use identical candidates/weights and report the support lost to this condition.

The audit's across-year shuffle was a descriptive control, not an inference
product. It is replaced here before implementation because an unrestricted
year shuffle could place a future observation into an earlier prediction.
The causal historical control tests whether the current source event adds
information beyond the donor's past seasonal departures.

Select alpha from{0,.25,.5,1} on source-validation MAE separately for real and
historical control. Alpha0 exactly preserves the current model. This selection uses
development validation; a non-worse selected development MAE is expected and
does not establish independent transfer. Assess magnitude, paired station
intervals, real-versus-history and partition/seed consistency, plus Q90, bias,
station errors and source-support coverage. Keep query cells and thresholds.
Do not choose a different method by region or K.

Save the library, selected scalar, inputs and predictions in this new directory.
Test exclusion of receiving labels, source/self identities, arbitrary receiver
IDs/calendars, future-library-value perturbation with fixed statistics, missing
months, zero-alpha equivalence and save/load. Independently replay products.
Use5,000 paired station draws and inspected plots. First complete/replay one
package, then process all nine. Do not restart older geographical/external runs.

## Subsequent research

If useful development gains appear, compare against an explicit tree with the
same innovation information and investigate fusion into the EXISTING GRU/head.
Training a neural branch requires nested source libraries excluding the query
station fold; do not use a donor OOF prediction whose fit included that fold's
DOC labels. The45 complementary forest fits already cached by the reference-
trajectory study may supply this geometry. New-site water quality stays absent.
Only a source-selected candidate proceeds to a separate version of geographical
confirmation; do not retune it from old test-region or external outcomes.
