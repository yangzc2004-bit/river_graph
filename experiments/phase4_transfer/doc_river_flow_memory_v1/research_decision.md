# River structure and monthly DOC–flow memory

Date: 2026-10-09. Exploratory source-role follow-up; no neural training.

## Finding and its model implication

**Recent observations support conditional hydrologic history information, but
static river form does not reliably explain the current/previous-flow response
differences across rivers.** A morphology-defined fixed lag is not supported.
The model direction is to learn dynamic allocation between current and history
states while testing the separate value of real upstream relationships.

The earlier flow study already tested contemporary C-Q response and found no
stable adjusted difference between coarse form classes. This round adds actual
preceding-calendar-month flow, genuinely chronological prediction, and separate
structural block comparisons for current, previous and combined response. It
does not classify the same networks again or infer event buffering from monthly
variation.

## Actual observations

DOC is restricted to 21,459 observed cells in the source-training union of
142/143/144. Previous flow is taken from the actual preceding dataset month,
even when that month has no DOC observation. Both flows must be observed,
finite and positive; absent, zero and reverse flow remain in the ledger.

| Population | Response stations | Paired months | HUC4 regions | Chronological stations / query months |
| --- | ---: | ---: | ---: | ---: |
| All source months | 206 | 13,846 | 49 | 72 / 4,722 |
| Observed-temperature adjustment | 205 | 13,565 | 49 | 69 / 4,566 |
| Since 2009 | 50 | 5,079 | 23 | 37 / 2,290 |

Station-specific season and year trend are controlled. The temperature
sensitivity also controls actual water temperature. Response is the partial
change in log1p DOC per unit log flow. Unlike log1p flow in cfs, centered log flow
is invariant to multiplying discharge by a unit-conversion constant. Zero flow
is outside this estimator, not a bad observation removed from old datasets.

Whole-region descriptor prediction uses fixed Ridge(alpha=10), preprocessing
inside each fold, and five HUC4-held-out folds. Explanatory response fits and
chronological checks answer different questions and are reported separately.

## Does previous flow improve later DOC prediction?

Earlier and later halves of each station's observed years are fixed before
fitting. All centering, trend reference and coefficients use earlier years only.
All arms score the identical later months. Errors first average months within
station and then stations equally. The main contrast adds previous flow to the
same current-flow/season/trend model.

| Population | log1p MAE reduction | 95% HUC4 interval | Native MAE reduction | 95% HUC4 interval |
| --- | ---: | --- | ---: | --- |
| All source months | −1.34% | [−7.94%, +2.77%] | −2.53% | [−7.82%, +2.56%] |
| Temperature adjusted | +0.81% | [−1.76%, +3.39%] | −1.51% | [−3.81%, +2.78%] |
| Since 2009 | **+9.80%** | **[+2.01%, +18.90%]** | +3.01% | [−3.34%, +11.30%] |

The recent log1p result has 26/37 stations improved. MAE is 0.2685 for current
flow and 0.2422 with previous flow. These are simple response-diagnostic models,
not the current complete neural DOC model. The recent subset is a declared
sensitivity, not a new primary endpoint. Full-history and temperature-adjusted
results do not establish the same improvement.

Relative to season/trend alone, the recent current+previous model reduces native
MAE by 8.28% [0.32%, 13.33%], but log1p gain is 3.81% [−4.51%, 11.84%]. Keep
this additional baseline distinct from the 9.80% comparison with current flow.

In 72 stations with identified responses in both halves, contemporary response
has the same sign in 79% and previous response in 64%. Same sign is descriptive,
not a significance test or evidence of consistent coefficient magnitude. The
observed scatter is heterogeneous, especially for previous-month response.

## Can actual structure predict those response differences?

Adding branching (drainage density and mainstem share) to the same environmental
and area context gives the following held-out **response-descriptor** MAE gains:

| Population | Current response | Previous response | Combined response |
| --- | ---: | ---: | ---: |
| All source months | −6.40% [−15.47%, −0.66%] | −8.93% [−16.82%, −2.35%] | −0.24% [−6.76%, +2.47%] |
| Temperature adjusted | −6.19% [−16.05%, +0.50%] | −1.78% [−10.54%, +5.90%] | −0.75% [−5.70%, +2.22%] |
| Since 2009 | +7.52% [−25.96%, +32.39%] | +6.72% [−23.68%, +31.30%] | +8.54% [−6.98%, +22.08%] |

Footprint, branching, paths and all-structure results for all three descriptors
and all three populations are retained (36 comparisons). No structural block
shows a dependable positive response-prediction improvement across populations.
Joint adjusted structural modifiers also do not establish a stable current or
previous response effect for branch density, mainstem share, sinuosity or route
distance variability. All intervals for these terms span zero in all populations.

The all-source joint moderation has 50 identified columns and scaled condition
44.7; the recent model has 48 columns for 50 stations, condition 98.5. Recent
interaction uncertainty is consequently large. Dropped collinear missingness
indicator terms are listed, never reported as estimated zero effects. These
associations cannot prescribe shape-specific transport coefficients.

The earlier positive branching information for station DOC level remains a
different result. Knowing a river's typical DOC level does not establish its
dynamic hydrologic response or validate a morphology-conditioned lag operator.

## Return to the graph model

Close this bounded analysis and return to the existing DOC model with three
design implications:

1. Keep measured hydrologic history and observation availability explicit.
   Current/history selection should be learned from states; this result does
   not assign a universal delay to a river-form class.
2. Treat form descriptors as candidate conditioning inputs. Compare the exact
   same descriptors in a local model and in a true-upstream graph operator so
   a static-feature improvement is not called a graph-message improvement.
3. Build the next comparison around dynamic **upstream state histories**, with
   true-network, no-message and matched non-upstream source controls. The prior
   frozen-readout study selected zero structural correction and suffered very
   sparse supported regional stations. Do not repeat it unchanged or assume
   direct upstream observed DOC exists everywhere. Separate observed upstream
   support from estimated upstream hidden states and report support strata.

Keep the complete current predictor as the fixed performance reference. The
next neural branch should begin as a zero residual, develop on source roles,
and test current versus history messages under the same local/ecological inputs.
Monthly candidate lags 0 and 1 are computational information choices, not
measured travel times. Wider lags remain a later comparison rather than an
assumed physical rule. A learnable operator remains a research hypothesis until
it improves the full-model comparison and true-edge controls.

Do not continue generating more static typology analyses on these records as a
prerequisite for the model. Independent synchronized DOC/flow would strengthen
process interpretation, while predictive graph experiments can proceed with
the available source-role data and explicit visibility.

## Independent discharge access

Official NEON product DP4.00130.001 is available, but the documented data API
requires an authorized token. One unauthenticated BLUE/2024-06 request returned
HTTP 403. No token was sought or used. The public openFlow viewer was inspected;
it is not a downloaded, QC-complete, archived raw time series for this study.
Independent NEON DOC–flow replication therefore remains unexecuted. Do not
replace actual NEON flow with a nearby unverified gauge or claim this ST357
study as external validation.

- [Official NEON discharge data and table definitions](https://www.neonscience.org/resources/learning-hub/tutorials/continuous-discharge-intro)
- [Official NEON data endpoint access requirements](https://data.neonscience.org/data-api/endpoints/data/)
- [Public NEON openFlow viewer](https://openflow.neonscience.org/)

## Reproduction and verification

```bash
uv run python scripts/audit_doc_river_flow_memory_sources_v1.py
uv run python scripts/analyze_doc_river_flow_memory_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_flow_memory_v1.py
uv run python scripts/plot_doc_river_flow_memory_v1.py
uv run --with nbformat --with nbclient --with ipykernel python scripts/build_doc_river_flow_memory_notebook_v1.py
```

The companion notebook executes all cells and recomputes the 18 temporal point
estimates. Standalone verification checks all 18 temporal and 36 descriptor
gains, all source cells, preceding-month flow alignment and identical query
sets. A sampled station response is independently fitted with the full nuisance
design instead of residualization. Tests cover unit invariance, collinear flow,
hidden DOC exclusion and chronological query-DOC perturbation. Figures were
actually viewed; the comparison legend was moved to avoid covering estimates.
Intervals use 5,000 paired whole-HUC4 draws, conditional on saved fitted
predictions; they do not include refitting uncertainty or multiplicity adjustment.

Full-suite and historical-audit results are recorded in verification_log.md.
