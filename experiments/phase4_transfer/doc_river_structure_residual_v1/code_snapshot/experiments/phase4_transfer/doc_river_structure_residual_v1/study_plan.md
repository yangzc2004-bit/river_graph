# River structure messages for unmonitored DOC reconstruction

This experiment asks whether real upstream observations and intervening river
structure add information to the complete environmental, temporal and source
retrieval model. Receiving stations have no DOC, pH or conductance inputs. It
uses the established 142/143/144 source training and validation roles and seeds
42/43/44. Historical geographical and external products remain fixed.

## Model and training

The current ecological encoder, observation-aware GRU, local residual readout,
source similarity attention and ecological memory integration are retained.
A zero-initialized message projection adds real river connectivity to this same
network. Three message components represent distance-conditioned conveyance,
area-weighted mixing and age-conditioned storage memory. These are predictive
representations, not estimated physical rates or concentration conservation.
Continuous path attributes condition their shared small encoder and gate.

The atlas proposal called for a frozen complete predictor and station-blocked
complete-model residuals. Reconnaissance found saved tree OOF predictions and
double-held donor references, but no complete-model OOF predictions. Before any
new fits, this pilot therefore changes to **joint neural training** on the
existing station-OOF environmental base. It does not fit a post-hoc graph branch
to in-sample complete-model residuals or label tree OOF as complete-model OOF.
All arms start from the same original encoder/GRU states and zero output heads,
not the fitted complete predictor. The no-river arm retrains the current
architecture at the identical 30 epoch, patience 5 budget. The saved complete
predictor is a separate fixed performance reference. The final prediction uses
the existing ecological memory integration; the river residual is not added
again after that integration.

## Source paths and availability

Candidates are up to twenty nearest eligible upstream source stations within
3000 km, found by tracing the atlas's directed station graph. Hidden intermediate
stations are allowed as physical waypoints, never as observation donors. Path
length, storage length and major junction counts are additive summaries of the
mapped station edge segments. This avoids calling absence of an upstream
monitor a physical headwater. Paths along tributaries, chains, confluences,
large mainstems and storage reaches use one shared network. The atlas's five
overlapping profiles remain diagnostic groups, not five independently fitted
models.

Messages read source log1p DOC departures from station-blocked tree references
at lags 0, 1 and 3 months. A latest observation may be retained for twelve months;
its actual age and validity are explicit. Missing observations are never
presented as new measurements. Source hydrology comes from the same causal daily
hydrology features already used by the current model. Monthly lag allocation
represents predictive memory, not river travel time.

For a training receiver in fold A, every source in A is excluded. A donor in B
uses the existing forest reference excluding both A and B. Neither receiving
labels nor donor labels enter that donor's reference fit. Validation banks use
only source training DOC and station-OOF source predictions. Validation labels
are used only for early stopping, scale selection and the existing memory blend.
No target test labels are read by development preprocessing or fitting.

## Comparisons

1. Saved complete current predictor: fixed reference.
2. Matched current retraining: new river parameters allocated but contribution
   fixed to zero; local/retrieval inputs and training budget unchanged.
3. Simple directed messages: identical river capacity and donors, but physical
   path attributes zeroed and component transformations reduced to identity.
4. Structure-conditioned messages: actual directed paths and continuous attrs.
5. Structure-conditioned rewiring: same receiving slots and path attributes,
   eligible non-ancestor donors matched by nearest log drainage area, without
   replacement. Rewired slots are a capacity/connectivity control, not real
   paths. Same eligibility rule applies separately to each training fold.

All four learning arms use the same optimizer and source-tail weighting as the
current model. Early stopping and scalar selection remain source-validation
procedures. A fitted zero-river prediction is also saved with all local and
similarity features unchanged; it diagnoses the contribution within that fit,
not an independently trained no-river architecture.

## Readout and next research decision

Report paired MAE against both fixed and matched current models; RMSE, R2,
station-equal MAE, Q90 MAE and signed bias. Use 5000 paired station bootstrap
draws, training seeds averaged within each split and splits equally weighted.
Split-specific results remain visible. Add continuous path length, upstream
availability and the five overlapping structure profiles. Save river lag mass,
zero-prior mass, entropy and component contribution diagnostics.

An advantage of real over rewired messages supports predictive value of actual
connectivity. An additional structure versus simple advantage supports path
conditioning. Where aggregate gains are small, inspect source support and
structure before expanding the model. Source development results select future
candidates; they are not external validation or untouched confirmation results.

All new products and executed source snapshots stay in this versioned directory.
Resume verifies completed stages and the execution snapshot. Old paper endpoints
and K-shot protocol are unchanged: this independent runner uses the later
unmonitored source-role task, not the frozen K-shot-v2 training entry point.
