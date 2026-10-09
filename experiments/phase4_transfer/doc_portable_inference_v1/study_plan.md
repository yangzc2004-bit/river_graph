# Portable inference for the existing unmonitored DOC model

## Scope

Package the fitted geographical model and its fixed comparators for new sites.
This step changes inference interfaces, not architecture, component selection
or training. The first source package, HUC4_1013×seed42, is a deterministic
interface-replay fixture, not a model chosen from geographical test scores.
External02040104 DOC predictions have not been evaluated.

The prototype is `scripts/portable_doc_reconstructor_v1.py`. It offers
`from_geographical_run`, `prepare_inputs`, `predict`, `predict_components`,
`predict_with_support`, `save` and `load`. Training continues through the
existing `run_ladder` pipeline. A final unified training facade remains part
of subsequent product integration; this prototype does not claim it is done.

## New-site information

Inputs are site identifiers, a consecutive calendar-month grid, temperature,
discharge, their visibility, coordinates,13 regime attributes and8 existing
daily-hydro features. New-site DOC/pH/conductance labels, histories and flags
are not read. The model accepts a different number of sites and dates and
does not use positional ST357 node identifiers. Site IDs that overlap the
saved source library are rejected.

All input scales are saved from the fitted source model. Hydro scales use its
source training cells. The historical backbone's coordinate/regime scales used
the known ST cohort; the adapter preserves those exact scales and never fits
normalization on the incoming cohort. Readout and ecological-profile scales
retain their source-only OOF definitions.

The source bank stores only train/context DOC inputs and source train OOF
prediction errors. Source observations align by exact calendar month; months
outside the source observation grid receive zero support instead of an
extrapolated water-quality value. Global support fractions retain the saved
reference-cohort denominator. Optional named river edges determine directional
source summaries; unknown sites with no supplied edges receive zero directional
support. The fitted model's no-message neural path remains unchanged.

Observation age uses the saved source calendar origin to retain the historical
M1 view at matching dates. A requested date range starts with causal padding;
users should include the prior11 months of covariates when that history is
available. Input months must be consecutive, with explicit missing-hydro flags.

## Components and support

Export environmental prediction, local temporal correction, net source-memory
fusion correction and final DOC. River correction is explicitly zero for
this fitted no-message recipe; an inactive transport branch is not presented
as an effective mechanism. Source-profile diagnostics identify donor stations.
The memory correction replaces part of the existing temporal correction;
it does not add the same environmental error a second time.

K0 has no target calibration. K1/3/5 accept explicit support values only and
use the source-validation-selected shrinkage saved for that procedure and K.
Support never enters the forest, GRU or source bank. Designated supports retain
base predictions and are excluded from scored query cells. These K curves are
retrospective station calibration, not a claim of real-time forecasting with
future support.

## Verification and next work

`verify_portable_doc_reconstructor_v1.py` checks all five complete procedures
against their saved primary query predictions. The unmonitored candidate's
raw, ecology, age, support and extra inputs replay bitwise. It also checks
saved-state loading and arbitrary new-node counts. Predictions use a small
floating-point tolerance for parallel forest sums and different neural batch
sizes. Unit tests cover hidden labels, future features, site reordering,
source-library exclusion, component closure and explicit support isolation.

Keep large fitted exports local. Commit source, tests, this plan, the small
verification report and model manifest. Finish five-seed geographical analysis,
then freeze the deployment-source fitting/ensemble protocol before evaluating
the independent external case. Retain every comparator and the fixed full
candidate rather than selecting per-region or per-K winners.
