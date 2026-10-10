# Actual source sampling support in the existing DOC river GNN

## Question and architecture

Does restoring the age and measured hydrological phase of upstream DOC samples
improve monthly reconstruction? Existing date research showed offset sampling
and variable discharge; this experiment turns that information into message
features rather than repeating morphology analysis. Retain the frozen complete
local/ecology/GRU/similarity model, real upstream candidates, monthly station-
blocked OOF innovations,0/1/3 memory slots, and ordinary two-head x32 attention.
Add a zero-initialized12-channel sampling encoder to its existing message state.
The scalar residual output also starts at zero. No new backbone or forest fits.
The new branch REPLACES the observed-DOC branch; innovations are not added twice.

Monthly DOC stays the result-row arithmetic mean. Source dates are summarized
with the SAME result weights: mean/latest date, within-month span, unique dates
and result count. These describe an aggregate interval, not a point sample.
Age is measured at fixed prediction month-end. Do not read a receiving DOC date,
DOC/pH/conductance, or hidden observation history. Dates only accompany visible
source cells; nested source banks exclude the entire receiving fold. The true
last donor month is recomputed from permitted visibility, not common matched age.

For actual sample days, measured nonnegative NWIS daily Q supplies normalized
sample-versus-monthly Q and exactly-seven-day phase. Preserve result-row weights
among observed pairs and record coverage fractions. Receiver phase uses its
fixed month-end and exactly seven days earlier. Missing/conflicting/reversing
Q is unavailable for these ratios; no interpolation or daily DOC is invented.
Existing signed monthly/daily covariates remain unchanged. Current-month inputs
are retrospective monthly reconstruction inputs, not within-month forecasts.

## Fixed comparisons

Source-development partitions142/143/144 x training seeds42/43/44.
Six fits/package:54 fits,30epochs,patience5, native MAE and source-Q90 weight2.

1. plain_upstream: sampling inputs zero, reproduces existing dynamic_lagged;
2. sampling_age: dates/support only;
3. sampling_hydro (PRIMARY): dates + sampled-source/receiver phase;
4. shuffled_sampling_hydro: source hydro summaries permuted within prediction-
   month x lag x receiving-fold pools, concentration/date support and receiver
   hydro fixed. Marginals preserved, correspondence removed;
5. matched_sampling_upstream;
6. matched_sampling_nonupstream: SAME real-slot timing/flow metadata, common
   DOC availability and path/age descriptors; donor innovation and existing
   hydro identity differ. This is an identity control, not fictitious transport.

All six retain the same extra encoder capacity; age/none inputs mask columns.
Keep the complete current model, strong trees, previous observed-DOC dynamic
branch and previous environmental-state branch as unchanged references.
Whole receiving-fold isolation and double-held environmental tree references
stay fixed. The complete base's training predictions are fitted, not complete-
model OOF. These validation roles are development data; geographical/external
results are not used to tune the experiment. Primary arm fixed before fitting.

## Interpretation and outputs

Native MAE/RMSE/R2, log error, Q90, station-equal errors, three-partition stability,
5,000 paired whole-station bootstrap draws. Primary contrasts: full timing vs
plain; full vs age; full vs shuffled; full vs previous environmental branch;
matched real vs nonancestor. Separate incremental effects from accumulated gain
versus the complete base. Report real date/flow coverage and age strata, keeping
unavailable inputs separate. Sampling support attention is information allocation,
not physical travel time or statistical causal effects. All arms stay reported.
Replay all predictions and rebuild inputs; check future/hidden-date exclusion,
receiver-date independence, flow QC, matched priors, zero/no-support behavior and
save/load. Inspect figures; write the scientific decision from actual results.
