# Confluence mixing and storage in the existing DOC river branch

## Scientific question

Can explicit upstream mixture and buffering operators improve reconstruction
beyond ordinary sparse attention? Previous environmental states, contrasts and
conditional readout cross-fitting did not establish a substantial extra river
benefit. This round changes propagation itself, on the identical frozen local
predictor and observed upstream DOC innovation bank.

## Architecture and controlled comparisons

Retain the complete ecology/hydro/GRU/similarity model, queries and separately
station-blocked environmental innovations. Five arms, all two heads x32:

1. plain_upstream: ordinary learned edge-by-lag attention;
2. confluence_mixing: suppress nested catchment duplicates and add a normalized
   source-mixture prior;
3. confluence_storage (primary): add bounded positive storage/confluence/distance
   attenuation and a storage-dependent bias toward older valid memory slots;
4. matched_storage_upstream;
5. matched_storage_nonupstream.

For each lag, retain a donor only if no usable downstream candidate already
represents its nested branch. Independent sampled tributaries may then contribute
together. If every remaining donor has nonnegative measured discharge and the
sum is positive, normalize monthly NWIS discharge (cfs); otherwise normalize
mapped drainage area for the whole lag group. Do not mix area and discharge
units. Missing/reversing/all-zero discharge invokes the area fallback. A measured
zero-flow donor receives only numerical epsilon weight. Unmonitored tributaries
and intervening sources/sinks remain unaccounted for.

Matched real/nonancestor arms use the SAME real-slot frontier, flow/area prior,
path descriptors, ages and common DOC availability. Only donor identity and its
innovation/hydro features differ. A fictitious nonancestor slot is a control,
not a real flow path. All encoders retain path features: mixing/storage ablations
test explicit operator use, not complete removal of geometry information.

These are signed log1p innovation mixtures, not concentration or DOC load
conservation. Lags0/1/3 are monthly information memory, not measured travel time.
Nonnegative attenuation and older-memory bias are hypotheses about buffering,
not observations of physical residence time. All river readouts start at zero;
empty banks reproduce the retained model exactly.

## Roles and budget

Source-development partitions142/143/144 x seeds42/43/44. Five river fits per
package:45 fits,30epochs/patience5, same native-MAE loss and source-Q90 weight2.
Current receiver DOC/pH/conductance and their history remain absent. The source
query fold never enters donor truth or its environmental tree reference.
The fixed complete model's source training predictions are fitted, NOT full-model
OOF. No new readout refits, forest fits, geographic or external model selection.
Keep every arm and epoch0 result. Retain complete current model, strong trees,
previous observed-DOC dynamic_lagged and expanded_upstream environmental branch.
The new operator replaces the observed-DOC branch, rather than adding its same
innovations twice. All comparisons use the identical receiving query cells.

## Analysis and completion

Report native MAE/RMSE/R2, log1p error, source-Q90 tail, station-equal error,
three-partition stability,5,000 paired whole-station bootstrap draws, support,
flow-vs-area allocation, nested suppression, buffering and lag distributions.
Primary contrasts: storage versus plain; storage versus mixing; storage versus
previous observed branch and previous best environmental branch; matched real
versus nonancestor. Gains versus the complete base are accumulated, not the new
operator's incremental gain. Examine sparse source support and station coverage.
Tests cover causal dates, nested donor fallback, flow missing/reverse/zero,
normalization, zero initialization, no support, checkpoint replay and old operator
compatibility. Preserve code snapshot and retained results; inspect plots before
writing the research decision. Source-development intervals do not establish
independent confirmation or physical causality.
