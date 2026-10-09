# Source-state-aware attention for unmonitored DOC

## Scientific question

The 20-candidate current-availability model improves geographical K0, while
widening the pool to 60 did not improve source-validation MAE. The present
attention weights depend on ecology and daily hydro; donor DOC residuals
enter only the values. Test whether letting the keys represent source DOC
state helps the receiving query select relevant contemporary experience.

## Fixed mechanism

Start from `doc_current_availability_attention_v1`, with 20 ecological
candidates and finite current availability. Keep the actual donor values,
ecological prior, zero prior, environmental reference, receiving query,
matched aggregate descriptors, encoder, 12-month GRU, native loss and fusion.

Append two channels to the existing key:

1. Current source OOF log1p residual, divided by a permitted source-only RMS.
2. Its source calendar-month mean, divided by the same RMS.

RMS uses the entire allowed source-training residual library, without
centering; zero remains zero residual. Query fold A is excluded, and each
donor reference excludes A and donor fold B. Each nested library has its own
normalizer; full validation uses the full permitted source library. Receiving
DOC/pH/conductivity never enter these keys. Fixed preprocessing may use source
training history for the spatial task; inference reads only a contemporaneous
permitted source observation. Padding, invalid donors and the zero prior have
zero key channels. No future source observation is read at inference.

Retain the old key coefficients and initialize the additional coefficients at
zero. Add only 128 parameters (two channels × two heads × 32 dimensions),
for 38,028 total. No new receiving features, backbone, head, window or loss.

## Three matched source arms

- **Current state:** both additional channels.
- **Seasonal state only:** zero current-residual channel, same seasonal channel.
- **Zero additional state:** both channels zero, same model capacity.

All three retain actual current source values and the same candidate validity;
the controls change selection keys only. Carry the preceding complete and
native-only 20-candidate model, matched-source procedures, retained complete
and trees unchanged. Keep splits 142/143/144 and seeds 42/43/44, 30 epochs,
patience 5 and existing validation scales. Nine packages, 27 neural fits,
90 reused references, no forest refit or acquired data.

## Scientific readout and continuation

Report incremental MAE/Q90 and bias relative to the actual preceding complete
and native-only models, current versus seasonal/zero keys, partition/seed
directions, station-equal errors, key-coefficient learning and allocation
entropy/prior mass. Use 5,000 paired station draws, seed means within source
partition, equal partitions. Keep accumulated gains over retained complete
and trees separate from this key contribution.

Complete technical fitting and saved-state replay before the fixed source
matrix. Tests cover source-only normalization, query/donor exclusion, zero
coefficient compatibility, finite gradients, causal source access, arbitrary
receiver counts and save/load. A useful incremental source-role result can
enter a fixed geographical replication; otherwise preserve the preceding
candidate and close this mechanism. Geographical/external outcomes do not
choose these channels, budget or controls. Keep earlier evaluated products.
