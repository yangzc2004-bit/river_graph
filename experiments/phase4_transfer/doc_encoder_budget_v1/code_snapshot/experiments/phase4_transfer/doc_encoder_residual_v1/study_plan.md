# Native-DOC representation tuning

The concentration-conditioned residual improves high-DOC reconstruction, with
small ordinary-concentration and false-high costs. Direct ecological interaction
does not resolve this specificity problem. We now ask whether the existing
spatial/ecological encoding, trained for an earlier residual task, should adapt
to the native-DOC objective together with the recurrent memory.

## Three matched encoder modes

1. **frozen:** the existing spatial/ecological parameters remain fixed.
2. **last_self:** train the final spatial self-linear layer.
3. **last_self_ecology:** also train the existing ecological encoder.

All three retain the same original encoder/GRU initialization and the same
concentration-conditioned scalar head (30 extra features; state interactions
with flow indices0/2/4 and predicted-concentration index28). The head starts at
zero. GRU/decay and head learning rates remain1e-4/1e-3; selected encoder
parameters use1e-5. Spatial dropout is off in every mode, as in the cached
frozen-encoding control. The existing input normalization is unchanged.

Only the no-message self path is used, matching the current spatial-transfer
residual. This experiment does not add river information or support a new
topological claim. Inactive message parameters and the old spatial scalar head
remain frozen. Each source-station input uses its saved station-fold-hidden
view, and its native base is the source OOF context prediction.

Raw chronological inputs are encoded within each12-month causal window so
gradients can reach the selected encoder layers. Left padding performs no
recurrent update. Validation/test labels remain hidden from raw inputs; target
support is only used by the existing post-hoc station adapter.

Each mode uses30maximum epochs/patience5/tail weight2 and overall source-
validation MAE checkpoint/scale selection, with epoch0 retained. Station
partitions142/143/144 and seeds42/43/44 match the earlier development panel.
The existing support basis and ecological residual memory remain fixed.
Direct and ecology-integrated predictions are reported for all modes and K.

Report overall MAE/log error/RMSE/R², Q90 error and bias, ordinary error,
false-high rate, partition/station effects, encoder parameter change and
source-validation choices. Compare partial tuning against the matched frozen
mode and extra ecological tuning against final-self tuning. The preceding
concentration head and best overall ecological model remain fixed references.

Float32 encoding can vary slightly with matrix batch size; cached versus raw
encoding equivalence is checked numerically. Saved-model predictions are
independently replayed. These are previously examined same-cohort station
partitions and retrospective support sets; every arm and failed combination
is retained.
