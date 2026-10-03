# Next DOC iteration: put hydrologic history into the recurrent state

The daily-hydrology experiment established useful additional information,
especially after support adaptation. The fixed fallback improves missing-input
K0 behavior but slightly harms integrated K5. Retain both completed versions.
The next model change should address the unadapted spatial prediction directly.

## Current architectural gap

The existing encoder maps each month to a 64-dimensional ecological/self state.
Its GRU receives that state and a history-valid indicator. Daily discharge
descriptors enter only the final native residual head at the predicted month;
their earlier values cannot influence recurrent memory.

Add a bias-free, zero-initialized `Linear(8,64)` hydrologic input adapter:

\[
e'_s=e_s+P d_s,\qquad
h_s=\operatorname{GRU}([e'_s,\mathrm{valid}_s],\gamma_s h_{s-1}).
\]

This adds 512 parameters and retains the existing spatial encoder, GRU dimensions,
observation-age/support decay and native residual head. With zero P, the old
hidden state is recovered exactly. Use the same frozen daily feature block;
no new normalization, imputation or DOC-derived feature is needed.

## Matched experiment

Compare three ways to use the same daily information:

1. Head only: current daily model architecture.
2. Current month also enters the GRU through P.
3. Full causal 12-month daily history enters the GRU through P.

Use three station partitions and three seeds, giving 27 neural fits. Initialize
all from the same original spatial/GRU/decay states, zero scalar residual head
and zero P. Keep all 38 head features and interactions, tail weight 2, empty-edge
self path, learning rates and source-validation checkpoint selection identical.
Use the existing 120 epoch ceiling/patience 5; previous fits stopped by 77.
Current-only versus full-history distinguishes better current-input integration
from additional temporal information. Keep K0/1/3/5 products, with K0 central.

Fit an additional matched ExtraTrees benchmark with current daily features and
with the same 12-month daily window, using the saved forest hyperparameters.
The historical Temporal RF-context is a selected ExtraTrees implementation.
These independent benchmark fits do not replace the neural model's frozen
source-OOF context base. Comparing them helps separate hydrologic information
value from recurrent architecture value.

## Interpretation

Measure direct and integrated MAE, Q90/ordinary error, absence of daily inputs
and partition consistency. History gains over current-only would support a
recurrent hydrologic-state mechanism. If both tree and neural models benefit,
the new observations themselves carry transferable information. If neither
benefits, focus next on cross-station domain shift and feature coverage rather
than increasing recurrent capacity.

Use existing visibility and causal windows: month-end reconstruction, padded
months skipped, no future daily month or held-out DOC labels. First verify zero
initial equivalence and causal influence, then complete the matched pilot.
This document is the next research design; its implementation and training
have not started in this completed fallback experiment.
