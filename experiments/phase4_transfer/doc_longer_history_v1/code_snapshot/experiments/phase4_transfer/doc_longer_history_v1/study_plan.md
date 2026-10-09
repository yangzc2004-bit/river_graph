# Two-year causal memory in the retained DOC residual model

## Scientific question

Joint source chemistry supervision learned pH/conductance associations but
did not improve receiving-site DOC. The current retained neural residual uses
a12-month rolling GRU. Test whether its reset discards useful information from
the preceding year. The experiment concerns memory horizon; it is not an
increase in optimization epochs or a new backbone.

## Single model change

Change only `lookback=12` to `lookback=24` in the existing
`EncoderNativeResidual`. Reuse its ecology self encoder, observation-aware
GRU/decay,64 hidden units, native residual head and current-month daily hydro
features. Keep all parameter dimensions and initial weights unchanged. Read
only the current and preceding23 months. Start-of-record padding carries no
valid history and cannot update the state. Receiving-site water quality remains
absent; source station-fold-hidden input views remain unchanged.

Use the retained log station-hidden forest and cached station-blocked OOF
predictions. Preserve all input/scaling arrays, native MAE with Q90 weight2,
learning rates,30-epoch ceiling, patience5, batch512, scale candidates and
existing ecological-memory integration. The preceding chemistry auxiliary
heads, nonlinear readout, median reference and extended optimization are absent.
There is no window-length scan.

## Experiment and evaluation

Partitions142/143/144 × seeds42/43/44: nine new fits. Compare complete and
neural-only24-month predictions with their retained12-month counterparts,
strong station-hidden trees and the preceding complete model. Keep every
source-validation DOC cell and source-derived Q90 threshold unchanged. No
previous geographical/external query is evaluated for model development.

As an inference diagnostic, evaluate the fitted24-month weights with history
truncated to12 months and the same selected residual scale. This is not another
trained or selectable candidate: it measures the contribution of the additional
past months after fitting, separately from changes in learned weights.

Report MAE, RMSE, R2, log error, Q90, bias, station error and partition/seed
direction. Use5,000 paired station draws with seed means and equal partitions.
Check causality, early-record padding, initial weights, input equality,
parameter count, saved prediction replay and parent prediction preservation.
Start with one fully exported and verified package, then complete all nine.
Never adopt the candidate from the first package. Useful source gains motivate
subsequent confirmation; negligible or negative gains close this window change.
