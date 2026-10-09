# Adapting the whole local encoder for new-station DOC

## Question

The retained native DOC fit updates the final self layer and ecological encoder
but freezes the first self layer inherited from the original source expert.
Test whether adapting this earlier representation improves receiving-site DOC.
This is a training-scope change in the existing two-layer encoder, not a deeper
network or a new attention branch. Its local expert continues to use empty
edges; the experiment does not establish river-message value.

## Single change

Use `encoder_mode="all_self_ecology"`: unfreeze both existing `self_lin`
layers and the ecological encoder. Keep edge/message parameters and the unused
spatial prediction head frozen. All parameters start at the same retained
initial values. Encoder learning rate stays1e-5. The model's allocated
architecture and total parameter count are unchanged; its trainable parameter
count increases by the formerly frozen first self layer and is reported.

Retain the12-month observation GRU, native residual head, log-trained
station-hidden forest, cached OOF predictions, input arrays/normalization,
daily hydro head features, native MAE with Q90 weight2,30 epochs/patience5,
batch512, all other learning rates, residual scale candidates and memory fusion.
Receiving stations have no DOC/pH/conductance. No auxiliary head,24-month
window, nonlinear readout or extended schedule is added.

## Evaluation

Source partitions142/143/144 × seeds42/43/44: nine new fits. Start with one
exported/replayed package, then continue the fixed set. Compare complete and
neural-only predictions with their retained last-self/ecology counterparts,
strong trees and the preceding complete model. All query cells and source Q90
thresholds remain fixed. Check first-layer update magnitude, frozen message
weights, initialization/input equality, label/future isolation and save/load.

Report native/log errors, Q90, bias, station errors and partition/seed directions
using5,000 paired whole-station draws, seeds averaged and partitions equal.
Do not evaluate old geographical/external queries for development. A reliable
source gain motivates separate confirmation. Small or negative gains close
this change; no encoder-learning-rate scan is planned.
