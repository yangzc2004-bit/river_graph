# Joint source chemistry supervision for unmonitored-station DOC reconstruction

## Question

Source pH/conductance pretraining improved its auxiliary reconstruction but
worsened the subsequent DOC fit. The longer DOC optimization experiment gives
only0.10% central improvement. This study asks whether simultaneous supervision
can shape shared ecology/GRU states toward source chemical variation while
maintaining the DOC objective throughout training.

The prediction task stays DOC at a receiving station with no DOC, pH or
conductance. Auxiliary labels are training targets at source stations only.
They do not enter raw input channels, receiving-site availability features,
standardization of receiving data or checkpoint selection.

## Model and objective

Keep the retained log-trained station-hidden tree, original station-blocked OOF
references, ecology self encoder, observation GRU and native residual readout.
Attach two auxiliary linear heads to the same hidden state during training:
source pH and log1p(conductance). Fit their target standardization on source
training stations. Discard the auxiliary heads for DOC inference.

Use

`loss = retained tail-weighted native DOC MAE + 0.1 * auxiliary standardized MSE`.

Give the two auxiliary indicators equal expected weight despite different label
availability. At each DOC update draw a deterministic auxiliary source-cell
minibatch from all available source auxiliary labels, including source months
without DOC. Use the same source input view for both tasks. Keep all retained
DOC learning rates, batch size, initialization, scale candidates,30-epoch budget
and patience5. The auxiliary heads use learning rate0.001. The regularization
weight0.1 is fixed before fitting; there is no weight scan.

## Matched experiment

Partitions142/143/144 × seeds42/43/44. Two jointly trained arms:

- Real source auxiliary labels.
- Source labels shuffled within each auxiliary indicator, keeping stations,
  availability, scalers, head initialization, minibatch sampling policy and
  maximum training budget matched. Actual stop epochs may differ under the
  same DOC-validation patience rule.

Keep the existing complete model and strong trees as fixed comparisons. Produce
neural-only and unchanged memory-integrated DOC predictions for both new arms
on the same all-observation K0 source-validation cells. Do not add longer
optimization, nonlinear heads or the conditional-median reference to this study.

## Evaluation and next action

Evaluate native MAE, Q90, bias and source partition/seed effects, with5,000
paired station draws. Compare real supervision with both the retained complete
model and matched shuffle. Auxiliary loss reduction alone is not DOC success.
Inspect whether any DOC gain concentrates at high DOC or novel ecology.

Implementation should first check that a zero auxiliary weight reproduces the
ordinary DOC optimization, source labels cannot be taken from receiving roles,
hidden auxiliary values do not affect scaling/loss, future inputs cannot affect
earlier predictions, and DOC checkpoint/save-load predictions replay. Run one
package through the full product/replay path before continuing all nine. Save
the executed code, resumable neural stages and study configuration in this new
directory. Preserve all older fits. Previous geographical/external data remain
unused for development and no old runner is restarted.
