# DOC concentration-space temporal residual experiment

## Motivation and scope

The v4 source-validation diagnosis identifies high DOC underprediction as a
large remaining error component. The original neural expert optimizes squared
standardized log residuals, whereas the reported primary error is native-unit
MAE. This experiment changes the scalar prediction objective on the existing
observation-aware GRU. It does not attribute all underprediction to that loss.
Raw-record and hydro diagnostics accompany the study; unusually high labels
are retained unless an actual data-processing defect is established.

The same three station partitions (142–144) and training seeds (42–44) are used.
These partitions have already been evaluated during development. This is a
development comparison, not an independent external confirmation.

## Two matched training arms

Both arms use the selected environmental ExtraTrees context prediction as a
fixed base, frozen spatial encodings, copies of the original expert GRU and
observation-decay weights, and a zero-initialized signed scalar linear head:

    DOC_new = max(0, DOC_context + scale * head(GRU(history)))

The first arm minimizes ordinary concentration-space MAE. The second uses
weight 2 for source-training observations at or above the source-training Q90
and weight 1 otherwise; its loss is normalized by the sum of weights. No
one-sided underprediction loss, tail oversampling or new input features is used.

Source-training forest predictions are the previously saved station-blocked
OOF predictions from the selected context forest. Neural source input views
hide labels of the corresponding held-out station fold. The pretrained neural
weights were learned on source stations; the entire neural model is not OOF.
Validation and target inputs expose source-training labels only.

Fixed settings: lookback 12 months, hidden size 64, 30 maximum epochs, patience
5, cell minibatch 512, Adam learning rate 1e-4 for GRU/decay and 1e-3 for the
new head, gradient norm clip 1. Overall source-validation K0 query MAE selects
the checkpoint and scale from {0, 0.25, 0.5, 1}. Epoch 0 is the unchanged
context model. Both arms are retained and reported; outer-test outcomes do
not select an objective, checkpoint, scale or default model.

## Support adaptation and controls

Report K={0,1,3,5} on the unchanged nested support and fixed query cells. All
three bases (context, native-MAE residual, tail-weighted residual) use identical
frozen v4 GRU support bases, plus a constant-only correction control. For each
base, support residuals are recomputed against its own corrected predictions;
the existing alpha/ridge grid is selected on source-validation episodes.
The old v4 fusion model is included as a performance reference.

K0 measures a change to the center prediction. K5 measures that change after
station adaptation; better K0 alone does not justify replacing a stronger K5
model. Support dates span the record and may follow a query: this is
retrospective reconstruction, not prospective forecasting.

## Evaluation

Primary diagnostic comparisons are each new arm against its matched context
base at K0 and K5. Also compare the two objectives and the old v4 fusion. Use
the existing partition-equal, seed-averaged paired MAE and joint station
bootstrap. Report native MAE/RMSE/R2, log MAE, Q90 and non-tail MAE, signed bias,
false Q90 alarms, changes per station, and concentration of residual gains.
Both positive and negative results remain in the comparison tables.

Outputs include saved weights, training traces, full-grid base/correction
components, adapted query predictions and replayable analysis. Old products
and dataset labels are unchanged.
