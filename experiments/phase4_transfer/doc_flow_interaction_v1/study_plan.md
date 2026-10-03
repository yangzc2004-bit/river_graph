# State-dependent flow corrections for DOC reconstruction

## Scientific question

The additive flow readout improves average station-transfer error, mainly at
ordinary DOC concentrations. Its numerical flow effect is shared across all
stations. This iteration asks whether conditioning that effect on the existing
recurrent state improves reconstruction, especially at high DOC.

Keep the original spatial encoder and environmental forest fixed. Reuse the
same ten causal discharge features and observation-aware GRU. Extend the
scalar head from [h, f] to:

    [h, f, flatten(h outer f_numeric)]

where h is the current 64-dimensional recurrent state and f_numeric selects
the relative anomaly and 1/3-month change columns (indices 0, 2, 4). The head
receives 192 additional interaction terms. Hidden-major outer-product ordering
is fixed. The scalar head is entirely zero initialized, including interaction
weights, so training starts at the context-only prediction.

The encoded recurrent state includes ecological, hydro and observation
information; it is a learned representation rather than a physical state.
Interaction coefficients describe predictive conditioning, not causal effects
or physical transport parameters. No new graph edges or future inputs are used.

## Two new arms and saved references

- `interaction_tuned`: train the original GRU, observation decay and new head.
- `interaction_frozen`: keep the original GRU/decay fixed and train only the
  identical expanded head.

The main architecture contrast is tuned interaction versus the previous
additive full-flow model (`flow_values`): inputs, recurrent training, objective
and training ceiling are unchanged. The new head adds 192 trainable weights.
Tuned versus frozen interaction measures the value of updating recurrence within
the expanded head. Frozen interaction versus the previous additive model is a
performance comparison that changes two factors, not an isolated interaction
effect.

Retain saved additive flow references, the original environmental context and
the v4 fusion adapters. The previous full-flow products are never overwritten.
All arms use the same frozen v4 GRU support basis plus the constant-only adapter.

## Training and data

Use the existing ST357 station partitions 142–144 and training seeds 42–44.
They have already been studied; these are development comparisons. Train
eighteen models with the same source station-blocked OOF context predictions,
source-fold-hidden DOC inputs, and full-source validation/test bases.

Keep tail weight 2 at/above the source-training Q90, native MAE, lookback 12,
30 maximum epochs, patience 5, minibatch 512, gradient norm clip 1, and Adam
rates 1e-4 for the recurrent parameters (when trainable) and 1e-3 for the head.
Source-validation K0 overall query MAE selects checkpoint and scale from
{0,0.25,0.5,1}, including the unchanged context model. The training objective and
selection criterion remain unchanged from the additive study.

For K={0,1,3,5}, recalculate each arm's support residuals against its own
predictions, then select the same alpha/ridge grid on source-validation
support/query episodes. Target query labels enter only final evaluation.
Source pretrained neural weights are not independently OOF; the forest bases
are station OOF. Support spans the record, making K-shot reconstruction
retrospective.

## Evaluation and completion

Compare tuned versus additive at K0/K5, frozen versus additive, and tuned
versus frozen. Report the environmental and previous fusion references as
well. Retain all arms and all K curves, with native/log error, Q90/non-tail
MAE, signed bias, false high-DOC rates, partition/seed directions and station
gain/loss concentration. Use the existing partition-equal, seed-averaged
estimand and 5,000 joint whole-station bootstrap draws.

A lower overall error alone does not demonstrate a high-DOC benefit. Read
the paired tail result alongside ordinary-concentration error, and distinguish
the interaction increment from any benefit of updating recurrent weights.
Do not change the objective, checkpoint or published comparisons using outer
query results.

Save the weights, training trace, complete flow-feature definition, native
residual components, full-grid predictions, support-adapted queries and replay
results. Tests cover old checkpoint compatibility, causal input use, zero-head
identity, interaction indexing and exact frozen-memory behavior.
