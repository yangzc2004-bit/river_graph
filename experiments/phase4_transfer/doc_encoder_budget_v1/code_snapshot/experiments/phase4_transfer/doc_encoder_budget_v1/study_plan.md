# Matched training-duration extension for the DOC encoder residual

The completed 30-epoch source-validation traces motivate this experiment:
14 of 27 fits reach the cap before patience is exhausted, including frozen
controls; ten establish a new validation best at epoch 30. The other fits
plateau earlier. This duration question is investigated before changing the
encoder learning rate or adding another architectural component.

Increase only the maximum training budget from 30 to **60 epochs**. Retain
patience 5, original expert initialization, learning rates, native residual
objective, tail weight 2, three encoder modes, source OOF bases, validation
queries, nine partition/seed packages, fixed support basis and ecological
memory. Begin every fit from its original weights and optimizer initialization.
The first 30 epochs must reproduce the corresponding earlier trace; fits
whose patience was already exhausted must reproduce unchanged outcomes.

The underlying model and per-run configuration use the existing
`doc_encoder_residual_v1` implementation. This new directory and execution
wrapper define the 60-epoch budget experiment. Its runtime snapshot also binds
this plan and the wrapper. Earlier 30-epoch products remain unchanged.

Keep the matched frozen control: a gain shared by all modes reflects longer
optimization rather than an encoder-specific contribution. Compare each
60-epoch mode with its own 30-epoch counterpart, and compare tuned versus
frozen encoders within the new budget. Report overall and Q90 MAE, ordinary
error, false-high rates, partition directions, validation choices and how much
additional validation improvement is obtained after epoch 30. All reported
models and K curves are retained.

This remains development on previously examined station partitions and reused
source validation. No new target-result threshold chooses the duration, learning
rate, mode or support setting.
