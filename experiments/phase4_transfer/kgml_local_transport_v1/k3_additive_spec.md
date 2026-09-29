# K3 additive complementarity diagnostic

K1 supplied a learned local temporal residual and K2 supplied a
message-only upstream residual. Before training a new joint model, this
diagnostic asks whether those two frozen corrections are complementary.

For each mask and seed, the message weight `alpha` is selected on the K1/K2
validation query only from a fixed grid `[-1, 4]` with step 0.05. The test
prediction is then formed in the standardized target space:

```text
RF-local + K1 local residual + alpha × K2 message residual
```

The test table reports this additive result beside K1 learned no-message
residual, K2 message-only and RF-context. This is a post-hoc complementarity
diagnostic; it is not being presented as a new confirmatory endpoint.

## Current readout

- Temporal extrapolation selects alpha around 0.25--0.30 and gains about 0.9%
  over K1 local residual; its station-bootstrap interval crosses zero.
- Spatial holdout selects alpha around 1.6--2.8 and gains about 1.2%; the gain is
  small even though its station-bootstrap interval is positive.
- RF-context remains stronger in the spatial holdout.

The next training step is a joint local-plus-message model only if the study
needs a mechanistic decomposition figure. The diagnostic does not justify a
large lagged-transport matrix by itself.
