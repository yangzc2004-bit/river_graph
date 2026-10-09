# K2 source-isolation pilot v2

The first implementation used two full graph forwards and was too slow for a
small mechanism screen. This version uses a message-only spatial encoder and a
centered temporal state:

```text
message state = temporal(message-only input) - temporal(zero message input)
```

The prediction-head bias and GRU/history-valid offset therefore cancel. With
an empty edge set the output is exactly zero. The message arm and null arm
share the same architecture, target, masks and training budget.

Pilot scope: DOC, `e2a_strict` and `e3_spatial_seed42`, seeds 42--44, eight
epochs for screening. The full 30-epoch K1 budget is reserved for a follow-up
only if the message effect is clearly positive.
