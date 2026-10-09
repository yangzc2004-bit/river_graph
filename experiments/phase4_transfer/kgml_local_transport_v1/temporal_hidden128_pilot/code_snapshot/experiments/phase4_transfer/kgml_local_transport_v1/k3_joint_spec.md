# K3 joint local--message residual pilot

The K3 post-hoc diagnostic found a small complementarity signal. This pilot
trains the two residual sources together against the same OOF RF-local error:

```text
prediction = RF-local + local temporal residual + upstream message residual
```

The local branch uses the K1 temporal model with an empty edge set. The message
branch uses the zero-preserving message-only GRU from K2. Both heads start at
zero, so the initial prediction is RF-local. The message branch has no bias
offset and its empty-message response is exactly zero.

Scope: DOC, `e2a_strict` and `e3_spatial_seed42`, seeds 42--44, six runs,
hidden 64, 20 maximum epochs, patience 5. The pilot is compared cell-by-cell
with K1 learned no-message residual and RF-context. A 30-epoch confirmation is
reserved for a clear improvement in the temporal holdout.
