# Continuation R5: residual and depth study

This continuation tests whether the spatial Transport GNN benefits from
additional propagation depth or skip connections. The temporal mechanism is
held at M1 (observation-aware memory), so this isolates the spatial trunk.

Variants:

- `res2`: two transport layers with residual skips;
- `res3`: three transport layers with residual skips;
- `res4`: four transport layers with residual skips;
- `res3_jk`: three residual layers with learned layer-wise fusion (Jumping
  Knowledge).

Each variant uses DOC, the temporal and spatial holdout families, three seeds,
and the same 30-epoch budget. All comparisons use paired station-month query
cells against the original M1 runs.
