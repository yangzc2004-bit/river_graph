# K5 dual-channel message pilot

K4 showed that target observations and hydro-ecological inputs each carry
river-message signal, while the all-input branch is strongest in temporal
extrapolation. K5 tests a model that keeps these channels separate until the
last spatial layer.

## Model

- target branch: target value/visibility/context and causal observation
  statistics;
- hydro-ecology branch: temperature, discharge, season, location, hydro
  regime and ecological embedding;
- both branches use the same directed upstream message layers;
- a two-way softmax gate combines the hidden states;
- gate inputs are flow, observation age, recent support and upstream support;
- the temporal module is the existing zero-preserving message-only GRU.

The empty-edge arm is the exact zero-message control. The K2 all-input
message branch remains the historical reference.

## Pilot

- DOC;
- `e2a_strict` and `e3_spatial_seed42`;
- seeds 42, 43 and 44;
- 8 epochs, patience 3;
- dual message versus zero-message, 6 configurations per arm pair.

The primary question is whether the gated combination improves on each
single-channel K4 branch and on the K2 all-input message branch. Gate weights
are descriptive: they are not interpreted as causal transport coefficients.
