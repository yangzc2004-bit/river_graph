# K4 channel-isolation pilot

## Question

The K2 message-only branch propagated all node inputs and the ecological
encoder. K4 separates two possible sources of its conditional gain:

1. **target-only messages**: target observations, visibility, age and support
   statistics on river edges;
2. **hydro-ecology messages**: temperature, discharge, season, location,
   hydro-regime and ecological encoder features, with target-derived channels
   removed.

The frozen K2 `all` message branch and the zero-message branch remain the
matched references. This is a diagnostic mechanism study, not a new primary
endpoint.

## Channel definitions

`target_only` retains the target value/visibility/context slots and the nine
causal observation statistics. It zeros hydrology, season, static location,
hydro-regime inputs and ecological encoder inputs before edge aggregation.

`hydro_ecology` retains the non-target spatial and ecological inputs but zeros
target value/visibility/context slots and all target-derived observation
statistics. Its temporal decay input is also zeroed so target observation age
cannot enter indirectly.

`all` is the original K2 message-only branch. `null` is the exact zero-message
control. All arms use the same RF-local base, masks, seeds and training budget.

## Pilot matrix

- analyte: DOC;
- masks: `e2a_strict`, `e3_spatial_seed42`;
- seeds: 42, 43, 44;
- 8 training epochs, patience 3;
- arms per output directory: message-only and zero-message.

The two non-reference directories are `target_only` and `hydro_ecology`.

## Interpretation

Positive paired gain over the zero-message arm means the retained channel set
contains predictive information in that missingness regime. The result does
not by itself establish physical transport or causal influence. The spatial
upstream-support contrast and the temporal observation-age contrast are read
alongside the channel comparison.
