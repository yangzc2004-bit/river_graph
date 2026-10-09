# Spatial validation confirmation

The E3 test set remains the frozen 20% station holdout from
`e3_spatial_seed42`. To align model selection with spatial extrapolation, the
source stations are split again by whole station: HUC6 `101900` is reserved as
the validation block, excluding any stations already in the E3 test set. All
remaining source stations form training.

The test station identities and query cells are unchanged. Validation labels
are hidden from training and early stopping, while they become visible only in
the terminal test view, matching the existing role protocol.

The confirmation matrix is DOC, E3 spatial validation, seeds 42--44, and:

- RF-context;
- the exact zero-message context null;
- context-base plus upstream message residual;
- context-base plus both-direction message residual.

The primary question is whether the upstream residual gain survives when the
model is selected on an unseen spatial block. The both-direction branch remains
a spatial interpolation diagnostic. No architecture or endpoint is changed.
