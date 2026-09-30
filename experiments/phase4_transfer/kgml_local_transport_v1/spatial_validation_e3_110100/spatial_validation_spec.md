# Spatial validation block with upstream support

This is a model-selection companion to the frozen E3 station holdout. The
original E3 test stations and their query cells are unchanged. HUC6 `110100`
is reserved as a spatial validation block because the remaining source graph
contains directed train-to-validation edges; the earlier `101900` block did
not. The choice uses only station metadata and graph connectivity.

The validation block is used to choose the spatial residual configuration and
any shrinkage or training setting. Its labels remain hidden from fitting and
are opened only for validation. The final E3 test remains the terminal
evaluation and is not used for selection.

This block is a targeted spatial diagnostic, not a claim that one HUC6 is a
representative sample of all spatial transfers. Results are reported together
with the original spatial-validation result so that the dependence on graph
support is visible.
