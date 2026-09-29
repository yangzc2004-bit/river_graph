# R8: Is RF/M1 complementarity carried by river messages?

Repeat the completed R7 DOC blend experiment with the M1 spatial trunk's
message path removed (`edge_set=empty`). The observation-age, recent-support,
and upstream/downstream support features still use the river edge table, so
this is a matched-input no-message control. Only the encoder's directed edge
messages are removed.

Use the same two masks (`e2a_strict`, `e3_spatial_seed42`), three seeds
(42--44), 30 epoch budget, RF 200 trees, and validation-selected alpha grid.
The test view is unchanged: test labels are hidden while alpha and checkpoints
are selected. Compare the no-message blend to both its RF endpoint and the
completed river-message R7 blend. This is a diagnostic of graph-message
complementarity, not a claim that a blend is an end-to-end graph residual
model.
