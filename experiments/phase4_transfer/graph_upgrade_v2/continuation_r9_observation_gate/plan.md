# R9 observation-gated transport pilot

Test the next hypothesis from R7/R8: graph messages are useful for temporal
extrapolation only when the receiving station has enough compatible historical
support. `msggate2` keeps the two-layer H2X spatial trunk and M1 temporal
memory, but learns a node/month gate on the directed message sum. The local
self path is unchanged. Gate weights are initialized at 0.5 and are
data-driven through the current hidden state, which includes observation and
hydro/ecology channels.

Run DOC, `e2a_strict`, seeds 42--44, 30 epochs and patience 5. Compare with
the completed M1, `msgres2`, and RF/no-message blend results. This is a
targeted temporal pilot; do not spend a spatial matrix unless its temporal
result is competitive.
