# Concentration-relative source information

## Question

Current source innovations are in native mg/L. The same absolute departure
can represent a different event at a low-DOC and a high-DOC source. Test whether
concentration-relative source departures transfer more consistently between
stations. This plan is saved before opening the current attention geographical
analysis. It does not select changes from individual geographical outcomes.

## One change to the existing attention model

Keep the ecological self encoder, observation-aware GRU, two32-dimensional donor
heads, current41-feature scalar readout, candidate pool, source/validation roles,
reference forests,30epochs/patience5, learning rates and native tail-weighted
loss. No new forest, backbone, graph depth or attention-size search.

Source values become log1p-DOC minus station-blocked OOF log prediction,
centered by the donor's source-only seasonal mean. The attention-weighted log
departure is converted to a native residual by multiplying by
`1 + receiving_environmental_prediction`. This is the local first-order
conversion from log1p units to mg/L, rather than a physical conservation rule.
The output projection still begins at zero. The existing aggregate native
innovation inputs remain, allowing the new branch to supply complementary
relative information. Parameter count stays37,900.

Use double-held-fold references and source libraries excluding the source-query
fold, exactly as in the completed native-value study. The receiving reference
is its already saved station-OOF training prediction or full-source validation
prediction. Non-observed source cells have a zero placeholder that is never
selected for loss. No receiving DOC/pH/conductance enters K0.

## Fixed source experiment

Partitions142/143/144 ×seeds42/43/44, three arms (27 neural fits):

1. Learned relative-current attention.
2. Learned relative-earlier-season attention, with matched current keys/support.
3. Fixed ecological-prior relative-current attention.

Reuse all90 double-held reference forests. Saved native-current attention,
native-fixed/historical attention, preceding complete readout, retained complete
and strong trees remain comparators. No geographical or external refit belongs
to this source experiment.

Report complete and native MAE/Q90/bias/station-equal error, source-partition
directions and5,000 paired station intervals. Compare relative-current with
native-current and its relative matched controls. Diagnose allocation and
receiving-concentration groups without picking group/K winners.

First verify units, source-query exclusion, zero fusion, causal inputs and
save/reload, then one technical package and all nine fixed packages. Keep the
current release until a complete candidate has useful source and geographical
evidence. Preserve negative results and return subsequent development to source
roles.
