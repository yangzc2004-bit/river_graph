# Numerical refinement before interpreting the placement experiment

The first coarse/fine replay found stable peak heights (maximum difference
2.72e-5) and durations (8.78e-6), but grid argmax sometimes switched between two
nearly equal separated peaks. The largest reported peak-time difference was
0.891875 relative units. This is a non-unique maximum, not a transport change.

The response kernels, geometry selection, fractions, matching rules and source
forcing were kept unchanged. Local peaks are now refined continuously using
their grid brackets; the earliest maximum within relative height tolerance1e-8
is reported together with the last near-equal maximum, peak count and an
ambiguity flag. Component maxima use the same refinement. A symmetric double-
peak regression checks independence of grid spacing. All scenario tables and
summaries are regenerated with this numerical correction before interpretation.

CSV replay also now explicitly reads the monitoring component identifier as a
string: whole-network component labels preserve HUC4 leading zeroes, while the
existing monitored-system identifiers remain separate categorical strings.
