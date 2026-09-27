# U2 visible-network-context pilot status

The first context implementation used raw visible-station counts and was
stopped after its first configuration showed degraded performance. Counts were
then changed to visible proportions and directional coverage fractions. The
corrected `all` context arm completed two seeds for strict temporal
extrapolation before the expensive unmonitored-station arm was stopped.

The corrected partial results are not sufficient for a full U2 gate. They show
mixed behavior: DOC strict-temporal MAE was 1.245 and 1.276 for seeds 42--43,
against the 30-epoch reference mean of 1.223; specific-conductance MAE was
216.9 and 199.8, against the reference mean of 221.8. The context block is
therefore retained as a diagnostic branch, not selected as the next main
model. No U2 result is promoted to a confirmatory claim.
