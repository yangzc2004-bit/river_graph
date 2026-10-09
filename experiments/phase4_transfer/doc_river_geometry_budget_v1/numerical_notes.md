# Deterministic routing composition

The first internal calculation deposited a branch delay and a deterministic
common delay separately on the numerical grid. Two fractional deposits add
unintended interpolation spread and produced approximately 1e-5 concentration
differences in the junction-position null. The corrected calculation composes
the deterministic path before grid deposition. A fractional-path regression
test verifies the null; no input, forcing parameter or scientific comparison
was selected from a better observed result. The first null table is retained
in `development_diagnostics/first_deterministic_null.csv`.

The distributed common-segment gamma kernel is bin-integrated and normalized.
Tiny discretized centroid differences remain below the 0.005 scenario time
step and are numerical, not physical timing effects.

Conservative mass differences of order 1e-16 and deterministic-null moment
differences of order 1e-18 are retained in numerical tables. Bootstrap intervals
at these scales do not establish physical effects, even if their rounding does
not span zero. The governing conservation/null checks use numerical tolerances.
