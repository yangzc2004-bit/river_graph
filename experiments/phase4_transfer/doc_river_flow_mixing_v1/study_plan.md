# Flow weighted DOC mixing at observed confluences

## Question

How much of the smaller receiving-stream DOC variation can be reproduced by
mixing the two observed upstream concentrations, and what remains to explain
along the common downstream path?

The prior campaign CV result has already been seen. This is a new exploratory
mechanism analysis, retaining all 11 eligible windows and their original
one-use, three-site campaigns. It does not change the original three river
forms or the previous experiment's tables.

## Comparisons

- Match each site's laboratory sample to discharge on its own fixed UTC+1
  calendar date. Use published daily means without interpolation or filling.
- Compute `C_mix = (Q_A*C_A + Q_B*C_B)/(Q_A+Q_B)` only with available,
  nonnegative upstream flow and positive combined upstream flow.
- Compare the source concentrations, flow-weighted mixture and actual receiver
  on exactly the same campaigns. Report missing-flow campaigns explicitly.
  A window needs at least five complete three-flow campaigns for variation
  summaries; all original windows remain in the availability table.
- Report mean concentration, raw SD and relative SD (CV) together. Smaller CV
  by itself is not smaller absolute variability or removal of DOC.
- Also compute a fixed-weight mixture using the mean upstream flow fraction
  within each window. Its variance decomposes exactly into two individual
  variance terms plus their covariance term. This distinguishes the averaging
  of branch signals from variation in the mixture weights.
- Report `(Q_A+Q_B)/Q_receiver`, catchment areas stated in the discharge files,
  and sampling-point/gauge separation. The two measured branches may cover
  only part of the receiving catchment. Do not label their load difference as
  DOC retention or infer an unobserved inflow's concentration as measured.
- Keep source path-length difference, common-path length and receiving station
  as the structural context. Repeated years are grouped by confluence.

## Outputs

All campaign calculations, gauge metadata, 11-window availability and complete
window comparisons; bilingual scientific figures; a research decision about
whether branch mixing explains the observed variation and what observational
comparison should follow. No neural training or new fitted response curve.

Inputs are the preserved laboratory campaigns, daily flow, mapped configurations
and original public-file preambles from `doc_river_event_observations_v1`.
Units remain mg C/L and m³/s; `C * Q` is g C/s. Daily-discharge weights are
descriptive approximations to flow at the laboratory sampling times.
