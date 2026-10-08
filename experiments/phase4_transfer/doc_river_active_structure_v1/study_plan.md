# Flow-dependent participation of real river pathways

Date: 2026-10-08. Exploratory continuation after the Krycklan water-state results.

## Question

Does a fixed mapped river arrangement have a different effective distribution of
monitored incoming water at high versus low receiver flow? Does the corresponding
DOC mixing summary change in the same way in elongated and broad networks?

Preserve the preceding 32 ST357 monitored frontiers, original whole-network
classes, source ordering and original cropped river paths. Do not recluster using
DOC or flow. No new predictor is fitted and no hidden-role DOC is opened. The
Krycklan confluences are separate case evidence, not assigned ST357 form classes.

## Hydro-only primary panel

For each fixed receiving network use every month between its first and last
original permitted common-DOC month. Low/high receiver-flow terciles use all
observed positive receiver discharge in that span, including months without DOC.
Participation descriptors require observed positive discharge at every fixed
monitored source and the receiver. Missing or nonpositive flows are excluded and
counted; no zero-share dry branches or missing flow are imputed.

Monthly source fractions are Q_j/sum(Q_sources). Calculate:

- effective monitored contributors, 1/sum(w_j²), and its fraction of the fixed
  source count;
- weighted source-to-receiver mean path and path SD;
- path SD normalized by the unchanged area-weighted mean path;
- common-corridor length divided by the current weighted mean path;
- monitored-source sum / receiver discharge, retained without clipping.

These describe participation among monitored sources, not whole-network branch
activation or parcel travel times. In particular, the physical source count and
path lengths do not change when the weights change.

Eligibility: at least 24 complete positive-flow months, three years, and six low
and six high months. Estimate raw high-minus-low changes and adjusted changes
from one within-network intercept, low/high indicators, annual sine/cosine and
linear-time regression. Report every eligible network and the exclusion ledger.
Static area weights are the unchanged reference, not substituted for measured
flow weights. Different local flow states need not occur in the same month.

## DOC-supported secondary panel

Use only the original common-DOC station-months and explicitly permitted source
training cells. On complete positive-flow dates, project all branch DOC and the
dynamic mixture and receiver onto one intercept, annual harmonic and linear-time
design. Require 24 dates and three years for the full projection, and at least
five low and five high dates for the state contrast.

Compute fixed-state-mean-weight mixture variance reduction relative to the
flow-share-weighted branch variances. Retain dynamic-mixture and outlet absolute
SD and their ratio. Decompose the high-minus-low fixed-weight summary by all-order
substitution of the correlation matrix, source SD vector and mean water fractions.
These substitutions are mathematical summaries, not causal interventions or
carbon removal. Report missing secondary contrasts as unavailable.

## Comparison and repeatability

Report all three original forms and all eligible networks. The primary change is
effective-contributor fraction; path dispersion and shared fraction locate the
structural participation change. Summaries give each network equal weight and
use 5,000 complete-overlap-system bootstrap draws, retaining nested receivers.
Keep the broad-minus-elongated contrast as an observational grouping, not an
independent matched shape effect. Intervals are conditional on the individual
network fits and measured hydrological records, and do not replace year-level
measurement uncertainty. A class with fewer than three represented systems has
no reported bootstrap interval.

Produce a complete reproducible panel, group contrasts, English/Chinese figures,
and a research decision connecting static river form, flow participation and DOC
coordination. Preserve the prior geometry, DOC reconstruction and field analyses.
