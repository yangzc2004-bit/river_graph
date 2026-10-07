# Internal structures within real river outlines

## Scientific question

Do the three existing whole-river outline classes contain different combinations
of junction balance and path-length dispersion? Which operation can those
combinations supply to DOC buffering under a controlled input, and what do the
observed tributary records show?

Keep every previous outline classification and DOC result. This is a new
exploratory structural analysis after seeing the previous mechanism results.

## Geometry first

Start with all 322 distinct complete NHDPlus upstream networks that received an
outline class, including secondary connections. No DOC value selects a network,
cut point or map example. Two junction-free networks remain a separate category.

- Junction balance: the existing median of `1-largest parent area/sum parent
  areas` across mapped confluences. This is a VAA parent-area proxy; it describes
  local junction organization, not the measured flow shares of the whole river.
- Path dispersion: incremental-catchment-area weighted distance SD / mean,
  recalculated from saved reachable channel paths and unique local areas.
- Divide each axis at its geometry-only median in the 322-network cohort,
  producing four **relative structural profiles**, not a new clustering claim:
  less/more balanced junctions crossed with concentrated/dispersed paths.
- Display an additional fixed-reference classification at balance=0.25 and
  path CV=0.5. It is a descriptive sensitivity, not a result-selected replacement.
- Select a real medoid per profile using balance, path CV and log basin area;
  no observation counts or DOC outcomes enter representative selection.

## Isolate relative path dispersion

Give every incremental catchment the same concentration pulse (Gaussian SD=.15),
with positive constant water input proportional to catchment area and total
flow one. Normalize paths by their own weighted mean, so every network has
mean travel delay exactly one. This removes differences in mean exposure/area
from the controlled pulse comparison. Units are relative scenario time, not
observed velocity or days. Compare actual, half and zero path spread at the same
mean delay. Retain constant mean concentration and total anomaly mass.

The balance descriptor and the path distribution are different measurements.
Do not interpret their profile association as an independent causal coefficient
or the controlled pulse comparison as a measured DOC-removal rate.

## Link observations, retaining the measurement scale

Map the previous 22-receiver/59-connection signal study to these profiles without
changing its dates or coefficients. Summarize native mixing variance reduction,
equal-amplitude mixing potential, source synchrony, and measured outlet/mixture
SD ratio. A sampled tributary pair is not the same object as the complete network;
report the alignment of pair balance with whole-network junction balance.

Use 5,000 complete monitoring-system bootstrap draws with receiver-equal means.
Intervals remain unavailable for profiles with one system. Pairwise profile
contrasts are exploratory, with shared systems resampled together. Also map the
existing 205-station flow-response fits for broader descriptive context, keeping
their geographic and hydrologic differences visible; no new DOC fit or tuning.

## Deliverables

Geometry/profile atlas, outline-by-profile count matrix, constant-mean routing
scenarios, observed buffer summaries and an English research decision.
Chinese and English figures use actual channels and basin boundaries. The
research decision must distinguish the new structural measurement, the modeled
operation and the observed DOC relationship.

## Diagnostic extension after first profile summaries

The first observed profiles do not follow the controlled-pulse ordering and have
different basin sizes. Retain their definitions and all 22 receivers. Supplement
the comparison with the alignment of full-network and actually gauged-pair
descriptors, plus continuous balance/path associations with outlet/mixture SD,
adjusting log basin area, source-drainage coverage and the other structural axis.
These four associations are exploratory checks after seeing the profile tables,
not new prediction fits or evidence selected to favor a profile. Report both
native and log-concentration outcomes and leave-system influence.
