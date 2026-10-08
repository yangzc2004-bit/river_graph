# Observation audit refinements

Date: 2026-10-09. Written after inspecting the downloaded schemas and record
issues, before producing the event audit. This is a suitability study, not a
pre-registered morphology-effect test.

The Arctic chemistry entity contains no discharge. Its EML methods explicitly
refer to separately published Arctic LTER discharge. Both the public legacy
metadata endpoint and the current catalog's public search endpoint returned
HTTP 403. Stop retries against those unchanged endpoints. Storm peak Q and
integrated water yield are event summaries, not substitutes for the complete
hydrograph. No event satisfies the planned joint DOC/positive-flow requirement
using the objects acquired here.

The combined logger grid is not consistently the DOC sampling grid. Report both
the modal spacing of all timestamps and that of finite DOC values. As a record
coverage diagnostic, use occupied nominal 15-minute bins, retaining the
published storm boundaries and their original local clock. Include both
boundary gaps in the maximum-gap check. This prevents a faster logger grid from
inflating coverage. It neither creates new DOC measurements nor resolves
uncertain storm timezones. Bare-date boundaries and disagreement between
published duration and elapsed timestamps are reported explicitly.

At duplicate site/timestamps, retain the sole finite DOC value if all finite
alternatives agree exactly. Where finite alternatives disagree, preserve the
alternatives in a ledger and leave canonical DOC missing. Do not average or
select a preferred alternative. Do not trim extreme concentrations simply
because they are large. Preserve the author-provided major-flood note.

Investigate identical varying numerical sequences across years. Compare ordered
finite DOC records within each catchment; report exact shared-value counts and
the longest exact sequence beginning with at least three distinct values in
its first ten observations. Sequences of at least 24 values flag the later
site/year for source resolution. This is a deliberately conservative record
reuse diagnostic motivated by the inspected archive, not a biological finding
or proof of its cause. Both associated years and timestamp spans remain visible.
Whole later site/years are set aside from provisional event candidates until
the source assembly or processing is explained. All 69 original storm intervals
remain in the output, including exclusions. A DOC record candidate is not a
complete flow-DOC event or an independent morphology replicate.

The Kuparuk monitoring coordinate and drainage area change in 2022 according to
the published study. Report monitoring coordinates by year; do not treat six
years as one invariant river frontier. Future geometry extraction must use the
appropriate outlet for each period.

For Kervidy-Naizin, distinguish raw DOC from laboratory-corrected DOC. The
source README excludes corrected DOC before October 2020 because of earlier
sensor drift, despite the dataset title beginning in 2010. The chemistry table
has no flow column. Its clock is explicitly UTC. The distributed corrected DOC
has modal spacing of 15 minutes in each available year; the README's 10-minute
instrument setting is reported separately, not inferred as the distributed
resolution. Daily Krycklan/Finland data and
NEON fDOM retain their original temporal resolution and measurement identity.

No morphology is inferred from a DOC trace, lake label, repeated value sequence
or instrument coverage. No peak-width comparison is released from this audit.
