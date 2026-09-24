# Stage 1 route decision (2026-09-24)

## Data-gate status

The local ST357 three-analyte availability and QC checks passed. The two
external HUC8 candidates screened so far did not meet the frozen DOC threshold:

- `02040104`: 7,064 QC-accepted DOC station-months;
- `02030103`: 3,242 QC-accepted DOC station-months.

The requirement is at least 10,000 station-months for each analyte. These DOC
counts use analyte-specific active masks on the union of valid Stream stations.
The all-three station intersection is a separate diagnostic. The pH and
conductance bulk responses contain the provider's `INCOMPLETE DATA` marker;
their coverage is unknown, not zero. Neither candidate has a completed graph,
feature, split, and provenance audit.

**These two screens do not establish that all external basins are
ineligible. External validation remains pending and unverified.** No external
basin has been selected or evaluated with a model.

## Internal fallback

The user-authorized ST357 multi-HUC6 fallback permits internal data and
baseline work while external validation remains pending. Results may describe
held-out HUC6 tasks within ST357 when the corresponding protocol is satisfied.
They cannot establish replication in an independent external basin. The
original charter, endpoints, and Phase 0–3 artifacts remain unchanged.

Further external work must use a versioned availability record and complete
the frozen data gate before selecting a case or inspecting its model results.
The current two failed screens do not authorize lowering those thresholds.

## Current execution boundary

The bounded current work is the **Stage 2A same-analyte support diagnostic**
and its corrections. It uses privileged target-analyte observations outside
the entire target HUC6 to construct a reference climatology. It is not the
leave-one-analyte transfer experiment. Within the hidden HUC6, support and
query tasks use only the frozen largest-component rows; the remaining HUC6
rows also remain excluded from source labels.

Exploratory query results have already been inspected. The correction record
and descriptive v2 specification acknowledge that exposure; the diagnostic
cannot be reclassified as preregistered confirmation.

**The overall Stage 2 gate is not evaluated.** Required learned controls,
missingness regimes, and information-source comparisons remain outstanding.
Stage 3 transfer training is not authorized. In particular, the output scale
of a target-unseen K=0 model remains unresolved: no target statistics or
unregistered target-specific priors may be invented to fill that gap.

The existing DOC-only manuscript remains the fallback if the internal
cross-analyte signal also fails.
