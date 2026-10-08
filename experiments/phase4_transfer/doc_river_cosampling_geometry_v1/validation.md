# Validation record

Date: 2026-10-08

## Source and analytical checks

- Accepted source-role activity months reconcile to the frozen native-DOC
  dataset. The verifier checks that excluded target-role months are absent.
- Source selection is invariant to DOC values and row order, respects the
  original routed frontier, and preserves short-record metadata. Tests also
  cover year-month grouping, missing sampling clocks and routing ancestry.
- Six focused tests passed. Full repository suite: 1,325 passed, two skipped,
  eight warnings, 43.62 seconds. This preceded the final visual label and
  receiver-nesting metadata refinements; those refinements have their actual
  rendered inspection and independent membership checks below.
- Ruff passed across the repository. Historical artifact audit exited zero;
  the existing 85 no-sidecar historical predictions and absent-dataset caveats
  remain unchanged.
- Final main verifier exited zero with full table replay. It independently
  reconstructed receiving-reference SD and excursion counts; maximum receiving
  log-reference discrepancy was 2.78e-16.
- Final disjoint-pair verifier exited zero. All 3,800 source-pair date
  intersections, dense-date counts and non-nested routing relationships were
  independently reconstructed. All 105 receiving-pair nesting flags were
  checked against mapped catchment membership.
- The 91-pair non-nested shortlist is a subset of the 105 metadata opportunities.
  Its ordering uses observation availability and area similarity, never DOC
  magnitudes or effect direction.

## Figures and delivery

- All four PNG figures were inspected. The English overview initially had
  adjacent category labels; these were wrapped and the final bilingual
  overviews inspected again. The single-river interval is explicitly absent.
- Observed campaign selection uses the densest receiver and year-month by
  metadata, with station/month tie breaks. It selects Loch Vale, July 1996,
  five dates. Both first and last displayed dates are visible.
- PDFs use the same Matplotlib canvas; they were not separately rendered.
- The three-card inline Sources payload contains approved aggregate counts and
  a labeled ten-row priority preview. Its renderer succeeded after adding the
  required preview note. No local absolute paths or credentials are embedded
  in that payload.

## Scientific scope

No new broad-versus-elongated effect is estimated from the dense seasonal panel,
because that panel contains only one mainstem-dominated receiving river. Broad
forms do have short co-sampled records; absent signal eligibility is not a zero
DOC effect. Gauge-pair combinations and repeated receiver pairs are not counted
as independent experimental rivers. No model training or transit-time fitting
was performed, and historical experiment products were preserved.
