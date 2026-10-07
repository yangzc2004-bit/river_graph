# Hourly paired flow and optical-DOC response

## Question

Does a downstream response simply reproduce an upstream wave later, or does
mixing broaden and reshape it? This is the timing link in the river-form study:
branch paths and junction arrangement → coincidence or separation of incoming
signals → outlet peak and variation. No new prediction model is trained.

## Observations and scope

Use the preserved Turbolo workbook and its audited paired product: San Nicola
(upstream) and Fitterizzi (downstream), 422 matching hourly timestamps in 15
continuous segments. These are author-selected windows, not 15 independent
storms. Retain the original event IDs and both excluded conflicting timestamps.
Source-clock timezone is unspecified and stays unspecified.

All 15 segments remain in the coverage ledger. The seven spanning at least
24 elapsed hours are the main coverage subset; shorter segments are diagnostics.
Segments with several half-height lobes or author-event IDs are identified as
multi-wave records, not silently converted to separate independent events.
This exploratory plan follows inspection of the parent plots and flow coverage.
The regular one-hour peak offsets motivate an explicit shifted-flow-shape check.

## Source-specific measurement distinction

The source article describes corrected fDOM-derived DOC. Section 2.3 states that
DOC at turbidity >600 FNU was estimated with discharge and preceding rainfall
regressions. This can induce flow–DOC timing dependence. The workbook provides
no explicit row-level reconstruction flag. Use >600 FNU as the *reported
retrieval regime*, not as a complete flag for all reconstructed rows. The
approximately 40-FNU level indicates optical attenuation relevance, not a
reconstruction threshold. Neither low-turbidity nor corrected DOC is laboratory
truth. Intervening inflows and land use prevent attribution solely to geometry.
The source laboratory calibration spans up to 3.88 mg C/L. Also record whether
an optical peak exceeds that range; this diagnostic does not select new peaks
or certify the accuracy of estimates below 600 FNU.

Source: [Senatore et al. (2023), sections 2.2–2.4](https://doi.org/10.1029/2022WR034397).
Data: [public Turbolo record](https://researchdata.cab.unipd.it/803/).

## Calculations fixed for this analysis

1. **Peak clock:** earliest/latest sampled maxima at each site. Report downstream
   minus upstream as an interval when maxima tie. Peak clocks have one-hour
   sampling resolution, not tracer travel-time precision. A boundary peak is
   flagged as possibly truncated; separated equal maxima are not one plateau.
2. **Dominant-lobe duration:** half height is minimum within the observed segment
   plus half its range. Find the rising and falling crossings enclosing the
   maximum. Interpolate a crossing only between adjacent observed hourly points;
   never bridge missing hours or extrapolate outside the segment. Record missing
   crossings and multiple lobes. This is a segment-relative waveform width,
   not a baseflow-separated event duration or residence time.
3. **DOC quality sensitivity:** retain all published values descriptively. Record
   turbidity at every tied peak and every sample used for its half-height lobe
   and crossings. If a peak/lobe enters >600 FNU, its timing/width is unavailable
   as an optical-only response. Never replace a excluded peak with the largest
   remaining low-turbidity point or interpolate through excluded values.
4. **Within-site clocks:** compare optical-DOC and discharge maxima, retaining
   optical-quality and boundary flags. Between-site DOC timing is summarized
   separately from flow timing; no flow-based DOC regression is fitted here.
5. **Shifted-flow shape:** at the published peak offset (not a searched best lag),
   align observed upstream and downstream flow values on the source clock.
   Report n pairs, centered correlation, a descriptive least-squares scale and
   offset, and range-normalized fit error. Strong proportionality describes the
   published series; it does not establish independent hydrograph measurements
   or a physical transfer kernel. The archive does not document rating curves.

No p-value or bootstrap treats hourly rows as independent catchments. Report
counts, paired differences and case diagrams, retaining unavailable cases.

## Products and next research decision

Save segment metrics, site metrics, quality flags, paired records, bilingual
figures, a reproducible source receipt and a research decision. Plot all segments
in a diagnostic atlas; show the earliest three main-coverage segments rather
than selecting favorable DOC outcomes. Check synthetic shifted pulses, plateau
peaks, incomplete crossings, gaps and high-turbidity exclusion.

Determine which observed timing statements survive the measurement audit, then
specify the branch–branch–outlet observations needed to relate pathway spread
and shared-trunk length to DOC wave overlap. Keep all three original river-form
classes unchanged. This paired reach is not a replicated form-class experiment.
