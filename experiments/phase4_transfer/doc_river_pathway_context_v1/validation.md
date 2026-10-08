# Validation

Date: 2026-10-08

- Eight new unit tests pass: endpoint cropping, shared suffixes, partial lake
  exposure, source-order invariance, invalid/disconnected paths, exact receiving
  variance, time-varying shares applied before projection, constant signals,
  duplicate dates and outcome-independent comparison selection.
- The complete repository suite reports **1,299 passed, two skipped, eight
  warnings**. Ruff reports all checks passed.
- The original-route and shortest-route analyses replay fully, including the
  5,000 system-bootstrap calculations, 32 physical receiving panels and their
  82/84 variance budgets. Source roles and physical receiver uniqueness agree.
- Receiving variance identities are independently recomputed directly from
  saved anomaly series. A separate resampling implementation concatenates
  complete system records and reproduces the high-coverage form contrast and
  its interval, without using the analysis contrast helper.
- All 12 English/Chinese PNG figures were actually viewed. The final flow plot
  uses readable decimal ticks; the main receiving figure shows SD ratios,
  retaining the algebraic variance contributions in the analytical tables.
  PDF exports use the same plotting canvas. No AI image is used as evidence.
- Historical `audit_artifacts.py --verify` exits zero. Its pre-existing
  missing-sidecar and unavailable historical-dataset qualifications remain;
  this does not turn those records into complete provenance verification.

The analysis is ready within its exploratory scope. The main qualification is
scientific: the high-coverage whole-form contrast is a positive lead, while
regional overlap and independently observed forms remain inadequate to isolate
shape. No new training, test-driven model tuning or external-validation claim
is introduced.
