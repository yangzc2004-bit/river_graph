# Research decision: time-aligned source DOC innovations

## Results

All nine source-development packages (142/143/144 ×42/43/44) are complete.
No forests or neural models were refitted. The existing complete prediction was
corrected by an ecology-weighted mean of same-month source DOC innovations.
The matched historical arm uses earlier same-calendar-month innovations, with
the same current donor availability. Receiving DOC, pH and conductance are absent
from both libraries. Seasonal source means and ecology preprocessing are fixed
before inference.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete |1.769053|9.075420|-0.638608|
| Same-month source correction |1.763691|9.021610|-0.640270|
| Earlier-season source correction |1.769053|9.075420|-0.638608|
| Strong station-hidden trees |1.866362|9.394458|-0.611014|

Same-month improvement over the actual retained complete model is0.303%
[-0.107%,0.747%], with two of three partition averages and six of nine packages
improving. Q90 improvement is0.593%[-0.050%,1.448%], with one of three
partition averages and four of nine packages improving. All three training-seed
averages improve, but the paired station intervals include zero. The5.50%
advantage over strong trees includes the already retained model's5.21% gain;
it is not the incremental contribution of this source mechanism.

Real alpha is0.25 in partition142,0 in143 and0.5 in144 for all three seeds.
Every historical arm selects alpha0 and reproduces the baseline exactly.
Because zero is available and selection uses this same development panel,
non-worse selected MAE is expected. These intervals do not remove selection
optimism or establish new geographical/external performance.

## Information availability and interpretation

Current source support exists in92–93% of query cells. Requiring an earlier
same-season reading for the matched control reduces this to84–85%, with mean
2.7–3.9 supported donors among at most20 ecological candidates. Weighted
innovation magnitudes average0.24–0.48mg/L. There is considerable observation
support; the small gain is not explained simply by sources being unavailable.

This follows the positive source synchrony finding with a modest predictive
direction. A single pooled scalar cannot adapt the correction to receiving-site
hydrological state or changing donor reliability. That is a research hypothesis,
not a demonstrated reason for the remaining error.

## Decision

Keep the current deployed model, external products and manuscript unchanged.
Do not run geographical confirmation for this scalar probe or scan alternative
neighbour counts/shrinkage constants against the development outcomes.
Make one structured learning test in the existing ecology/GRU residual: use
same-month innovation and support as additional readout information, with
matched historical and availability-only arms. Include an enriched strong-tree
comparison using exactly the same new information. This is source-only mechanism
development; the retained complete model remains the performance comparator.

Training libraries must exclude the query station fold from both donor labels
and donor-reference fitting. Inspection shows that the45 existing complementary
forests each omit one fold, not both a query fold and donor fold. They cannot
directly supply these doubly held-out donor residuals. The new learning study
needs separately recorded pair-fold reference fits; do not silently reuse a
forest trained on the query fold. Its plan is saved separately before fitting.

## Verification

All libraries, selection grids, donor candidates and predictions replay bitwise.
Parent outputs, query cells and source thresholds are unchanged. Tests cover
selected-source extraction, no self donor, earlier-season control dates, future
value/availability perturbations with fixed source statistics, arbitrary receiver
IDs/calendars, zero-alpha equality and save/load.977 tests pass, two skip;
Ruff and the historical audit pass. Source-generated figures were actually
inspected. Large fitted/input caches remain local; related files are saved for
submission because Git index writes are unavailable in this environment.
