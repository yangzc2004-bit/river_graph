# Updated external replication: current-source DOC release

## Decision

Complete this fixed release and retain its geographically supported source-
attention mechanism. External HUC02040104 does not establish an additional
cell-weighted gain over the preceding complete release. No external tuning
or per-station/K winner selection follows this result. Continue the already
authorized temporal compatibility check and manuscript integration. Further
performance development returns to source training/validation roles.

## Fixed experiment and exposure

Five source fits use303 ST357 training and54 source-validation stations,
seeds42–46. The model combines the retained station-hidden environment tree,
ecological encoder,12-month observation-aware GRU and twenty-candidate
current-availability source attention. Wider pools and source-state key
channels were negative source probes and are excluded. All five source fits
and serialized exports pass replay. The receiving130 stations are disjoint
from ST357, with520 months,6,514 observed DOC cells and5,864 fixed query cells.

This external case was inspected for the preceding release. Its original
selection was based on availability; the present comparison is an updated
external replication, not a newly blinded test. The new architecture was
developed in source roles and confirmed with ST357 geographical withholding.
The point generator reads only saved source states and label-free external
inputs before saving point products and source-calibration policies.

## Main K0 results (mg/L)

| Procedure | Cell MAE | Station-equal MAE | Bias |
|---|---:|---:|---:|
| Earlier full model |0.984065|0.907013|-0.410810|
| Strong station-hidden trees |0.840835|0.904738|-0.034247|
| Preceding complete release |0.914994|0.841214|-0.434428|
| Current-source complete release |0.910577|0.821436|-0.441794|

Paired station bootstrap uses5,000 draws, keeping all months of each station
together. Current-source versus preceding release gives0.483% reduction
[-0.393%,1.560%], insufficient to establish an incremental cell-MAE benefit
in this basin. Versus the earlier full model, cumulative reduction is7.468%
[5.075%,9.543%]; this includes earlier station-hidden residual changes and
must not be attributed solely to attention. Versus strong trees, the point
MAE is8.294% worse [-19.440%,4.435% relative gain]. The interval crosses zero,
so neither equivalence nor a general superiority claim is warranted. The
station-equal point estimate favors the new release, a different estimand
reflecting unequal observation counts; it does not replace the primary one.
External RMSE is1.376907mg/L and R² is-0.129111, versus1.240233 and0.083919
for strong trees. The negative R² makes the remaining external reconstruction
gap explicit alongside the absolute-error comparison.

Source validation ensemble MAE is1.791012 versus1.828532 for the preceding
release:2.052%[0.481%,3.693%]. It selected the fitted procedure and is not
an independent confirmation. Native-only and complete predictions coincide
in this deployment because every source validation fit selects zero static
memory fusion.

## Support, high DOC and uncertainty

Fixed-query K0/1/3/5 MAE is0.925335/0.925335/0.828634/0.746269. K1 shrinkage
is selected as zero on source validation. K5 improves19.351% against its own
fixed-query K0, while its incremental difference from the preceding K5
release is-0.035%[-0.869%,0.792%]. Do not combine versions by K.

The source-derived Q90 group has16 cells at11 stations; all tools miss these
high-DOC events, and tail estimates remain unstable. New Q90 MAE9.577730
versus preceding9.566088 shows no tail improvement. Mean bias remains
negative and is more pronounced than the tree reference.

Source-validation empirical90% intervals yield83.267% external coverage
and median width2.737396mg/L. The preceding release gives84.925% and2.889311;
strong trees92.508% and3.536959. Narrower intervals trade away coverage.
Calibration used source validation that also selected early stopping, so
there is no conformal guarantee. K5 coverage is87.381%, width2.397276.

Across receiving observed cells, current-source availability is82.208%
(source-seed mean), with1.684 valid donors on average. Mean zero-prior
attention mass is0.8005. These diagnostics describe the limited use of the
source bank in this case; they do not identify the cause of concentration
underprediction or turn weights into physical transport coefficients.

## Verification and delivery

All prior prediction rows are preserved exactly. New point/scoring hashes,
source export manifests, three arbitrarily named receiving-site replays,
hidden-chemistry and future-hydro input contracts pass. The full-grid PNG
was actually inspected, including its paired intervals, fixed-query curve
and coverage-width panel. Saved states and old results remain separate.

The portable five-member release is saved in the source-fit directory as
release.json, pointing to five local serialized member exports and containing
source-only support/interval policies. A future external test should use
a new independently constructed case. Strengthen low-concentration regime
transfer and high DOC in source-role research before another geographical
confirmation; keep this external version's predictions unchanged.
