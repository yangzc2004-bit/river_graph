# Research decision: two-year causal memory

## Results

Nine source-validation packages (142/143/144 ×42/43/44) are complete. Only
the existing recurrent window changed from12 to24 months. All dimensions,
initial states, environmental reference/OOF arrays, input normalization,
30-epoch/patience5 training settings and native tail objective were unchanged.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained12-month complete |1.769053|9.075420|-0.638608|
| New24-month complete |1.772521|9.077504|-0.622468|
| Retained12-month neural-only |1.772035|9.053055|-0.625464|
| New24-month neural-only |1.773103|9.051231|-0.606989|
| Same24-month weights, inference truncated to12 |1.827763|9.542057|-0.932487|
| Strong station-hidden trees |1.866362|9.394458|-0.611014|

Complete-model gain over the retained complete model is-0.196%
[-0.729%,0.381%]: one of three partition averages and five of nine packages
improve. Complete Q90 gain is-0.023% [-0.371%,0.277%]. Neural-only gain over
the matched retained neural-only model is-0.060% [-0.710%,0.640%]. These
5,000-draw whole-station intervals do not establish an overall upgrade.

## What the history diagnostic answers

The24-month neural model beats its own same-weight12-month inference
truncation by2.991% [1.297%,4.787%]; its Q90 gain is5.144%
[3.128%,8.131%]. All three partition averages are positive. The model therefore
uses the extra preceding months after training. Truncation is a distribution
change for weights trained with24 months, not a matched12-month trained
competitor. This sensitivity must not be presented as a2.99% performance
improvement over the actual retained model. Training on24 months still produces
essentially the same receiving-site error as the retained12-month procedure.

Keep the12-month portable release. Close the window change without a36-month
scan or geographical/external rerun. Older-history sensitivity is a completed
diagnostic, not evidence for adopting the longer model or a physical residence
time. The earlier geographical/external/temporal results remain unchanged.

## Next representation question

Inspection of the retained encoder confirms that native DOC fitting updates
the final self layer and ecological encoder, while its first self layer stays
frozen. The next source study will unfreeze both existing self layers and the
ecological encoder at the same small encoder learning rate. This tests whether
DOC fitting needs to adapt the earlier input representation. Keep the original
12-month horizon and all other settings; do not carry forward failed auxiliary
heads or change graph depth. A separate plan precedes any such fitting.

## Verification

All nine native/integrated predictions and the same-weight truncation diagnostic
replay bitwise. Source OOF population, identical parent predictions, inputs,
initial weights, parameter count and query identities were checked. The new
window tests cover padding, older-history influence, future exclusion,
hidden-label isolation and exact saved prediction replay.969 tests pass, two
skip; Ruff and the historical audit pass. Source-derived Q90 thresholds and
equal-partition/seed weighting are unchanged. Figures were generated and the
PNG inspected. Large fits remain local; only related results are listed for
submission, with Git writes unavailable under the execution policy.
