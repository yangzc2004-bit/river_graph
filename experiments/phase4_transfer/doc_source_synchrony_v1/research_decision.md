# Research decision: source DOC residual synchrony

## Finding

The source-only audit is complete. It uses source training DOC and seed42
station-blocked OOF environmental residuals in partitions142/143/144. Receiving
DOC values and all pH/conductance labels are absent from the calculation.

| Donor group | Lag (months) | Real correlation | Calendar-preserving shuffle | Difference |
|---|---:|---:|---:|---:|
| Nearest ecological sources |0|0.166158|0.042386|0.123772|
| Nearest ecological sources |1|0.073686|0.014753|0.058933|
| Nearest ecological sources |3|0.031975|0.010589|0.021387|
| Distant ecological sources |0|-0.019079|-0.005942|-0.013137|

For nearest sources, real-minus-shuffle is positive in all three partitions at
every tested lag. Same-month differences are0.108/0.117/0.146. At lag0, an
average201 of232 source recipients have at least one usable nearest pair,
with an average533 eligible directed pairs per partition. A usable pair needs
12 common observed months and nonconstant real/control series. Distant-source
coverage is lower (107 recipients and329 pairs), so the nearest-versus-distant
contrast also changes population. The real-versus-shuffle comparison uses
exactly matched station pairs and observed-month support.

Correlation is averaged across donors within recipient, recipients within
partition and then equally across partitions. Source stations/pairs/partitions
overlap; these are descriptive associations. They are not independent
replications, a physical transport coefficient, or a measured receiving-site
prediction gain. The year shuffle preserves each donor's seasonal distribution
and availability, so its difference is consistent with additional time-aligned
source information beyond residual seasonality.

## Decision

Proceed to the simple, separately planned source-innovation prediction probe in
`doc_source_innovation_transfer_v1`. Keep the current complete model and add a
small calibrated correction from ecologically similar sources observed in the
same month. First compare with a causal earlier-season source control and
measure receiving-site support. Do not train another attention module before
checking whether this information improves predictions.

The audit's unrestricted year shuffle is only a descriptive control. Prediction
controls must use earlier source values and matched availability, avoiding a
future observation being moved into an earlier key. Frozen source preprocessing
and source statistics remain fixed for the inference perturbation contract.

Source innovations use existing, available source DOC; receiving stations remain
entirely water-quality-free. This is a new timing-information mechanism rather
than another scan of depth, epochs, encoder scope or chemistry-loss weights.
Monthly weather inputs remain a separate access task. Previous geographical
and independent-basin outcomes are unchanged and do not select this method.

## Verification

Parent artifacts and source roles were verified; source station IDs are disjoint
from receiving roles. Two new tests verify hidden-cell exclusion, seasonal
availability preservation, lag alignment and unavailable/constant cases.973
tests pass, two skip; Ruff and the historical audit pass. An initial standalone
test import needed the repository's explicit scripts-path convention; its failed
log is preserved, and no data/algorithm changed. Source-derived figures were
generated and the PNG inspected; cropped labels and a panel-label overlap were
repaired, retaining the initial layout. No prediction model was fitted or adopted by
this diagnostic. Related files are saved for submission; large caches remain
local and Git writes remain unavailable under the environment policy.
