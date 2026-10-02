# Spatial transfer section draft

## Regional dynamics and target-station calibration

Spatial transfer failed when the model was asked to infer a new station's
concentration level from regional descriptors alone. We therefore separated
the problem into two operations. First, a source-regional expert was trained
on stations outside the E3 target set to transfer shared seasonal and
hydroclimatic variation. Second, a station-specific residual was estimated
from a small number of target-station observations and added in log1p space.
The query cells were fixed before applying any support level, so the K curve
measures calibration rather than changes in test composition.

On the 2,316-cell paired query, mean absolute error decreased monotonically
from 2.466 mg/L with no target support to 2.320, 2.160, and 2.023 mg/L with
K=1, 3, and 5 support observations, respectively. Five support observations
therefore reduced error by 17.93% relative to the no-support regional expert.
The station-clustered bootstrap difference at K=5 was -0.419 mg/L (95% CI
[-0.820, -0.141]). Shuffling residual corrections across target stations
removed the gain, indicating that the improvement is station-specific rather
than a consequence of simply exposing more labels.

The benefit was heterogeneous. Across 43 held-out stations, 34 improved at
K=5 and the median station-level reduction was 2.5%. Stations with larger
absolute no-support bias tended to obtain larger K=5 gains (Pearson
correlation approximately 0.92). This supports a two-part interpretation:
regional features transfer temporal and hydroclimatic variation, whereas a
few local observations correct the target station's concentration baseline.

## Comparison with graph and tree alternatives

Applying the same support correction to the existing five-seed KGML product
reduced its E3 MAE from approximately 2.823 to 2.544 mg/L, but it remained
below the source-regional expert. Upstream-message and no-message variants did
not separate reliably in this spatial holdout. A final learner comparison
under the same source pool, support set, and nested selection protocol gave
outer K=5 MAE values of 2.024 for ExtraTrees, 2.048 for random forest, and
2.086 for histogram gradient boosting. The existing five-seed ExtraTrees
product (2.023 mg/L) therefore remains the production result; replacing the
tree learner did not provide a new gain.

These results position the current model as a conditional spatial-adaptation
method rather than a universal graph-superiority claim. The strongest
improvement comes from combining regional dynamics with a small amount of
target-station support. The remaining error is concentrated in local station
state that is not identifiable before support observations become available.

## Main spatial-transfer table

| Target support K | MAE (mg/L) | Relative reduction |
|---:|---:|---:|
| 0 | 2.466 | 0.00% |
| 1 | 2.320 | 5.90% |
| 3 | 2.160 | 12.41% |
| 5 | **2.023** | **17.93%** |

## Figure references

- K curve and bootstrap: `regional_residual_product_v1/k_curve.png`.
- Station mechanism: `adapter_comparison_v1/station_mechanism.png`.
- River-network map of station-level gains:
  `station_transfer_map_v1/station_transfer_map.png`.

