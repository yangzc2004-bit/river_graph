# Source-validation support-transfer diagnosis

Reconstructed 9 completed packages; all saved source-validation MAEs reproduce within 1e-12. Only validation station DOC labels were used. Target prediction files were not read.

This is a descriptive diagnosis on reused selection data, not new confirmation. Means average seeds within partition and then partitions equally. Station summaries average seeds first; stations repeated across partitions remain descriptive station-partition pairs. MAE tables use all fixed validation queries; support-transfer station diagnostics use auxiliary-active queries.

## Source-validation MAE

| Model | K0 | K5 |
|---|---:|---:|
| neural_chemistry_chemistry_aug | 1.769764 | 1.605068 |
| neural_chemistry_integrated_chemistry_aug | 1.769265 | 1.591733 |
| neural_chemistry_integrated_legacy | 1.769265 | 1.598153 |
| neural_chemistry_legacy | 1.769764 | 1.608525 |
| point_integrated_legacy | 1.804067 | 1.604602 |
| point_legacy | 1.804428 | 1.616446 |
| tree_chemistry_chemistry_aug | 1.910344 | 1.604882 |
| tree_chemistry_legacy | 1.910344 | 1.612055 |
| tree_prior_legacy | 1.944702 | 1.631860 |

## What changes with five support measurements

The integrated chemical decoder's validation advantage over the general point model is 0.034802 mg/L at K0 and 0.006449 mg/L at K5. Chemical coordinates add 0.006420 mg/L of K5 improvement over the legacy coordinates. These contrasts use identical fixed validation queries; they are not held-out confirmation.

The integrated decoder uses nonzero ecological mixing in 2/9 packages at K0 and 8/9 at K5. Increasing gamma reduces the share of the temporal/chemical expert before station calibration. Accordingly, a smaller K5 decoder contrast can reflect both support recalibration and changes in the ecological mixture, rather than failure of the chemistry inputs alone.

## Support-query residual transfer

Positive residual means underprediction. The adapter transfers a shrunk support mean plus a linear ridge shape. A chemistry correction common to support and query can be partly cancelled by the mean-residual recalibration: its log-scale contrast includes query correction minus alpha times mean support correction. That is an accounting property, not proof of why a performance contrast changes.

| Model | Support/query mean correlation | Opposite signs | Mean absolute mean mismatch | Median shape/query correlation | Stations worsened |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_chemistry_aug | 0.875 | 21.0% | 0.104 | 0.131 | 24.7% |
| neural_chemistry_integrated_chemistry_aug | 0.883 | 22.2% | 0.104 | 0.118 | 27.2% |
| neural_chemistry_integrated_legacy | 0.880 | 23.5% | 0.104 | 0.063 | 27.2% |
| neural_chemistry_legacy | 0.875 | 21.0% | 0.104 | 0.078 | 24.7% |
| point_integrated_legacy | 0.885 | 22.8% | 0.104 | 0.073 | 28.4% |
| point_legacy | 0.879 | 21.6% | 0.104 | 0.066 | 24.7% |
| tree_chemistry_chemistry_aug | 0.909 | 24.1% | 0.104 | 0.112 | 30.9% |
| tree_chemistry_legacy | 0.909 | 24.1% | 0.104 | 0.056 | 36.4% |
| tree_prior_legacy | 0.907 | 22.2% | 0.106 | 0.057 | 32.1% |

Chemical distance uses the frozen two-dimensional whitened chemical representation, only auxiliary-active support/query dates, and nearest same-station support. It is not a physical chemical distance or a causal estimate. The station table records gaps, support coverage, shape magnitude and residual signs. No kernel, bandwidth or new predictive operator was fitted.

## Interpretation and one next operator hypothesis

Support and query station-mean log residuals correlate at 0.880; their signs differ for 23.5% of station-partition pairs. The useful mean correction should therefore be retained. The weaker component is within-station shape: median shape/query residual correlation rises from 0.063 to 0.118 with chemical coordinates.

Chemical distance is only weakly associated with support-mean residual mismatch (median within-station correlation 0.047). These diagnostics do not justify a distance-based reliability gate or a claim that distant chemical states cause failure.

If the augmented linear basis does not generalize, a focused operator test would retain the current base and shrunk station mean but replace the chemical linear extrapolation with a bounded, similarity-weighted interpolation of centered support residuals. Use the same frozen two chemical coordinates, a bandwidth derived from source-station chemical distances, and the matched availability-only control. This tests nonlinear support-to-query transfer without another neural architecture or loss sweep. It is a hypothesis motivated by weak shape transfer, not an established distance mechanism or an authorization to fit a new model.
