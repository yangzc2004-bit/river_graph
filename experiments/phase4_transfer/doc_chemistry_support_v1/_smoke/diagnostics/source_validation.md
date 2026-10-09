# Source-validation support-transfer diagnosis

Reconstructed 1 completed packages; all saved source-validation MAEs reproduce within 1e-12. Only validation station DOC labels were used. Target prediction files were not read.

This is a descriptive diagnosis on reused selection data, not new confirmation. Means average seeds within partition and then partitions equally. Station summaries average seeds first; stations repeated across partitions remain descriptive station-partition pairs. MAE tables use all fixed validation queries; support-transfer station diagnostics use auxiliary-active queries.

## Source-validation MAE

| Model | K0 | K5 |
|---|---:|---:|
| neural_chemistry_chemistry_aug | 1.879005 | 1.713008 |
| neural_chemistry_integrated_chemistry_aug | 1.879005 | 1.709714 |
| neural_chemistry_integrated_legacy | 1.879005 | 1.714481 |
| neural_chemistry_legacy | 1.879005 | 1.717345 |
| point_integrated_legacy | 1.900843 | 1.735844 |
| point_legacy | 1.900843 | 1.742710 |
| tree_chemistry_chemistry_aug | 2.112339 | 1.752934 |
| tree_chemistry_legacy | 2.112339 | 1.755937 |
| tree_prior_legacy | 2.124283 | 1.775575 |

## What changes with five support measurements

The integrated chemical decoder's validation advantage over the general point model is 0.021838 mg/L at K0 and 0.021364 mg/L at K5. Chemical coordinates add 0.004766 mg/L of K5 improvement over the legacy coordinates. These contrasts use identical fixed validation queries; they are not held-out confirmation.

The integrated decoder uses nonzero ecological mixing in 0/1 packages at K0 and 1/1 at K5. Increasing gamma reduces the share of the temporal/chemical expert before station calibration. Accordingly, a smaller K5 decoder contrast can reflect both support recalibration and changes in the ecological mixture, rather than failure of the chemistry inputs alone.

## Support-query residual transfer

Positive residual means underprediction. The adapter transfers a shrunk support mean plus a linear ridge shape. A chemistry correction common to support and query can be partly cancelled by the mean-residual recalibration: its log-scale contrast includes query correction minus alpha times mean support correction. That is an accounting property, not proof of why a performance contrast changes.

| Model | Support/query mean correlation | Opposite signs | Mean absolute mean mismatch | Median shape/query correlation | Stations worsened |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_chemistry_aug | 0.883 | 20.4% | 0.110 | 0.090 | 37.0% |
| neural_chemistry_integrated_chemistry_aug | 0.890 | 20.4% | 0.110 | 0.073 | 33.3% |
| neural_chemistry_integrated_legacy | 0.890 | 20.4% | 0.110 | 0.021 | 33.3% |
| neural_chemistry_legacy | 0.883 | 20.4% | 0.110 | 0.054 | 33.3% |
| point_integrated_legacy | 0.886 | 20.4% | 0.111 | 0.044 | 31.5% |
| point_legacy | 0.877 | 25.9% | 0.112 | 0.091 | 29.6% |
| tree_chemistry_chemistry_aug | 0.908 | 22.2% | 0.110 | 0.052 | 29.6% |
| tree_chemistry_legacy | 0.908 | 22.2% | 0.110 | 0.035 | 31.5% |
| tree_prior_legacy | 0.908 | 18.5% | 0.115 | 0.049 | 33.3% |

Chemical distance uses the frozen two-dimensional whitened chemical representation, only auxiliary-active support/query dates, and nearest same-station support. It is not a physical chemical distance or a causal estimate. The station table records gaps, support coverage, shape magnitude and residual signs. No kernel, bandwidth or new predictive operator was fitted.

## Interpretation and one next operator hypothesis

Support and query station-mean log residuals correlate at 0.890; their signs differ for 20.4% of station-partition pairs. The useful mean correction should therefore be retained. The weaker component is within-station shape: median shape/query residual correlation rises from 0.021 to 0.073 with chemical coordinates.

Chemical distance is only weakly associated with support-mean residual mismatch (median within-station correlation 0.102). These diagnostics do not justify a distance-based reliability gate or a claim that distant chemical states cause failure.

If the augmented linear basis does not generalize, a focused operator test would retain the current base and shrunk station mean but replace the chemical linear extrapolation with a bounded, similarity-weighted interpolation of centered support residuals. Use the same frozen two chemical coordinates, a bandwidth derived from source-station chemical distances, and the matched availability-only control. This tests nonlinear support-to-query transfer without another neural architecture or loss sweep. It is a hypothesis motivated by weak shape transfer, not an established distance mechanism or an authorization to fit a new model.
