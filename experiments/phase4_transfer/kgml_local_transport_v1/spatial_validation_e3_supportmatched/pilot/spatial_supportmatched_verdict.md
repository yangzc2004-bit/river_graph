# Support-matched spatial validation verdict

The validation block was selected from source stations using only the graph and
observation masks. Its visible-upstream fraction (0.4033) matches the frozen
E3 test fraction (0.4034), and it contains 7 supported and 13 unsupported
stations. The frozen E3 test stations and labels were not used for selection.

## Validation block

| arm | MAE |
|---|---:|
| RF-context / exact zero-message null | 2.52684 |
| context + upstream message residual | 2.52766 |
| context + both-direction residual | 2.53496 |

The upstream branch is 0.032% worse than RF-context on the validation block;
the paired station bootstrap interval is [-0.00184, -0.00001] mg/L. The
both-direction branch is 0.321% worse. Model selection on this block therefore
chooses the RF-context null.

## Frozen E3 test (reported once)

| arm | MAE |
|---|---:|
| RF-context / exact zero-message null | 2.65364 |
| context + upstream message residual | 2.65257 |
| context + both-direction residual | 2.65186 |

The upstream branch improves the test MAE by only 0.040% (bootstrap CI
[0.00002, 0.00215] mg/L), while the both-direction interval crosses zero. The
test ordering does not overturn the validation result: the graph correction is
small and unstable across spatial splits.

## Interpretation

Matching the validation block to the test support structure removes the earlier
validation-design mismatch, but it does not produce a robust spatial gain. The
confirmed spatial model remains RF-context. The upstream residual branch is
retained as a conditional diagnostic for stations with visible upstream support,
not as a general spatial-extrapolation improvement.
