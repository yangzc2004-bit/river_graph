# Explicit multi-hop context verdict

The RF-context baseline was augmented with exact directed upstream summaries
for two and three graph hops. The validation block was the support-matched
spatial split; the frozen E3 test was evaluated once after the feature choice.

| upstream context | validation MAE | frozen E3 MAE |
|---|---:|---:|
| 1 hop (RF-context) | 2.52684 | 2.65364 |
| 2 hops | 2.52249 | 2.65964 |
| 3 hops | 2.51755 | 2.66210 |

The extra hops improve the validation block by 0.17% and 0.37%, but degrade
the frozen E3 test by 0.23% and 0.32%. The validation improvement therefore
does not transfer to the terminal spatial holdout. Adding explicit spatial
context depth is not a reliable fix for the current spatial extrapolation
problem.

The result reinforces the current model decision: keep one-hop RF-context as
the spatial baseline and treat deeper graph/context variants as diagnostics.
