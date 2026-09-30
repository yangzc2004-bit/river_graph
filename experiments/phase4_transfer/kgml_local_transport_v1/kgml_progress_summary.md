# Local--Transport KGML progress summary

## K1: local residual pilot

The RF-local plus learned no-message residual is the strongest KGML branch in
the temporal holdout. It improves the RF-local base by about 22% in the
original K1 comparison, while the river, bidirectional and no-message
residuals are nearly identical. The gain is therefore primarily local
ecology--hydrology and temporal support, not isolated topology.

## K2: source isolation

The zero-preserving message-only branch improves over its matched empty-edge
null by 9.50% in temporal extrapolation and 0.56% in spatial extrapolation.
This establishes a small conditional river-message signal, especially when
the prediction task is temporally displaced.

## K3: joint local plus message residual

Joint optimization does not help. It is 4.39% worse than the learned local
residual in temporal extrapolation and statistically indistinguishable in
spatial extrapolation.

## K4: channel isolation

Both target-observation and hydro-ecology message inputs retain signal:

| channel | temporal gain vs null | spatial gain vs null |
|---|---:|---:|
| target observation | 3.68% | 0.57% |
| hydro-ecology | 4.60% | 0.60% |
| all input channels | 9.50% | 0.57% |

The full message branch is materially better than either restricted channel in
the temporal task, suggesting complementary information rather than a single
dominant channel.

## K5: dual-channel gate

The gated dual branch improves over the null by 3.77% in temporal and 0.56% in
spatial extrapolation. It does not beat the K4 hydro-ecology branch and is
6.33% worse than the K2 all-input branch in temporal extrapolation. The gate
does not convert channel separation into an additional predictive gain.

## K6: cross-analyte replication

The K2 all-input message branch was tested against the K1 local residual branch
for pH and specific conductance, using the same temporal and spatial holdouts.
The DOC result does not replicate across analytes:

| analyte | temporal message gain | spatial message gain |
|---|---:|---:|
| DOC | +9.50% | +0.56% |
| pH | -3.07% | +0.34% |
| specific conductance | -0.56% | -0.36% |

The pH temporal loss is supported by its paired confidence interval; the other
pH and conductance differences are small and compatible with zero. The current
evidence therefore supports an analyte- and missingness-dependent river
message effect, rather than a universal advantage of the graph branch.

## Spatial context residual pilot

The E3 spatial holdout was revisited with RF-context as the base. This base
already uses visible current-month global, upstream and downstream summaries;
the new graph branch was trained only on its station-blocked OOF residual.
Across three seeds and 2,531 held-out station-month cells:

| model | MAE | gain vs RF-context |
|---|---:|---:|
| RF-context / exact zero-message null | 2.6838 | -- |
| context + upstream message residual | 2.6636 | **0.75%** |
| context + both-direction message residual | 2.6725 | 0.42% |

The upstream gain is supported by the station-clustered bootstrap interval
[0.0013, 0.0402] mg/L. It is concentrated in cells with visible upstream
support (MAE 1.7871 to 1.7547); cells without visible upstream support show
almost no change. The both-direction diagnostic is smaller and its interval
crosses zero.

A stricter confirmation then reserved HUC6 `101900` as an unseen spatial
validation block while keeping the E3 test stations fixed. Under this spatial
model-selection protocol, the upstream gain fell to 0.017% (bootstrap CI
[-0.0010, 0.0020] mg/L), and the both-direction branch was slightly worse.
The first E3 result is therefore useful exploratory evidence for conditional
upstream correction, but it is not yet a robust spatial-extrapolation gain.
RF-context remains the strongest confirmed spatial baseline.

## Unified spatiotemporal fusion pilot

The existing temporal residual expert and RF-context expert were combined in
the same transformed target space with an observable-feature gate.  The gate
was trained from validation errors and evaluated once on the terminal test
predictions; it never reads test targets.  The feature sets used age,
upstream support, season, flow, and optionally the missingness family and
expert prediction disagreement.

The best low-complexity baseline was a family-specific constant blend,
`z_fuse = (1-alpha) z_RF + alpha z_residual`, with alpha selected from the
validation set:

| missingness family | alpha | RF-context MAE | residual MAE | fused MAE |
|---|---:|---:|---:|---:|
| E1 random point | 0.40 | 1.4490 | 1.4908 | 1.4277 |
| E2b partial time | 0.74 | 1.0571 | 0.8030 | 0.7935 |
| E2a strict time | 0.74 | 1.0840 | 0.9763 | 0.9362 |
| E3 spatial holdout | 0.00 | 2.6791 | 2.9543 | 2.6791 |

Across the four query collections the descriptive pooled MAE was 1.5134 for
the family blend, compared with 1.5943 for RF-context and 1.6119 for the
residual expert.  The learned per-cell classifier gate reached 1.5186 with
the strongest feature set, but it mixed experts in E3 and was worse than the
RF baseline there.  Direct validation-loss optimization was also less stable
than the family-constant blend.  The next unified model therefore uses the
family blend as the reference and treats a per-cell gate as an extension;
spatial cases currently stay RF-dominant while temporal cases use the
temporal residual contribution.

## Support-matched spatial validation

The spatial validation design was then matched to the frozen E3 test's graph
support structure. A deterministic 20-station holdout was selected using only
graph connectivity and observation masks, with visible-upstream support 0.4033
versus 0.4034 in E3 (7 supported and 13 unsupported validation stations).

On this validation block, RF-context had MAE 2.52684, the upstream residual
branch 2.52766, and the both-direction branch 2.53496. The upstream branch was
0.032% worse with a paired station bootstrap interval of [-0.00184, -0.00001]
mg/L. On the frozen E3 test, the upstream branch improved RF-context by only
0.040% and the both-direction interval crossed zero. This matched-support
confirmation therefore keeps RF-context as the spatial baseline and treats the
graph residual as a small conditional diagnostic rather than a general spatial
gain.

## Explicit multi-hop context diagnostic

To test whether the spatial limitation was simply a short message range, the
RF-context baseline was augmented with exact directed upstream summaries at two
and three hops. The support-matched validation MAE decreased from 2.52684 to
2.52249 (two hops) and 2.51755 (three hops), but the frozen E3 test MAE
increased from 2.65364 to 2.65964 and 2.66210. Deeper explicit context thus
fits the validation block without improving the held-out spatial test, so more
graph depth is not the next upgrade.

## Model decision

Stop adding spatial architecture in this branch. Keep the K2 all-input
message-only model as the interpretable conditional river-message module and
keep RF-context as the confirmed spatial baseline. The cross-analyte test shows
that the river-message mechanism is strongest for DOC under temporal
extrapolation. The spatial-validation result says that the next improvement
should target station-to-station transfer of residual structure, rather than
another graph-depth or gate search.
