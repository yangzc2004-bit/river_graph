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

## Model decision

Stop adding spatial architecture in this branch. Keep the K2 all-input
message-only model as the interpretable conditional river-message module and
keep K1 no-message local residual as the strongest overall KGML predictor.
The next scientific extension should test whether the same decomposition holds
for pH and specific conductance, rather than adding more layers or gates to
the DOC model.
