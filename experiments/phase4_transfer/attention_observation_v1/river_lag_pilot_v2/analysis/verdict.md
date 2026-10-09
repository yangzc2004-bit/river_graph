# River lag attention pilot verdict

The DOC pilot compared the causal edge-by-lag attention residual with a matched no-message residual arm across three seeds and two masks.

- In strict temporal extrapolation, mean MAE was 0.9846 for river attention and 0.9868 for no-message (about 0.2% relative improvement; seed direction was mixed).
- In spatial holdout, mean MAE was 2.9567 for river attention and 2.9570 for no-message (about 0.01% relative improvement; the paired differences were effectively zero).
- Conditional lag mass was nearly uniform across 0, 1, 3, 6 and 12 months, so the current pilot did not learn a sharply identifiable transport lag.

This is a pilot mechanism result, not evidence that river topology is a general performance driver. The useful next experiment is a leaner lag comparison (fixed lag versus learned lag) or an explicit flow-conditioned lag prior, rather than adding more spatial depth.
