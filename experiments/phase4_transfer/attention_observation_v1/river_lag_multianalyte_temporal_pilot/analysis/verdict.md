# Multianalyte river lag pilot verdict

The strict temporal extrapolation pilot compared the upstream edge-and-lag attention residual with a matched no-message residual for pH and specific conductance (three seeds each).

- pH: river MAE 0.1936 versus no-message 0.1940, a 0.2% mean reduction. Seed directions were mixed.
- Specific conductance: river MAE 104.87 versus no-message 105.87 in raw units, a 0.94% mean reduction. The reduction was driven by two of three seeds; the third was essentially tied.
- The conductance Q90 MAE was lower by about 10.6 units on average, while pH Q90 was lower by about 0.003.

The river arm includes the existing same-month upstream spatial message as well as the lag attention; the no-message arm removes both. Therefore these results establish a small conditional graph benefit, especially for conductance, but do not isolate the incremental value of learned lag attention. A matched same-month-attention arm should be run for conductance before making a lag-specific claim.
