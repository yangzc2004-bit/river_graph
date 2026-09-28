# Continuation R4: observation-aware multi-scale pilot

This continuation combines the two mechanisms that showed complementary
behavior in R3:

- observation-aware memory (M1), which uses the age and support of visible
  target observations;
- causal multi-scale paths (M3), which separate short, seasonal and long
  temporal structure.

The combined arm is named `m13` in the experiment files and
`observation_multiscale` in products. It is a research experiment name, not a
paper model name.

The pilot contains 3 analytes x 2 holdout families x 3 seeds = 18 runs. It is
compared on identical station-month query cells with M1, M3 and the temporal
random forest. The combination was implemented after the separate M1/M3 pilot
results and is therefore a follow-up mechanism test, not a retrospective
change to the earlier comparisons.
