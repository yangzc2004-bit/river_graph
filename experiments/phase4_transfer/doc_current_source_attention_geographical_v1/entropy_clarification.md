# Attention entropy units

The saved diagnostics calculate per-head Shannon entropy as
`-sum(weight*log(weight))`, then average heads and cells. Values0.573,
0.659 and0.524 reported in the original research decision are in natural-log
units; they are not normalized by the number of available candidates.

The original decision's phrase "normalized entropy" is a wording error.
Prediction weights, diagnostic arrays, numerical summaries and all performance
results are unchanged. This clarification preserves the original decision and
its executed copies. Subsequent relative-source reporting uses the correct unit.
