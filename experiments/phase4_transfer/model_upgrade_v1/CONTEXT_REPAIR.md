# Context branch implementation repair

Review after U4 found two defects in the existing U2 context feature path:

1. Directional sums were divided by degree-normalized counts rather than raw
   observed-neighbor counts. At branching nodes, this changes the intended mean.
2. Context was written beginning at channel 11 rather than 10. This left the
   first context slot empty and overwrote the first hydro-regime channel.
   The temporal wrapper adds history validity later, so it is not an input slot.

Both defects are now fixed, with a branching-node hand calculation and a
channel-preservation test. Stored U2 runs, including `u2_context_v2`, used
the defective implementation and are retained as implementation diagnostics;
their scores must not be used to infer the scientific value of context.
Any future context experiment needs a new output directory and fresh runs.

U1, U3, U4 and the completed U5 configuration do not enable this context branch,
so their predictions are unaffected. Frozen predictions have not been rewritten.
