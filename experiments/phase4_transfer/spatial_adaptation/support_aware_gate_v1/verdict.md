# Support-aware regional gate pilot

## Design

This development pilot combines the existing spatial KGML prediction
(`residual_context_msgdelta`) with the source-pool-40 ExtraTrees regional
expert. It uses the exact support-matched validation mask used by the KGML
artifacts for internal selection and the frozen E3 test mask for one outer
evaluation. K=0 and K=5 use the same paired query (five target months per
station are reserved in both conditions). Three existing seeds (42--44) are
used; no model is retrained.

Candidates are global KGML, regional expert, a fixed convex blend, support
residual corrections, and a hard support-aware gate. The gate selects the
regional residual when the absolute station mean log1p support residual is
above a threshold. Candidate and threshold are selected on internal
station-heldout validation only.

## Result

Internal validation selected the regional expert for both K=0 and K=5. The
support gate did not improve over that regional expert on the internal block;
therefore no gate was selected for the outer score. On the outer E3 query,
the selected regional candidate scored MAE 2.469 (K=0 and K=5 in this pilot),
while the K=5 support-corrected candidates scored 2.019--2.020. The best
fixed blend (weight 0.75 on the regional residual) scored 2.019, essentially
the same as the regional residual (2.020). The corresponding global KGML
baseline was 2.641 at K=0 and its K=5 residual correction was 2.180.

The paired station bootstrap for the internally selected candidates gives an
outer reduction of 0.358 MAE versus global K=0 (95% interval -0.687 to
-0.116; 43 stations), driven by the regional expert. The support-aware gate
itself is not a new source of gain: it was not selected internally and does
not beat the fixed residual/blend at K=5.

## Interpretation

For this E3 spatial transfer, source-pool regionalization is the main useful
component. Opening five target-station labels enables a strong residual
correction of the global KGML prediction, but a threshold gate based only on
the magnitude of that correction is not better than a fixed residual/blend.
The next model work should focus on improving the regional expert or learning
a richer support-conditioned combination, rather than adding this gate in its
current form.

All numbers are development-extension diagnostics; the outer E3 block is
scored once after internal selection.
