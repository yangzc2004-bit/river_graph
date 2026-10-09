# Research decision: exact inverse conversion of relative source corrections

## Completed experiment and result

All nine source packages142/143/144 ×42/43/44 are complete, comprising27
neural fits and90 reused double-held-fold references. No new forest was fitted.
All candidate arrays, source roles, initial weights, predictions, ecological-
memory components and correction diagnostics replay bitwise. Analysis uses
5,000 paired station draws, with seed means followed by equal source-partition
weights. These are source-validation development results.

The change replaces only the source correction's first-order inverse by
`(1+B)*expm1(u)`. Native local readout, tail-weighted native loss, zero initial
output,37,900 parameters and30epochs/patience5 remain unchanged. This is
not whole-model log-output training or a physical conservation constraint.

| Complete procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Exact current-source ratio |1.747190|2.204888|8.921308|
| First-order relative current source |1.748102|2.205626|8.932044|
| Exact current-source fixed prior |1.747671|—|8.939792|
| Exact earlier-season source |1.768245|—|9.071785|
| Native-value current-source attention |1.756645|—|8.980932|
| Retained complete |1.769053|2.232407|9.075420|
| Strong station-hidden trees |1.866362|—|9.394458|

Exact current-source correction improves retained complete by1.236%
[0.570%,1.975%], positive in all nine packages, three partitions and three
training seeds. Q90 gain is1.698% [0.758%,3.110%], with the same directions.
These include the previously demonstrated benefit of relative source transfer.

The **isolated exact-vs-first-order overall increment is0.052%
[-0.209%,0.315%]**, positive in four packages and two partitions. Its Q90
increment is0.120% [-0.103%,0.373%]. The experiment does not establish an
additional benefit from the inverse conversion. Native-only MAE is1.752051
versus first-order1.751668: -0.022% [-0.419%,0.366%]. Keep the simpler
first-order candidate for further development; do not launch a geographical
matrix for this nearly redundant conversion.

Gain over bare strong trees is6.385% [3.676%,9.372%], which includes the
retained model's existing capability. Gain over native-value complete attention
is0.538% [0.120%,0.980%]. Neither is the exact operator's incremental effect.

## Controls, concentration behaviour and diagnostics

Complete gain over matched fixed-prior correction is0.028%
[-0.431%,0.478%], with only one positive partition average. Gain over the
earlier-season arm is1.191% [0.466%,1.965%], positive in every package and
partition. The corresponding Q90 gains are0.207% [-0.121%,0.708%] and
1.659% [0.688%,3.110%]. Current observations retain value, but learned
allocation and exact curvature do not show a new overall source-validation gain.

The learned current branch assigns mean zero-innovation prior mass0.375;
mean per-head Shannon entropy is0.820 in natural-log units. The relative
output has mean-0.00229 and within-package standard deviation averaging
0.0412. Before the selected final residual scale, mean absolute native source
correction is0.1557mg/L. The exact
minus first-order correction averages0.00610mg/L; it is nonnegative by
the exponential's curvature. This describes operator magnitude, not a causal
error decomposition or proof that curvature is useless in other data.

Complete signed bias is-0.604979mg/L, versus first-order-0.605140 and
retained-0.638608. Reference-concentration, donor-support and hydrological-
availability summaries are retained. No subgroup outcome replaces the overall
procedure or chooses a new geographical model.

## Verification and next mechanism

Zero-fusion equality and source-projection derivative, finite causal inputs,
save/reload and every fitted diagnostic pass. All three arms preserve37,900
parameters.992 tests pass with two skips; Ruff and historical audit pass.
The actual figure PNG was inspected. Original source/geographical/external
products and portable deployment remain unchanged.

Close this exact-conversion experiment without a larger lag, epoch or exponent
scan. The next source question concerns information removed by season centering:
can source stations' persistent seasonal OOF bias help a receiver beyond the
current anomaly and existing annual ecological-memory correction? Test full
relative residual, seasonal-only and matched historical/fixed arms inside the
same first-order attention operator. Return to142/143/144 source roles; do
not tune this mechanism from geographical or external concentrations.
