# Source concentration–hydrology correction: research decision

## Result and model choice

All9/9 source-development packages are complete, using partitions142/143/144
and seeds42/43/44. Both correction modes were fitted exclusively to source
station-hidden OOF tree predictions; every prior prediction remains unchanged.
There are11 matched arms. No old target, geographical or external query was
scored to select this mechanism.

| Procedure | K0 MAE, mg/L | Mean bias, mg/L | Source-Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained complete model |1.769053|-0.638608|9.075420|
| Strong station-hidden trees |1.866362|-0.611014|9.394458|
| Trees + log-affine correction |1.861948|-0.762165|9.500539|
| Trees + concentration/hydro correction |1.858876|-0.787967|9.519020|
| Complete + log-affine correction |1.777193|-0.789748|9.172332|
| Complete + concentration/hydro correction |1.782415|-0.815543|9.191678|

The conditional complete arm has -0.755% gain [-1.850%,0.372%] against the
retained complete model, with0/3 improving partition directions. The log-affine
complete arm gives -0.460% [-1.307%,0.455%], likewise0/3. Thus neither is added to
the retained deployment. Negative point estimates and consistent unfavorable
partition directions make an additional correction unattractive; the intervals
do not establish equivalence or a universal harm.

The conditional tree-only gain is0.401% [-0.759%,1.608%], with3/3 positive
directions but uncertain magnitude. The full conditional arm remains4.113%
[1.570%,6.870%] better than its equally calibrated tree control. That contrast
does not constitute an upgrade over the actual retained complete model.

## Interpretation and next mechanism

The source-OOF correction shifts the validation prediction downward by roughly
0.12–0.23 mg/L across packages. Existing complete residuals already move the
reference, with mean directions varying by partition. Their additive combination
increases aggregate negative bias and tail error. These observations support
testing correction overlap and concentration scale; they do not prove that
overlap alone caused every unfavorable result.

The next source experiment changes the existing neural head's output units:
learn a log1p concentration-relative correction while retaining the same
ecological/GRU backbone and native mg/L MAE objective. This asks whether a fixed
absolute correction transfers poorly between concentration regimes. It is
different from fitting another external bias, a larger attention gate or the
previous distributional NLL/mixture experiment. The existing native expert and
strong tree remain fixed matched comparators.

## Analysis and figure verification

Main estimates average individual seeds inside a development partition and
then give the three partitions equal weight. Intervals use5,000 paired station
draws; each station's months remain together across overlapping partitions.
All saved corrections replay bitwise and prior prediction frames match exactly.
The first verifier compared mixed-type arrays containing unavailable historical
diagnostics, treating paired NaNs as different values. Pandas exact-frame
comparison repaired that analysis check; fitting, predictions and endpoints
were unchanged. The original failed log and execution snapshot are preserved.

PDF/SVG/PNG figures use the reviewed metrics and paired intervals. Their first
legend overlapped bottom labels; increasing the bottom margin repaired the
layout. The final rendered image was inspected. Hydro strata are descriptive
groups, not an intervention on monitoring availability. The legacy dataset,
all earlier models, previous paper and geographical/external results are retained.

Full suite after the new operators:932 passed,2 skipped. Ruff passed and the
historical artifact audit exited0. Git submission remains pending under the
environment's read-only Git policy; related files are retained for permitted
submission. These results are source development, not a new independent test.
