# Concentration-relative DOC residual: source-development decision

## Result and model choice

All nine packages are complete: source partitions142/143/144 × training
seeds42/43/44. Retain the existing native-concentration model. Changing its
neural residual to log-concentration units does not improve the actual retained
complete procedure on this development panel.

| Procedure | Equal-partition K0 MAE, mg/L | Mean bias, mg/L | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained complete native residual |1.769053|-0.638608|9.075420|
| Log-concentration complete residual |1.777794|-0.633345|9.139942|
| Log-concentration neural residual alone |1.777945|-0.625819|9.127858|
| Strong station-hidden trees |1.866362|-0.611014|9.394458|
| Preceding complete model |1.829579|-0.667535|9.225863|

The new complete procedure's relative MAE gain over the retained complete model
is -0.494%, with a5,000-draw paired station-bootstrap interval
[-1.414%,0.402%]. Only one of three partition directions improves. Q90 gain is
-0.711% [-1.651%,0.044%]. Neural-only gain against its matched preceding
neural-only model is -0.334% [-1.455%,0.790%]. These results do not support
replacing the native mg/L readout.

The new model still improves over strong trees by4.745% [2.264%,7.400%]
and over the older complete model by2.830% [1.110%,4.615%]. Those contrasts
largely retain earlier residual-model advantages; they are not evidence that
the present change improves the current model.

## What was tested

Only the neural correction's output units changed. It predicts an additive
log1p residual and combines it multiplicatively with the fixed environmental
reference, while retaining native mg/L MAE training, tail weight2,30 epochs,
patience5, the existing ecological/observation-aware GRU and source input views.
The initial backbone weights match the preceding candidate's initial weights;
the new head starts at zero. Existing source-only trees, OOF residual reference,
validation selection and ecological-memory fusion remain available.

This multiplicative correction reduces aggregate bias slightly, but does not
improve absolute reconstruction error or the high-DOC tail. The tested output
parameterization is not the main performance bottleneck. This does not rule
out other distributional objectives or concentration-specific mechanisms.

## Reproduction and execution history

`scripts/analyze_doc_log_concentration_residual_v1.py --bootstrap-draws 5000`
replays the saved neural and complete products exactly, checks every sidecar
and confirms that all parent comparator predictions remain unchanged. Seeds
are averaged within partition; partitions are weighted equally. Bootstrap
draws retain all months of each sampled station and preserve stations shared
between development partitions. Seeds do not increase the ecological sample
count. Numerical sources are in `analysis/sources.json`.

V1 stopped after its first fit because the existing ecological-memory helper
required an explicit month count. V2 supplies that unchanged dimension and
persists the neural stage before downstream export. Failed V1 execution files
are preserved; no V1 prediction package was completed. The repair changed no
scientific setting. The full suite at this version passed932 tests, with two
skips; Ruff passed.

## Next mechanism

Test whether source training should hide entire water regions rather than
scatter hidden stations among monitored neighbors. Construct region-hidden
source feature views and nested region-blocked forest OOF references, retaining
the existing native residual architecture and training budget. This targets
the mismatch between source observation outages and geographical deployment.

The present comparisons use repeatedly evaluated source-validation roles.
They are development evidence, not new geographical confirmation or external
basin validation. Old geographical and external queries have not selected this
output-unit change or the next regional training design.
