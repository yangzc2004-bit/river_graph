# Research decision: environmental trajectories in the existing DOC GRU

## Completed result

All nine source packages (142/143/144 ×42/43/44) completed45 nested station-
hidden forest fits and18 DOC residual fits. Current-reference and full-history
arms each add64 zero-initialized projection weights to the SAME retained GRU;
all original input features, DOC targets, forest configuration and fusion remain.

| Procedure | Source K0 MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete model |1.769053|9.075420|
| Current-reference complete |1.769488|9.069520|
| Reference-history complete |1.769113|9.078530|
| Reference-history native residual |1.772360|9.054875|
| Strong station-hidden trees |1.866362|9.394458|

Full-history gain over the actual retained complete model is-0.0034%
[-0.1481%,0.1564%], two of three partition directions positive. Against the
matched current-reference complete model it is0.0212%[-0.2053%,0.2508%].
Q90 gain versus retained is-0.0343%[-0.1477%,0.0543%]. Native-only also does
not improve. All intervals use5,000 paired station draws, training seeds
averaged within partition and equal partition weight.

The5.21% advantage over strong trees comes from the existing model and must
not be credited to reference history. Keep the portable release; this new
projection does not warrant another geographical or external replication.

## Scientific reading and next work

The DOC-aligned dense reference contains a learnable concentration signal, but
feeding it through the existing GRU gives essentially the same complete
prediction quality. Together with the previous source chemistry pretraining
result, current evidence favors finding an additional environmental information
source over another rearrangement of the same inputs.

Audit precipitation inputs as the next mechanism: monthly precipitation and
causal antecedent accumulation can represent wet periods that temperature,
discharge and static ecology may describe incompletely. This is a hypothesis,
not a conclusion from the present experiment. First check data availability,
units, grid resolution and time alignment without DOC-based selection. Then
compare the existing model and strong trees under identical new covariates on
source roles only. Do not confuse extra weather information with a pure network
architecture gain or add unconfirmed chemical/reference branches to the model.

The frozen daily-flow metadata already contains120,541 station-months before
and after the monthly mask, with zero excluded months. Simply unmasking that
cache cannot supply new hydrology. No local precipitation/reanalysis cache was
found in the project data paths; new data must be obtained and audited first.

## Reproduction

All native and complete products replay bitwise from saved states. Source fold
records exclude held stations before either feature construction; dense observed
OOF predictions agree with the retained cache within1e-12 and exact old values
remain the DOC residual targets. The reference mean/std recomputes from dense
source-only OOF predictions, and matched arms receive identical reference values.
Original parent products/backbones are unchanged. Four new tests verify zero-
projection compatibility, permitted causal/current influence, source-only
scaling, arbitrary station count and save/load.954 tests passed, two skipped;
Ruff and historical audit passed. The four-panel scientific figure was viewed;
labels, intervals and partition comparisons are clear. Large forests and inputs
remain local. This is source validation inside ST357, not an external result.
