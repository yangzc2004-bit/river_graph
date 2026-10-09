# Research decision: selecting current source DOC information

## Completed source development

Nine packages at source partitions142/143/144 and seeds42/43/44 are complete:
27 neural fits,90 reused double-held-fold references and no new forests. Every
receiver has no DOC/pH/conductance input. The existing ecological self encoder,
observation GRU, native residual objective,41-column readout,12-month window
and static memory fusion remain. Two32-dimensional attention heads allocate
individual source innovations using ecology/current hydrology and GRU queries.

All nine candidate/library-role/initial-weight/neural/memory/diagnostic replays
pass.985 tests pass with two explicit skips; Ruff and historical audit pass.
The four-panel PNG/vector figure was actually inspected and is legible. Large
fits/candidate arrays remain local. Git index writes are unavailable in the
current filesystem policy; related files are retained in the scoped whitelist.

## Results on selected source-validation stations

Seeds are averaged inside partitions, then the three partitions are equally
weighted. Intervals use5,000 paired station draws. These are development results
on validation stations used for checkpoints/fusion, not independent confirmation.

| Complete procedure | MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained release |1.769053|9.075420|
| Preceding current-source readout |1.761078|8.999148|
| Learned current-source attention |1.756645|8.980932|
| Fixed ecological-prior attention |1.759505|8.995962|
| Learned earlier-season attention |1.769581|9.069789|
| Saved enriched trees |1.865680|9.396532|

The new complete procedure improves **0.701% [0.220%,1.239%]** over the actual
retained complete release: three positive partition averages, seven of nine
packages. Q90 gain is **1.041% [0.351%,2.131%]**, positive in all partitions
and all nine packages. This is a small reliable development improvement.

Against learned earlier-season attention, complete MAE improves0.731%
[0.244%,1.291%], with all nine directions positive; Q90 improves0.980%
[0.244%,2.153%]. Current information has value beyond the matched historical
input in this selected development panel.

The incremental COMPLETE advantage over the fixed ecological prior is only
0.163% [-0.197%,0.495%], one of three partitions positive; Q90 is0.167%
[-0.106%,0.556%], also one partition. Compared with the preceding complete
current-source readout, the gain is0.252% [-0.081%,0.583%]. Therefore the
overall complete advantage cannot yet be attributed specifically to adaptive
donor selection. Fixed weighting accounts for most of it.

The neural-only comparison contains a conditional attention signal. Learned
current Q90 MAE8.928370 improves0.449% [0.157%,0.973%] over fixed-prior native
8.968661, and0.532% [0.129%,1.178%] over preceding native8.976144. Both
directions are positive in two of three partitions. Overall native advantage
over fixed prior remains uncertain:0.110% [-0.289%,0.462%]. Relative to retained
native, overall gain is0.628% [0.047%,1.243%] and Q90 gain1.377%
[0.643%,2.574%], with all three partitions positive. Keep neural-only and
complete contrasts separate; memory fusion changes the marginal comparison.

The5.844% complete advantage over saved enriched trees includes prior model
ability. Those trees see aggregate innovations, not individual donor hydro keys;
it is not a completely matched raw-input architecture contrast.

## Attention diagnostics and interpretation

Matched source support is available in84.23% of receiving cells, averaging
3.23 sources and2.65 sources with daily hydro support. Learned-current mean
zero-prior mass is0.579 versus0.662 for fixed ecological weighting; mean entropy
is0.764 versus0.897. The module reallocates information rather than assigning
all cells to a single fixed donor. These averages include unsupported cells,
where the prior has mass one. They are descriptive allocation diagnostics,
not physical transport or causal coefficients.

## Next fixed replication

Keep the current portable release, external predictions and manuscript intact.
Proceed with a separate fixed geographical version of ALL THREE attention arms,
using the established five HUC4 roles and seeds42–46. This is justified by the
complete improvement over the actual release/current-versus-historical control,
and the positive matched neural-only Q90 signal. It does not declare overall
adaptive allocation established or meet the5% geographical working target.

The replication asks whether that conditional tail signal survives geographical
shift and whether adaptive allocation beats the fixed prior as a whole procedure.
Reuse all250 verified pair references and existing source preprocessing/initial
weights/forests. Fit75 neural models; do not select region/K winners. First
verify the technical1013/42 package, then carry the same three arms through
all25 packages. Save point states before opening scoring/support labels.
This remains retrospective replication on already evaluated ST357 roles,
not external-basin validation. Further mechanism choices return to source roles.
