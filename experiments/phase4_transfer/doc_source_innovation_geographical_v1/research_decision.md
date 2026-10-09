# Research decision: time-aligned source DOC in geographical reconstruction

## Completed replication

All25 fixed packages are complete: five HUC4 regions1013/1019/0708/1030/1101
and seeds42–46, including250 double-held-fold reference forests,25 neural
residual fits and25 equally informed ExtraTrees fits. Receiving stations provide
no DOC/pH/conductance at K0. No regional or K-specific winner was selected.
The same source-development candidate was carried through every package.

This is retrospective geographical replication on already evaluated ST357
roles. It is not independent external-basin validation. Existing geographical,
external, temporal and portable-release products remain intact.

## Main result

Primary K0 uses all valid target DOC cells. Seeds are averaged within each
region and five regions receive equal weight. Intervals use5,000 paired station
draws; months from the same station stay together.

| Procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Retained complete |2.282266|2.504735|10.519953|
| Same-month source complete |2.272858|2.498975|10.471173|
| Retained neural-only |2.265595|2.485495|10.494278|
| Same-month source neural-only |2.268006|2.488006|10.445286|
| Strong station-hidden trees |2.344885|2.547670|10.520142|
| Trees with identical source information |2.314529|2.516639|10.481605|
| Older complete model |2.360283|2.556920|10.632267|

Complete improvement over the actual retained complete model is
**0.412% [-0.044%,0.924%]**, positive in four of five region averages.
Its Q90 improvement is **0.464% [0.292%,0.670%]**, with positive directions
in all five regions. Station-equal MAE and RMSE also decrease slightly; signed
bias is essentially unchanged (-0.649285 versus-0.649449mg/L).

The complete model improves3.072% [0.099%,5.856%] over bare strong trees,
with five positive region directions. Against trees with the SAME new source
information, the gain is1.800% [-1.498%,4.904%], positive in three regions.
The older complete-model gain3.704% [1.298%,6.273%] includes the retained
model's prior improvements. It is not the incremental benefit of this branch.

The enriched tree itself improves1.295% [0.499%,2.123%] over its retained
bare counterpart, positive in three regions. Thus current source information
can also help an explicit-feature model. It does not establish a graph-specific
or attention-specific advantage.

Matched neural-only overall MAE is0.106% worse [-0.163%,0.401% worsening],
although its Q90 error improves0.467% [0.296%,0.609%] in all five directions.
The whole-procedure improvement cannot be described as a demonstrated overall
neural-only gain. The complete procedure includes validation-selected static
memory fusion. Keep these two contrasts separate.

## Adaptation and failure environments

The curve population always excludes the same five candidate support cells,
including K0. Do not compare its K0 directly with the all-observed primary K0.

| K | Retained complete MAE | Source-informed complete MAE | Relative improvement |
|---|---:|---:|---:|
|0|2.296495|2.286835|0.421%|
|1|2.160459|2.156229|0.196%|
|3|1.987946|1.972513|0.776%|
|5|1.909790|1.902424|0.386%|

K3 improvement has interval[0.416%,1.125%]; K0/K1/K5 overall intervals cross
zero. Q90 curve improvements range0.280–0.534%, with positive overall paired
intervals and five positive region directions. No procedure is switched for K.
Equally informed trees have the best K3/K5 point estimates among these complete
comparators; those results stay visible.

Q90 identification remains weak outside1013: mean recall is zero in0708,
1019,1030 and1101. Their unique tail-cell counts are66,43,15 and11,
respectively; the last two are unstable. Region1013 has904 cells above its
source-derived threshold and mean recall0.987. These substantially different
tail populations are reported rather than treated as equally abundant events.
Lower Q90 MAE does not establish reliable high-DOC event detection.

Support, ecological novelty, source-similarity and hydro-availability strata
are saved. Empty strata are retained as absent; summaries state the number of
contributing regions (some ecology groups have only two or three). The no-donor
stratum also has a small improvement, consistent with the learned local readout
changing. Do not assign the entire complete-procedure gain to source donors.

## Verification

All25 nested-reference/source-library/input/neural/memory/support products replay.
Libraries, full-grid neural and memory predictions and support policies agree
bitwise; forests agree within1e-12 parallel-reduction tolerance. Initial weights,
budgets, target roles, unchanged parent products and prediction sidecars are
verified. The full suite passes979 tests with two explicit skips, Ruff and the
historical artifact audit pass. The four-comparator vector/PNG figure was
actually inspected; adding the candidate filter avoids ambiguous forest
contrasts in its interval panel. Analysis, prediction and training data were
not changed by that plot correction.

## Research choice

The5% improvement goal against both primary comparators was not reached.
This version provides a small, consistent high-DOC error benefit and a useful
source-information result, not a major overall spatial-performance advance.
Retain it as an evaluated candidate and keep the current portable release.
Do not run another regional search, select region/K winners, or recalibrate on
the previously evaluated external basin.

The next source-only mechanism should test **hydrology-conditioned allocation
of current donor innovations**. Current candidates share one fixed ecological
weighting scheme. Earlier retrieval used source residual profiles rather than
the contemporaneous donor departures tested here. A finite matched attention
experiment can therefore ask a different question: whether the current receiver
GRU state and source hydro conditions improve the use of actual same-month
source information. Start again on142/143/144 source training/validation roles,
reuse the verified double-fold references, keep matched historical and fixed-
weight controls, and compare with both the retained complete model and this
source-informed candidate. No geographical/external result selects attention
dimensions or an operational winner.
