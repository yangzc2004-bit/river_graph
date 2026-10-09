# Temporal compatibility: completed research decision

## Question and scope

Six fresh temporal refits are complete: unobserved periods and observation-
assisted periods, seeds42/43/44. Each fits its own environmental reference,
backbone and native residuals using the original ST357 time roles. Deployment
weights are not reused because they include observations in these time test
periods. Legitimate past DOC/context remains available in this monitored-site
experiment; its inputs differ from water-quality-free new-station K0.

This is a retrospective compatibility check of the station-hidden native
recipe and existing validation fusion. It does not repeat the independent
basin deployment or test its already-fitted portable ensemble.

## Central prediction result

MAE in mg/L, individual-seed errors averaged within the fixed time split:

| Procedure | Unobserved periods | Observation-assisted periods |
|---|---:|---:|
| Original20-cap hybrid |0.874739|0.806937|
| Current39-feature context trees |1.093144|1.046858|
| Ordinary47-feature daily trees |1.028129|1.004568|
| Strong station-hidden daily trees |0.897097|0.936269|
| Current120-cap native residual |1.062944|1.012899|
| Current full temporal fusion |0.905180|0.853555|
| Upgraded30-extra-cap native residual |0.894939|0.933583|
| Upgraded full temporal fusion |0.785746|0.797619|

The upgraded full procedure improves over current full fusion by13.19%
[3.91%,20.69%] in unobserved periods and6.55% [-2.32%,13.45%] in assisted
periods; both have3/3 positive seed directions. Its gains against strong
station-hidden trees are12.41% [8.60%,15.84%] and14.81% [11.79%,17.55%],
also3/3. The second current-fusion contrast remains uncertain. Against the
original20-cap hybrid, gains are10.17% [-0.10%,18.98%] and1.15%
[-8.27%,9.56%]. Preserve these distinct comparators.

All intervals use5,000 paired station-bootstrap draws. There are2,223 unique
query cells at69 sites in the unobserved task and1,778 cells at67 sites in the
assisted task. Repeated training seeds do not add observations.

## What generates the gain

The native residual by itself adds little to the new strong trees:0.24%
[-0.37%,0.90%] and0.29% [-0.39%,1.08%]. Both intervals cross zero. The upgraded
native expert improves its old native reference, but that comparison also
changes the environmental reference and adds optimization; it does not isolate
an improved temporal operator.

All six current and upgraded complete procedures select the existing log-affine
fusion family on validation. Its bias/concentration adjustment contributes to
the complete-procedure gain. In seed42, the upgraded native scale is zero and
the final fusion effectively calibrates the environmental prediction. Inspect
`analysis/fusion_diagnostics.json` for selected coefficients/scales. A calibrated
tree-only comparison is a useful future source-role diagnostic; do not call the
current raw-tree contrast an isolated neural-memory contribution.

The scientific choice is to retain the upgraded procedure's temporal result,
with its exact information condition and fusion described. No weights or
coefficients are changed from test results. The main portable new-station
product and its geographical/external results remain unchanged.

## High DOC and station weighting

Training-derived Q90 identifies12 cells in the unobserved task and7 in the
assisted task; both are unstable. Upgraded fusion Q90 MAE is6.947001 and
7.351621 mg/L, versus current fusion7.042250 and7.793332, and strong trees
6.542141 and6.750706. These small groups do not establish a tail advantage.

Upgraded fusion station-equal MAE is1.064175/1.082852. Current fusion gives
1.120115/1.049786. The assisted task improves cell-weighted error but has
slightly higher station-equal error and bias0.240777 vs0.161151. These summaries
remain visible alongside MAE; the entire task is not described as uniformly
improved.

## Replay and analysis repair

All six source snapshots, completed stages, inputs and saved prediction hashes
were verified. Full-grid replay of all eight fitted procedures agrees within
7.2e-14 mg/L. Exact query cells and truth were checked. The archived assisted
test mask is unsorted; review originally compared sorted parquet cells with
that unsorted array. Sorting expected cell identities fixed the verifier and
analyzer. The failed log is retained, and `analysis_repair.json` records that
training, predictions and endpoints were unchanged.

The PDF/SVG/PNG figure was inspected. Its first left labels were clipped;
expanding margins repaired the plot without changing numbers. Required software
checks are925 tests passed,2 skipped; Ruff clean; historical audit exit0.
Large fitted tree/checkpoint/full-grid caches remain local.

## Research handoff

Geographical confirmation, independent basin replication, empirical interval
coverage/width, a portable five-seed new-station model and this temporal check
now answer the authorized plan's evaluation questions. The new manuscript
integrates them with the environmental/temporal method and future research.

The strongest further performance question is source-dependent residual and
fusion calibration across concentration/hydrological regimes. Existing direct
retrieval and equal-station loss did not solve it. Any new mechanism starts
again on142/143/144 source roles, with calibrated tree references where relevant,
and retains the completed geographical/external version rather than tuning it
on its evaluated queries.
