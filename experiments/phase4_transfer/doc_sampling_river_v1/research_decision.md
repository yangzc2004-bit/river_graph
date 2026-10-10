# Research decision: source sampling support in the DOC river branch

## Decision

**Do not replace the retained environmental-state river candidate or the released
DOC predictor.** Actual source dates and sampled-flow phase were successfully
integrated into the existing sparse upstream GNN, but this integration gives no
useful incremental reconstruction gain. The full sampling arm is slightly worse
than ordinary monthly attention and the shuffled correspondence control, and
worse than the preceding environmental-state branch. Keep the complete experiment
as a mechanism result; stop adding date/phase features to this scalar readout.

This is ST357 source-validation development on partitions142/143/144 and seeds
42/43/44, not geographic confirmation or external-basin validation. No completed
geographical/external results or retained predictions were used to tune it.

## Completed implementation

Nine packages, six declared arms each: **54 neural fits, no new forests**.
The original complete ecological/hydrological/observation-aware GRU/similarity
predictor and its90-dimensional receiving query are frozen. Retain real sparse
upstream paths, double-held environmental-tree DOC innovations, monthly slots
0/1/3, two attention heads x32, native-MAE fitting, source-Q90 weight2,
30epochs/patience5. A zero-initialized12-channel linear sampling encoder adds to
the existing message encoding. The output head also starts at zero. This replaces
the observed-DOC correction rather than duplicating it. Every arm has9,448
parameters, including the extra encoder; its inputs are zero in the ordinary arm
and date-only in the age arm. The date/flow weights remain zero in1/9 full-arm
checkpoints (epoch0 selected), and are learned in the other8 (maximum norm1.856).
The null result is not an unconnected or wholly zero sampling module.

All22,571 frozen DOC station-month targets reconcile with accepted raw records
(maximum difference3.052e-6mg/L) across357 stations.32,259 accepted result rows
in the grid describe the sampling dates. Monthly DOC and OOF innovations are
unchanged result-row means. Mean/latest date, span, distinct days and result
counts use the original result weights. An aggregate is not falsely assigned a
single last-sample concentration/date. Mean and youngest age features saturate at
396days; larger carried ages remain possible in the existing monthly-age/lag
inputs. Actual sampling timestamps within a day are not used.

Nonnegative measured daily discharge supplies sample-versus-monthly Q change
and the normalized change from exactly seven days earlier. Recorded fractions
separate missing pairs from measured zero phase. Receiving flow phase uses fixed
calendar month-end and seven days earlier. Receiving DOC dates and receiving
DOC/pH/conductance are absent. Missing/conflicting/reversing discharge is
unavailable for the new ratios; original signed covariates are retained, with
no gap filling or invented daily concentration. Same-month covariates describe
retrospective monthly reconstruction, not a within-month forecast.

The shuffled arm permutes sampled-source flow metadata within prediction-month,
lag and receiving-fold pools. Date support, innovations and receiver hydro stay
fixed. The matched nonancestor arm copies real-slot timing/flow metadata, path
attributes and common DOC availability; only donor innovation and existing
hydro identity change. This compares donor identity, not transport along a
fabricated river. Receiving folds remain absent from source libraries and their
OOF environmental references. Complete-base source fitting predictions are
fitted, not complete-model OOF predictions.

## Central results

Seeds are averaged within partition; three partitions receive equal weight.
Intervals use5,000 paired whole-station draws, with a station resampled jointly
across its months and all role partitions.140 distinct receiving stations and
7,897 unique station-months appear in8,857 partition-cell occurrences; repeated
seeds do not increase ecological sample size.

| Model | MAE,mg/L | Source-Q90 MAE,mg/L | log1p MAE |
|---|---:|---:|---:|
| Complete retained current predictor |1.723002|8.895024|0.248207|
| Ordinary upstream attention |1.717226|8.852561|0.247090|
| Actual source dates/support |1.717158|8.855450|0.247048|
| Dates + measured phase (primary) |1.717288|8.852718|0.247085|
| Shuffled source-flow correspondence |1.717178|8.854021|0.247118|
| Previous environmental-state river branch |1.714729|8.841157|0.246694|
| Matched real upstream |1.717628|8.863108|0.247248|
| Matched nonancestor identity |1.721545|8.891079|0.247936|
| Matched strong environmental trees |1.866362|9.394458|0.275922|

Positive relative gain means lower MAE.

| Comparison | Gain,% |95% station interval,%| Improved packages/partitions |
|---|---:|---:|---:|
| Full sampling vs ordinary attention |−0.003607|[−0.044268,0.033723]|4/9;0/3|
| Full sampling vs dates only |−0.007562|[−0.047491,0.027189]|4/9;0/3|
| Full sampling vs shuffled phase |−0.006403|[−0.069071,0.043863]|4/9;0/3|
| Full sampling vs previous state branch |−0.149241|[−0.314899,−0.013576]|2/9;0/3|
| Dates only vs ordinary attention |+0.003954|[−0.025565,0.039209]|5/9;3/3|
| Matched real vs nonancestor identity |+0.227506|[−0.054639,0.502489]|7/9;2/3|

The dates-only direction is consistent at the partition mean, but its effect is
0.004%, not a material improvement. The full arm's gains versus the complete
current predictor (+0.331661%, interval[−0.056533,0.683087]) and trees (+7.987426%,
interval[5.255499,11.079260]) are accumulated procedure comparisons, not newly
caused gains from sampling dates or phase. Do not advertise the tree difference
as this module's contribution.

Q90 full sampling versus ordinary attention is−0.001777%, interval
[−0.085290,0.045957]. Its+0.030842% versus date-only and+0.014709% versus shuffled
phase both have intervals spanning zero. It remains worse than the previous
state branch (−0.130767%, interval[−0.309708,−0.005715]). Station-equal MAE likewise
does not improve:2.171274 for full sampling,2.170970 ordinary,2.170593 shuffled,
and2.168432 previous state branch. All primary tail populations exceed20 cells.

## What the coverage diagnostic shows

| Partition | Any observed upstream DOC/query cells | Both sampled-source and receiver phase/query cells | Multiple represented sample days/query cells | Median youngest source mean age |
|---|---:|---:|---:|---:|
|142|519/2283 (22.73%)|478/2283 (20.94%)|207/2283 (9.07%)|15days|
|143|667/3068 (21.74%)|535/3068 (17.44%)|274/3068 (8.93%)|15days|
|144|1840/3506 (52.48%)|721/3506 (20.56%)|991/3506 (28.27%)|14days|

Only50 of140 receiving stations have usable observed upstream DOC somewhere
in the evaluated population. Phase coverage among valid donor slots is
83.29%,84.13%,73.37%; simultaneous receiving phase remains much sparser. These
counts expose the information available to the operator. They do not prove that
coverage alone causes the null result.

Even on observed-upstream cells, full sampling vs ordinary attention is
−0.010348%, interval[−0.226381,0.169238]. On cells with both measured phases it is
+0.022294%, interval[−0.181037,0.228899],38stations. Source-age≤31days has
+0.012786%, interval[−0.244135,0.244735]. The apparent source-phase-missing subgroup
improvement involves only15stations and spans zero. No subgroup establishes a
conditional sampling-information gain. Unsupported cells reproduce the old
complete prediction exactly, rather than being improved by this river module.

## Scientific interpretation and next direction

Restoring sample timing is feasible and does not change the target task, but
**date and hydro-phase information used to allocate scalar upstream innovations
has not improved this monthly DOC predictor**. Neither the overall nor supported
subgroups establish the hypothesized gain. The result is specific to this
integration and task; it does not show that observation timing or river transport
is ecologically irrelevant. Actual dated DOC remains sparse, and metadata does
not reconstruct an unobserved daily concentration trajectory.

Together with the preceding mixing/storage experiments, this suggests stopping
minor extensions of the same frozen-query, scalar-innovation attention readout.
The next model investigation should train upstream temporal states jointly with
the receiving local/GRU correction on station-held reconstruction episodes.
An important design issue is that the retained complete model's source fitting
predictions are in-sample, although donor innovations are OOF: a new training
version should represent an unseen receiving station throughout the local
predictor and its loss, rather than only cross-fitting its scalar output head.
Keep ordinary attention and the previous environmental-state branch fixed as
references. This is a proposed next experiment, not a result or promised gain;
return to source roles for development and retain geographical/external outcomes
as outcomes of their evaluated versions. No wider deployment of this null module.

## Reproduction and checks

Use the uv-managed environment and entry point:

```bash
uv run python scripts/run_ladder.py --experiment doc-sampling-river-v1
uv run python scripts/verify_doc_sampling_river_v1.py --rebuild-inputs
uv run python scripts/analyze_doc_sampling_river_v1.py --bootstrap-draws 5000
uv run python scripts/verify_analysis_doc_sampling_river_v1.py
uv run python scripts/plot_doc_sampling_river_v1.py
```

Nine rebuilt input packages exactly match their saved caches;54 checkpoints
reproduce their predictions bitwise. Ordinary attention reproduces the previous
dynamic_lagged predictions bitwise in all packages. Fixed products are unchanged.
An independent implementation recomputes all model MAE/log/Q90 and18 primary
station intervals. Nine new tests cover result-row date weights, receiver-date
independence, hidden/future dates, actual-vs-common donor age, flow gaps/reversal,
shuffle pools, zero initialization, no support and checkpoint replay. Full suite:
1,483passed,3skipped,8warnings. Ruff and historical provenance audit pass; historical
no-sidecar/parquet-only qualifications remain as recorded. Both figures were
actually viewed; only a legend-position repair was needed. Saved execution code
was not changed after fitting. Large metadata, fitting arrays and checkpoints
remain local; scoped source/results and evaluation sources are committed.
