# Execution progress: DOC reconstruction at unmonitored stations

## Latest sampling-aware river update (2026-10-10)

Completed54 river fits on source roles142/143/144 x42/43/44 in
`doc_sampling_river_v1`, using actual source sample dates and measured daily-flow
phase with receiving sampling metadata absent. Primary MAE1.717288mg/L ties
ordinary monthly attention1.717226; incremental gain−0.003607% (station interval
crosses zero). Actual flow correspondence does not outperform shuffled metadata.
Previous environmental-state candidate1.714729 remains better; do not adopt this
module. All frozen22,571 monthly targets reconcile with raw dates. Full evidence
and proposed next joint station-held river/local training are in
`docs/doc_sampling_river_progress_20261010.md` and the experiment's English
`research_decision.md`. All9 reconstructed inputs,54 checkpoints, independent
metrics/18 intervals and both scientific figures verified. Pytest1,483passed,
3skipped; Ruff/history audit pass. No automation resumed.

## Latest confluence/storage update (2026-10-10)

Completed45 river fits on source roles142/143/144 x42/43/44 in
`doc_confluence_storage_v1`. Primary MAE1.716517mg/L improves ordinary attention
by0.041% (station interval crosses zero) but remains worse than the preceding
state branch1.714729. Explicit storage adds no gain over mixing-only. Preserve
the previous candidate and released model. Next mechanism priority is actual
upstream observation-time alignment with daily hydro events, not more
attenuation/depth variants. Full results and propagation controls are in
`docs/doc_confluence_storage_progress_20261010.md` and the experiment's English
`research_decision.md`. All9 packages,45 checkpoints, exact rebuilt priors,
independent metrics/15 intervals and two scientific figures verified;
pytest1474 passed,3 skipped; Ruff/history audit pass. No automation resumed.

## Latest river learning update (2026-10-10)

Conditional readout-crossfit development is complete:54 scalar-head fits and45
river fits in `doc_river_readout_crossfit_v2`. MAE1.714736mg/L essentially ties
the preceding upstream-state1.714729; Q90 is slightly worse. Do not adopt it.
The loss of the new readout is station-held; retained whole-model features and
components are not fully OOF. Next priority is explicit confluence/storage
message mixing. Full result, implementation repair and evidence are in
`docs/doc_river_readout_crossfit_progress_20261010.md` and the new experiment's
English `research_decision.md`. All nine packages, independent analysis and
figures verified; pytest1463 passed,3 skipped; Ruff/history audit pass. No
automation restarted.

## River structure residual pilot complete (2026-10-06)

`experiments/phase4_transfer/doc_river_structure_residual_v1` now contains all
nine source-role packages (142/143/144 ×42/43/44),36 neural fits and successful
exact source-bank/prediction replays. Training is finished; do not restart it.
The current ecological encoder, observation-aware GRU and similarity retrieval
have been extended with actual directed upstream path messages, continuous
length/confluence/drainage/storage attributes, explicit observation age and
lag0/1/3. The new river output starts at zero. Comparisons retain a fixed complete
current predictor, matched no-river refit, simple messages and rewired sources.

Equal-split K0 MAE: current1.723002, matched refit1.723029, simple1.723066,
structure1.724014, rewired1.722055, strong trees1.866362mg/L. Structure messages
raise overall error0.0587% versus the fixed current model (gain interval
[-0.5476%,0.4052%]); Q90 error improves0.1744%, with an interval crossing zero.
The existing deployed release and manuscript results stay fixed.

Useful source-development heterogeneity: paths at most50km improve2.649%
[0.617%,4.306%] against matched refit,23 unique stations; mainstems improve1.193%
[0.245%,2.560%],28 unique stations. Unsupported query months degrade0.496%.
Real and rewired source availability differ. An explicitly post-readout common-
support diagnostic gives nearby-path improvement4.848% against matched refit
[1.052%,7.210%],14 stations/570 unique station-months; its advantage over rewiring
has an interval crossing zero. These are selected ST357 source validation
results, not geographical or external confirmation.

The direct fitted river contribution improves error relative to the same
network with its river output zeroed, but changes to jointly refitted nonriver
paths offset that contribution. `research_decision.md` prioritizes a protected
current prediction and selective short-path river learning. It records the
pre-training switch from unavailable complete-model OOF targets to joint neural
training on environmental tree OOF bases and double-held donor references.

Analysis includes5000 station draws, physical profiles, path distances, source
availability, station-equal error, fitted zero-message effects and numerical
refit differences. Two PNG/PDF/SVG figures have been viewed. Full pytest1039
passed,2 skipped; Ruff and historical audit pass. Large fitted checkpoints and
candidate arrays remain local; related small products and code go in the scoped
pending submission list. No recurring automation was created or resumed.

## River structure atlas complete (2026-10-06)

The user authorized a physical river structure and DOC atlas after discussing
the current release's absence of neural river messages. New work is in
`experiments/phase4_transfer/doc_river_structure_atlas_v1`; no model was fitted.
Source-role development observations/predictions and separate historical
validation messages are used; completed geographical/external tests stay fixed.

All357 stations match full NHDPlus VAA topology. Overlapping profiles include
92 low-order tributaries,184 chain settings,161 major-confluence vicinities,
69 integrated mainstems and184 upstream lake/reservoir-path settings. The206
stations lacking sampled upstream neighbours must not be treated as physical
headwaters; VAA marks12 station reaches as headwaters. All324 monitored edges
resolve along physical mainstems: median105.104km,49 intervening mapped junctions
and3 major junctions. Two station positions use flagged midpoint approximations.

At141 fixed source station pairs, seasonal-anomaly DOC correlation medians are
0.332/0.155/0.106/0.085/-0.055 at0/1/3/6/12 month lags. This is descriptive
association, not a travel-time estimate. Current-model source-validation gain
against strong trees is10.26% in low-order tributaries,10.14% in chain settings,
5.06% near major confluences,3.10% in mainstems and6.20% for storage-path sites.
Those gains come from the complete similarity-retrieval model, not river GNNs.
Separate historical upstream-message validation gains remain negligible.

The atlas contains station/path tables, DOC behaviours,5,000-draw paired
station analyses, balanced-lag diagnostics and four inspected PNG/PDF/SVG
figures. Independent verification passes; pytest1029 passed,2 skipped; Ruff
and the historical audit pass. English `research_decision.md` and
`next_experiment.md` describe an observation-supported structure-aware river
residual with conveyance, tributary mixing and storage-memory operators.
Existing release and manuscript results remain unchanged. Related new files
are added to the scoped pending submission list; Git permissions remain read-only.

## Latest delivery: current-source release complete (2026-10-06)

The current-availability model has completed five-region/five-seed confirmation,
five-seed deployment fitting, updated external replication, two temporal checks
and manuscript v3. All fixed packages are complete. Do not restart completed
training or reinterpret this delivery as a new source-development experiment.

Internal equal-region K0 MAE is 2.248532 mg/L: 4.109% lower than strong matched
trees [1.113%, 7.109%], with all five regions improving, and 1.478% lower than
the retained complete release [0.416%, 2.672%]. The working 5% objective against
both comparators is not reached. In the already inspected external HUC02040104,
the five-member release scores 0.910577 mg/L versus 0.914994 for the preceding
release; the 0.483% increment has an interval crossing zero. Trees retain lower
cell-weighted external error (0.840835 mg/L). K5 reduces the new release's fixed-
query error by 19.351% against its own K0. Source-calibrated external coverage
is 83.267%, with median interval width 2.737396 mg/L.

Deliverables: `doc_current_source_portable_v2_r1/release.json`, its saved five
member exports, `scripts/portable_current_source_ensemble_v2.py`, updated external
products and analysis, `doc_current_source_temporal_v3_r1`, and
`docs/paper/latex/unmonitored_doc_draft_v3.tex` / `.pdf`. The inference facade uses
named new sites and frozen source preprocessing; K0 uses no receiving chemistry.
The selected branch has no neural river-edge messages: attention represents
ecological/hydrological source similarity. Geographical withholding is within
ST357; the external basin is disjoint but was seen for the preceding release.

Research delivery is complete. Git submission remains blocked by the current
read-only Git filesystem; the scoped pending file list preserves the work.
The completion-triggered request to delete the DOC heartbeat was rejected by
the tool: "MCP tool call requires approval, but approval policy is never."
The schedule is still enabled; no filesystem or UI workaround is used. Further
unchanged heartbeat runs should remain quiet and not repeat completed research.
Future mechanism development returns to source roles 142/143/144. Current-source
coverage, concentration bias and high DOC provide concrete next questions;
completed geographical/external/temporal predictions stay fixed.

## Renewed source-only performance development

The user requested continued research after delivery of the completed deployment
and manuscript. The previous evaluation cycle below is preserved. Its completed
geographical/external/temporal runners are not restarted.

`doc_concentration_hydro_v1`:9/9 small source-OOF calibration packages complete,
with11 arms and5,000 paired station-bootstrap analysis. Conditional calibration
gives integrated MAE1.782415 vs retained1.769053: -0.76% gain
[-1.85%,0.37%],0/3 positive partition directions. Its tree-only improvement is
0.40% [-0.76%,1.61%]. Retain the experiment; do not add its correction to the
deployed model. The analyzer's NaN-equality comparison was repaired without
altering predictions or the saved training code. Figures and an English decision
record document the attempted concentration/hydro mechanism.

`doc_log_concentration_residual_v2`:9/9 complete, saved-state replay and5,000
station draws complete. Integrated MAE1.777794 vs retained1.769053: -0.494%
[-1.414%,0.402%],1/3 positive directions. Q90 is slightly worse. Do not adopt
the log-concentration neural readout. Keep native mg/L MAE/tail training and the
current release. Its English `research_decision.md` distinguishes gains over
the older model/trees from failure to improve the actual retained full model.

V1 optimized its first neural fit but stopped before product export because
the existing ecological-memory helper needed an explicit month count. V2
supplies that unchanged dimension and saves a resumable fitted-neural stage
before memory export. The failed V1 sources/config/log are retained; no V1
prediction package was completed. Scientific settings are unchanged.

`doc_regional_source_training_v2`:9/9 region-hidden source training packages,
complete replay,5,000 station draws and inspected plots complete. Integrated
MAE1.833923 vs retained1.769053: -3.667% [-5.500%,-1.929%],0/3 positive
directions. The regional tree worsens by1.717% versus strong station-hidden
trees. Its neural residual improves its own regional tree by3.381%, but the
full version still regresses. Do not adopt regional training as tested. V1's
HUC8/HUC12 startup failure occurred before fitting; its saved sources/config/log
and V2's repair remain.935 tests, two skips; Ruff and historical audit passed.

`doc_nonlinear_native_residual_v1`:9/9 source packages, replay and5,000 station
draws complete, figures inspected. Nonlinear complete MAE1.760777 vs
retained1.769053:0.468% [-0.505%,1.581%],7/9 positive package directions,
2/3 partitions. Matched neural-only gain0.460%; source evidence remains
uncertain over the actual retained model.938 tests, two skips; Ruff and audit
passed. A console-column-name reporting error was repaired without changing
predictions or calculations; failed log and source snapshot are preserved.
The fixed32-unit head is a candidate; the portable release is not replaced.

`doc_nonlinear_geographical_replication_v1`:25/25 fixed packages complete,
saved-state replay,5,000 station draws and inspected figures complete. Equal-
region complete MAE2.278136 vs retained2.282266:0.181% [-1.320%,1.630%],
3/5 positive directions. Native-only worsens by0.532%; Q90 by0.232%. K3/K5
are slightly worse. Do not replace the current portable release with this head.
These already evaluated ST357 geographical tasks are retrospective replication,
not independent external validation. The failed analyzer import and its repair
are preserved; training and predictions were unchanged.938 tests, two skips;
Ruff and the historical audit passed. Read its English research decision.

`doc_ecological_composition_v1`:9/9 complete, replay,5,000 station draws and
inspected plots complete. Detailed treesMAE1.868740 vs matched expanded
totals1.893667:1.316% [0.020%,2.639%],3/3 positive partitions. But retained
strong trees1.866362 are still slightly better, and the retained full model
remains1.769053. Do not report the matched-control improvement as an overall
upgrade or replace the existing tree.941 tests, two skips; Ruff/audit passed.

`doc_composition_encoder_v1`:9/9 matched encoder packages complete, bitwise
native/integrated replay and5,000 station draws complete, plots inspected.
CompleteMAE1.769340 vs retained1.769053: -0.016% [-0.096%,0.061%]. Q90
gain0.082% [0.016%,0.198%] is small; no overall upgrade. Both encoder arms
use22 zero new input columns and the original tree/OOF base.944 tests, two
skips; Ruff/audit passed. A synchronous progress callback name was explicitly
bound after fitting for Ruff; original execution copies remain. Do not refit
or deploy this candidate. Read its English research decision and source error
diagnostics: Q90 cells9.93% carry43.76% of absolute and84.67% of squared
error; persistent station mean bias20.84% versus within-station variance79.16%.
These are descriptive source-validation error budgets, not causal attribution.

`doc_antecedent_hydro_v1`:9/9 packages complete, saved forest replay,5,000
station draws and inspected plots complete. StatesMAE1.874402 vs matched
availability1.890018:0.826% [0.423%,1.230%],3/3 positive partitions. Yet
the retained strong tree1.866362 is better: -0.431% gain [-1.150%,0.163%],
0/3 directions. Do not change the environmental reference based on this study.
947 tests, two skips; Ruff/audit passed. English research decision complete.

`doc_antecedent_residual_v1`:9/9 complete, bitwise saved-state replay,5,000
station draws and inspected plots complete. FullMAE1.767694 vs retained1.769053:
0.077% [-0.053%,0.217%],3/3 positive partitions. Matched availability gain
0.031% [-0.029%,0.095%]. Q90 gain0.101% [-0.077%,0.357%]. Keep the
retained portable release; no material upgrade or new geographical run is
justified.947 tests, two skips; Ruff/audit passed. All four recent source studies
completed9 packages each. Two neural analyzers now count directions separately
within overall/Q90 subsets; archived originals and repair records preserve the
unchanged predictions, MAE and CIs. English research decisions are complete.

`doc_source_auxiliary_states_v1`:9/9 packages complete (18 auxiliary and18 DOC
fits), source-fold scaler/role/checkpoint recomputation, bitwise native/complete
replay,5,000 station draws and inspected figures complete. Complete MAE1.788135
vs retained1.769053: -1.079% gain[-1.886%,-0.331%],0/3 partition directions.
It also loses1.034% to the matched label-shuffle arm. True pH/EC auxiliary MSE
improves in all nine packages, but that representation does not improve DOC
in this sequential pretraining design. Do not deploy it or start geographical
replication.950 tests, two skips; Ruff/audit passed for its completed version.
Read its English research decision. Large source fits and all old results remain.

`doc_reference_trajectory_v1`:9/9 packages complete,45 dense nested source-
hidden forest fits and18 DOC residual fits. Full-history complete MAE1.769113
vs retained1.769053: -0.0034% gain[-0.1481%,0.1564%],2/3 positive partitions.
Matched current-reference gain0.0212%[-0.2053%,0.2508%]; Q90 gain versus retained
-0.0343%[-0.1477%,0.0543%]. No performance upgrade. Bitwise native/complete
replay, source OOF/scaler/fold checks,5,000 station draws and inspected plots
are complete.954 tests, two skips; Ruff/audit passed. Keep the current portable
release; do not run a geographical matrix for this projection. Large45 forest
fits and dense OOF grids remain local. Read its English research decision.

`doc_native_reference_objective_v1`:9/9 packages (27 fixed reference fits),
saved-state replay,5,000 station draws and inspected figures complete. Native-
target ExtraTrees MAE2.161084 versus retained full1.769053 worsens22.16%
[12.62%,34.01%],0/3 positive partitions; versus retained forest1.866362 it
worsens15.79%. Its Q90 improves3.47% against the full model but does not offset
the overall regression. Log/native L1 boosting MAE1.891568/1.924264 also lose
to the retained full model. Keep the log-trained reference and existing neural
residual.959 tests, two skips; Ruff/audit passed. A forest-sum precision test was
repaired with exact inputs/tree states and1e-12 reduction tolerance; no fitting
changed. See its English research decision and preserved initial test log.

`doc_leaf_distribution_reference_v1`:9/9 packages, exact saved distribution/tree-
state replay,5,000 station draws and inspected figures complete. Conditional
median MAE1.853846 versus log forest1.866362:0.671% gain[-0.841%,2.344%],
seven of nine packages and two of three partitions positive. Q90 is1.567% worse
against the forest. The bare median reference still loses4.793% to the actual
retained complete model1.769053. Native distribution mean2.131545 is worse.
Do not deploy either readout or claim reliable reference superiority.962 tests,
two skips; Ruff/audit pass. Read the English research decision.

One exploratory integration was completed in `doc_leaf_median_residual_v1`:
refit the SAME ecology self encoder, observation GRU, native residual head and
memory fusion on source station-OOF conditional medians. The small central
development signal motivates testing whether the retained tail-aware residual
can repair the median's Q90 loss; it is not a declaration that the preceding
reference study passed. Use142/143/144 ×42/43/44, unchanged30-epoch/patience5
budget, tail weight2 and source-only selection. Verified complementary forests
already saved by `doc_reference_trajectory_v1` supply the OOF partition geometry;
repopulate leaves only from that fold's source labels. Do not retrain the same45
forests or use in-sample residuals. The new OOF-isolation and existing neural
contracts pass (16 targeted tests). No old geographical/external runner is
restarted, and the current release remains the comparator.

All nine median neural packages, five-fold OOF exclusion/source normalization
checks, bitwise neural/memory replay,5,000 station draws and inspected figures
are complete. Integrated MAE1.778791 vs retained1.769053: -0.551% gain
[-1.529%,0.415%],0/3 positive partition averages. Q90 is0.847% worse
[0.014%,1.948%]. Its4.05% improvement over the median reference and4.69%
over strong trees do not establish an upgrade over the current complete model.
Keep the current portable release; do not replicate this candidate geographically.
963 tests, two skips; Ruff/audit pass. Read its English research decision.

Completed source-only experiment: `doc_extended_optimization_v1`. The nine retained
30-epoch fits stopped after7–24 epochs under patience5. Test the SAME effective
model with120 maximum epochs and patience15, preserving the original log tree,
OOF references, all inputs/normalization, initial weights, loss and learning
rates. No median reference or nonlinear head is carried forward. The common
optimizer prefix must match the retained trajectory. The longer validation
minimum is expected to be non-worse by construction; source magnitude and
later selected epochs determine whether confirmation is worthwhile. This is
an optimization hypothesis, not an established explanation for existing error.
Its separate study plan was saved before fitting. All9/9 fits, common trajectory
checks, bitwise neural/memory replay,5,000 station draws and inspected plots are
complete. Complete MAE1.767218 vs retained1.769053:0.104% gain
[-0.068%,0.281%]; Q90 gain0.041%[-0.087%,0.196%]. Three fits select a later
checkpoint; six retain the original. No material upgrade; keep the portable
release and do not repeat geographical/external confirmation.964 tests, two
skips; Ruff/audit pass. Read its English research decision.

`doc_joint_source_states_v1`:9/9 packages (18 DOC fits) complete, bitwise
source/auxiliary-head/native/memory replay,5,000 station draws and inspected
figures complete. Complete MAE1.773236 vs retained1.769053: -0.236% gain
[-0.492%,-0.027%],0/3 positive partitions and2/9 packages. Matched shuffle gain
-0.225%[-0.512%,0.040%]. Q90 gain0.021%[-0.092%,0.171%]. Real auxiliary
MSE improves for both targets in all nine fits, but does not improve DOC transfer.
Do not adopt this model, scan auxiliary weights or repeat geographical/external
confirmation. Source-only joint supervision and earlier sequential pretraining
are both completed negative transfer experiments.967 tests, two skips;
Ruff/historical audit pass. Read its English research decision.

`doc_longer_history_v1`:9/9 source fits complete. Extending the existing
GRU from12 to24 months gives complete MAE1.772521 vs retained1.769053:
-0.196% gain[-0.729%,0.381%],1/3 positive partition averages. Q90 gain-0.023%
[-0.371%,0.277%]. All initial weights, inputs and parameter counts match; neural,
memory and same-weight truncation predictions replay bitwise. Figures inspected;
969 tests, two skips, Ruff/audit pass. Its own fitted24-month weights beat their
12-month inference truncation by2.991%[1.297%,4.787%], showing use of older
history but not an upgrade over the independently trained retained12-month
model. Close this change without a36-month scan or geographical/external rerun.
Read its English research decision; release unchanged.

`doc_full_encoder_v1`:9/9 source fits complete. Tuning both existing self
layers and ecology gives complete MAE1.769569 vs retained1.769053:
-0.029% gain[-0.302%,0.245%],1/3 positive partitions and4/9 packages.
Matched neural-only gain-0.006%[-0.380%,0.348%]; complete Q90 gain
0.016%[-0.174%,0.225%]. All first-self layers update, with trainable
parameters35143 vs31559; allocated architecture, initial weights,12-month
GRU, reference/OOF inputs and30-epoch/patience5 budget remain unchanged.
Bitwise native/memory replay, source-isolation checks,5,000 station draws
and inspected figures are complete.971 tests, two skips; Ruff/audit pass.
Keep the retained release. No learning-rate scan or geographical/external
rerun for this candidate. Read its English research decision.

`doc_source_synchrony_v1`:source-only residual audit complete. Nearest
ecological source station pairs have mean same-month correlation0.166158
versus0.042386 after calendar-preserving year shuffle; real-minus-shuffle
is positive in all three source partitions. Lag1/3 signals are weaker.
An average201/232 source recipients have a nearest pair with at least12
common observed months. Receiving-site labels were not inspected. This is
descriptive source information, not a measured new-site prediction gain or
physical transmission coefficient.973 tests, two skips; Ruff/historical
audit pass. The source-generated figure was inspected and its label spacing
repaired; the initial layout remains. Read its English research decision.

`doc_source_innovation_transfer_v1`:9/9 source-only information probes complete.
Same-month source innovations improve complete MAE1.769053→1.763691:
0.303%[-0.107%,0.747%],2/3 partition averages and6/9 packages positive.
Q90 gain0.593%[-0.050%,1.448%]. All historical arms select zero correction.
Current donor support covers92–93% of query cells; matched current/past support
covers84–85%. Receiving sites have no DOC/pH/conductance inputs. Source library,
selection and predictions replay bitwise;5,000 station draws and inspected
figures complete.977 tests, two skips; Ruff/historical audit pass. No adoption
or geographical/external rerun. Read its English research decision.

Current mechanism: `doc_source_innovation_learning_v1`. Its separately saved
plan tests the SAME ecological encoder/GRU/native residual with three new
current-month readout inputs: source innovation, support count and weight mass.
Real, earlier-season and availability-only arms have identical shapes and67
additional trainable parameters; two enriched strong-tree controls use the same
information. No new backbone or graph depth. For each query fold A and donor
fold B, a new reference omits BOTH folds. The45 previous complementary forests
omit only one fold and cannot directly supply this nested bank. Ten new pair
forests per package are cached; each source library excludes its query fold.
Receiving chemistry remains absent. A fixed1mg/L innovation scale avoids
cross-bank label-dependent normalization.

All nine packages (90 pair references,27 neural fits and18 enriched-tree fits)
are now complete. The first package's real-cohort hidden-label contract and
neural/memory/library replay pass.979 tests, two skips; Ruff/historical audit
pass. The initial analyzer falsely rejected tuple/list-equivalent interaction
indices; fitting and predictions were unchanged. Use the separately saved
`scripts/analyze_doc_source_innovation_learning_v1_r1.py`, keeping its original
and execution snapshot. `design_clarification.md` records that all SEVEN retained
daily-hydro interactions are preserved and the innovation index38 is appended;
the old plan's reference to four was a prose error. Extra inputs38→41 and
trainable parameters31559→31626 agree with the executed model.

All9-package saved-state replays,5,000 station draws and inspected figures
are complete. Complete MAE1.761078 vs retained1.769053:0.451% gain
[-0.045%,1.032%],2/3 partitions and7/9 packages improve. Q90 gain0.840%
[0.150%,1.906%], also2/3 partitions and7/9 packages. Matched neural-only
MAE gain0.509%[-0.026%,1.122%]; Q90 gain0.850%[0.111%,1.967%].
Real source information beats historical/availability complete point estimates
in all three partition averages, but their overall intervals still cross zero.
Same-information trees improve only0.037% over bare strong trees. The5.61%
complete-versus-enriched-tree gap includes the retained model's existing ability;
it is not the isolated source-information gain. No-donor groups also improve,
so the entire effect cannot be assigned to current donors alone. Read the English
`research_decision.md` and `validation_summary.json`. The initial plot layout is
retained; its legend overlap was repaired and the final figure inspected.

Active next version: `doc_source_innovation_geographical_v1`, with a study plan
saved before its first fit. The fixed same-month structure (+67 parameters,
12months,30epochs/patience5) proceeds to five established HUC4 roles
1013/1019/0708/1030/1101. Reuse verified parent preprocessing/initial weights/
OOF reference; fit ten new double-held-fold references, one neural model and
one equally informed tree per package. First perform huc4_1013_seed42 input/
replay checks, then all15 packages at42/43/44; complete the SAME candidate and
controls at45/46 (25 total). This is retrospective ST357 geographical replication,
not independent external validation. Numerical test outcomes do not change the
fixed continuation or choose per-region/K winners. Receiving chemistry remains
absent; source seasonal/library statistics are spatial-training statistics.

Training entry: `run_ladder.py --experiment doc-source-innovation-geographical-v1`.
Check this NEW root's training.log/progress.json/runs/*/complete.json. Do not
restart earlier completed runners. The first technical package is complete; its nested references, source library,
full-grid neural/memory predictions and support adapters replay successfully.
All25 fixed packages completed (250 pair references,25 neural fits and25
enriched-tree fits). Both training invocations exited normally; no training
remains active. All25 source-library/nested-reference/neural/memory/support replays,5,000
paired station draws and inspected figures are complete. Equal-region complete
MAE2.272858 vs retained2.282266:0.412% gain[-0.044%,0.924%],4/5 regions
improve. Q90 gain0.464%[0.292%,0.670%],5/5 regions. Neural-only overall
MAE worsens0.106%; equally informed trees improve1.295% over bare trees.
The complete gain over the OLDER model is3.704%, not the incremental0.412%.
K3 improves0.776% with a positive paired interval; K5 improves0.386% with
an interval crossing zero. Q90 recall remains poor outside1013; two regions
have fewer than20 high cells. The5% working goal is not met. Keep the current
portable release, external products and main manuscript. Read its English
research_decision.md and validation_summary.json.979tests,two skips; Ruff
and historical audit pass for that executed version. Geographical plot label/
comparison filtering was repaired; fitting and numerical analysis unchanged.

`doc_current_source_attention_v1`:all9 source packages complete (27 neural fits,
90 reused double-held-fold references, no new forests). Candidates/roles/initial
weights/neural/memory/diagnostics replay bitwise;5,000 station draws and inspected
figures complete. CompleteMAE1.756645 vs retained1.769053:0.701% gain
[0.220%,1.239%],3/3 partitions positive. Q90 improves1.041%
[0.351%,2.131%],3/3 partitions and9/9 packages. Matched earlier-season
complete gain0.731%[0.244%,1.291%], all nine directions positive.
Adaptive-versus-fixed COMPLETE gain0.163%[-0.197%,0.495%] is uncertain,
only1/3 partitions positive. Matched NEURAL-ONLY Q90 improves0.449%
[0.157%,0.973%],2/3 partitions positive. Thus current information helps,
while adaptive selection has a conditional tail signal, not established overall
complete superiority.985tests,two skips; Ruff/historical audit pass. Saved trees
see aggregate innovations, not individual donor hydro keys. Read its English
research_decision.md, design_clarification.md and validation_summary.json.
The portable release and preceding external/main manuscript are unchanged.

`doc_current_source_attention_geographical_v1`: all25 fixed packages complete,
75 neural fits and250 reused double-held reference forests; no new forests.
All candidates, roles, initial states, full-grid components, support and attention
products replay bitwise. Five thousand paired station draws and inspected figures
are complete. Complete equal-region K0 MAE2.266609 vs retained2.282266:
0.686% gain[0.031%,1.397%], four of five regions. Q90 improves0.929%
[0.431%,1.511%], all five directions. Gain over strong bare trees is3.338%
[0.391%,6.137%]; over the OLDER complete model3.969%, which includes earlier
improvements. Overall adaptive-vs-fixed gain0.262%[-0.059%,0.590%] remains
uncertain; matched Q90 gain0.485%[0.142%,0.908%] is positive in all five
regions. Native-only overall performance is essentially unchanged from retained,
while its Q90 improves0.921%. Complete K3/K5 improve1.281%/0.937% with
positive paired intervals. The5% overall goal is not reached; portable release,
external products and manuscript stay at their evaluated versions. This is
retrospective ST357 replication, not external validation. Read its English
research_decision.md and validation_summary.json. The executed fitting version
passed985 tests; the current workspace now passes988, two skips. Ruff and
historical audit pass. Do not restart this completed runner.

`doc_relative_source_attention_v1`: all9 source packages complete,27 neural
fits and90 reused double-held reference forests; no new forests. Its plan
preceded opening geographical attention numerical results. Only individual donor
values changed to source-only, season-centered log1p residuals converted through
one plus the frozen receiving environmental prediction. Native loss, aggregate
inputs, ecology/GRU,37,900 parameters and30epochs/patience5 stay fixed.
CompleteMAE1.748102 vs retained1.769053:1.184% gain[0.610%,1.839%],
all nine packages/three partitions/three seeds positive. Q90 gain1.580%
[0.744%,2.864%], all nine positive. Gain over native-current complete is0.486%
[0.166%,0.827%], all nine positive. Native-only gain over native-current is
0.525%[0.223%,0.848%], likewise all nine positive. Relative learned current
and fixed-prior COMPLETE are tied:0.006%[-0.530%,0.498%], so learned allocation
is not established. Every source package replays bitwise;5,000 station draws,
inspected plots and English decision complete.988tests,two skips; Ruff and
historical audit pass. These are selected source-validation results. Keep old
portable/external/main manuscript products; do not restart this source runner.

Completed version: `doc_relative_source_attention_geographical_v1`. Carry all three
relative arms to five fixed HUC4 regions1013/1019/0708/1030/1101 ×seeds42–46,
with75 neural fits and250 reused references. No new forest or tuning. Compare
with actual retained complete, native-current attention, preceding source model,
strong trees and matched relative historical/fixed arms. Preserve full vs native,
all-observed primary K0, fixed-query K0/1/3/5, Q90/recall/bias/station-equal
and ecology/hydro/source-support/reference-concentration strata. This remains
retrospective ST357 replication, not independent external validation.
The first1013/42 technical package completed and replayed bitwise, including
nested libraries, initial weights, full-grid components, attention and K support.
All25 fixed packages are complete; the first was retained without refitting.
988tests,two skips; Ruff and the historical audit pass for this executed version.
Do not modify fitting code or source decision after the saved snapshot.
Entry: run_ladder.py --experiment doc-relative-source-attention-geographical-v1.
All25 candidate/library/nested-exclusion/component/attention/support products
replay bitwise.5,000 station draws, inspected figures and English research
decision are complete. Equal-region completeMAE2.258278 vs retained2.282266:
1.051%[0.361%,1.754%], four of five regions improve. Q90 gain1.050%
[0.542%,1.602%], all five improve. The isolated increment over preceding
native-value attention is0.368%[0.151%,0.566%], four positive regions.
Complete gains over matched fixed/earlier-season arms are0.429%/0.800%
with positive intervals. Native-only overall gain over retained is0.269%
[-0.176%,0.620%]; native-only vs fixed is not established. Complete overall
benefit includes validation-selected ecological-memory fusion. Station-equal
gain0.746%;66/104 stations improve. Q90 recall remains poor outside1013,
including zero in0708/1019;1030/1101 have unstable15/11 tail cells.
Gain over strong trees is3.693%[0.722%,6.506%]; the5% objective is unmet.
Do not restart the completed runner or replace previously scored external
products. Read this version's research_decision.md and validation_summary.json.
New mechanism development again returns to source roles142/143/144. Never
select region/K winners or retune from completed external DOC scores.

Completed source version: `doc_ratio_source_attention_v1`. Its plan preceded
opening the relative geographical numerical analysis. Replace only the source
correction's first-order inverse by `(1+B)*expm1(u)`, preserving the native
local readout/objective,37,900 parameters, inputs, initialization, controls and
30epochs/patience5. This does not repeat whole-model log-output training.
Source roles142/143/144 ×42/43/44,27 neural fits and90 reused references;
no new forest. Zero correction/derivative, future-input and reload contracts
pass. All nine fixed packages, full replay,5,000 station draws, inspected
figures and English decision are complete. Exact completeMAE1.747190 versus
first-order1.748102:0.052%[-0.209%,0.315%], only two positive partition
averages. Native-only worsens0.022%, likewise uncertain. Exact current-source
gain over retained is1.236%[0.570%,1.975%], but it includes previously
demonstrated relative transfer. Do not count this as the operator's increment.
Mean curvature correction is0.00610mg/L. Keep first-order; close this version
without a geographical matrix or exponent scan.992 tests,two skips;
Ruff/historical audit passed for the fitted version. Do not restart it.

Completed source version: `doc_source_level_attention_v1`. A separately saved
plan tests information removed by source season centering: full relative OOF
residual, full fixed-prior, full earlier-season and seasonal-mean-only donor
values inside the unchanged first-order model. Seasonal-only retains current
aggregate readout features; it does not remove all current source information.
Source roles142/143/144 ×42/43/44,36 neural fits,90 reused references,
37,900 parameters and30epochs/patience5. All donor seasonal means exclude
query fold A, with each donor B's reference excluding both A and B. No receiving
DOC/pH/EC or additional forest fitting. Necessary seasonal reconstruction,
candidate equality, query exclusion, future-input and save/reload tests pass.
All nine packages, bitwise fitted-state replay,5,000 station draws, inspected
figures and English decision are complete. Full current source values give
completeMAE1.736968 vs retained1.769053:1.814%[1.031%,2.750%], all nine
packages/three partitions/three seeds positive. Isolated gain over anomaly-only
is0.637%[0.084%,1.284%], eight packages and two partitions positive.
Native-only isolated gain0.659%[0.033%,1.384%]. Full current improves
seasonal-only complete by0.628%[0.256%,1.034%]; this tests individual donor
values while retaining current aggregate features. Additional Q90 gain over
anomaly-only is not established:0.050%[-0.662%,0.736%]. Gain over fixed
prior0.341%[-0.101%,0.869%] likewise remains uncertain.995 tests,two skips;
Ruff/historical audit pass. Do not restart this completed source runner or
modify its source decision during geographical fitting.

Completed geographical version: `doc_source_level_attention_geographical_v1`.
All25 packages,100 neural fits,250 reused references, exact replay,5,000 station
resamples, inspected figures and English decision are complete. Full current
source completeMAE2.266236 versus anomaly-only2.258278: gain-0.352%
[-0.996%,0.364%], only one positive region. Q90 gain-0.209%
[-0.476%,0.039%]. The source-development seasonal-bias increment did not
reproduce geographically. Against actual retained complete, overall gain0.702%
[-0.336%,1.883%], four positive regions; Q90 gain0.843%[0.332%,1.395%],
all five positive. Versus strong trees, gain3.354%[0.263%,6.448%].
K3/K5 are worse than anomaly-only by0.896%/0.846%, both supported paired
intervals; K5 worsens in all five regions. Keep anomaly-only relative attention
as the stronger fixed geographical candidate. No regional/K winner mixture.
This is retrospective ST357 geography, not external validation. Do not restart
this runner or change its evaluated model. Old portable/external scores remain.

Completed source-only version: `doc_source_monthly_hydro_attention_v1`.
The source temperature/discharge keys add256 zero-initialized coefficients to
otherwise unchanged full-source attention. Current/source-mean/zero-input arms
use38,156 parameters and30epochs/patience5. All9 packages/27fits/90references,
bitwise replay,5,000 station draws, viewed figure and English decision are done.
Current-key completeMAE1.736354 versus preceding full source1.736968:
0.035%[-0.178%,0.242%], two positive partitions; native-only0.017%, uncertain.
Current values do not beat source means; Q90 is0.151% worse with a supported
interval. New weights learned, but added month-specific source hydro does not
establish incremental value. Close this mechanism without geographical training
or a tuning scan.999 tests/two skips, Ruff/historical audit passed.

Completed source-only version: `doc_current_availability_attention_v1`.
All9 packages/18neural fits/90reused references complete, bitwise replay,
5,000 paired station draws, inspected PNG and English decision saved. Same20
candidate identities, keys, receiving query, matched aggregate inputs and37,900
parameters; current-only availability opens finite donor DOC even without an
older same-season observation. CompleteMAE1.723002 vs matched full1.736968:
**0.804%[0.277%,1.335%]**, all3 source partitions and seed means positive.
Native-only0.790%[0.177%,1.434%]. Actual current values beat expanded seasonal
values by1.093%[0.582%,1.610%]. Accumulated gain vs retained complete2.603%
[1.642%,3.681%], and vs strong trees7.681%[4.958%,10.785%]; these include earlier
changes. Complete Q90 increment vs matched full0.364% is uncertain; accumulated
Q90 vs retained1.988% is supported. Current donor-supported cells rise to92–93%
from84–85%. Source-only development, not geographical confirmation.

Completed fixed geographical version:
`doc_current_availability_attention_geographical_v1`:25 packages/50neural
fits/250reused references, no forest; all bitwise replayed,5,000 station
analysis, inspected final PNG and English decision done. Complete K0 MAE
2.248532 versus matched full-source2.266236: isolated0.781%[0.282%,1.341%],
4/5 regions positive. Against actual retained complete2.282266:
**1.478%[0.416%,2.672%]**,4/5. Against strong trees2.344885:
**4.109%[1.113%,7.109%]**,5/5. Working5% goal over both is not reached.
Actual source values beat same-availability individual seasonal means0.968%
[0.455%,1.547%], all5positive. Against preceding anomaly-only complete2.258278,
0.432% gain has an interval crossing0; Q90 worsens0.292%, uncertain.
Q90 versus retained improves0.761% with supported interval. Native-only
isolated increment0.816% is supported; native Q90 is0.438% worse than
anomaly-only with supported interval. Preserve all complete procedures.
Fixed-query K5 improves0.879% versus retained; K3 is0.818% worse than anomaly-only
(supported). Do not combine regional/K winners. Current source access increases
supported test cells strongly in1013 (65.5%→86.7%) but1019 still has only62.6%
supported; remaining bias and tail detection need source-role research. This is
ST357 geographical withholding, not external validation.1002 fitting-version
tests/two skips; subsequent wider-source version1005 tests/two skips, Ruff and
historical audit passed. Old portable/external scores remain unchanged. Do not
restart this completed runner.

Completed source-only version: `doc_wide_source_attention_v1`. Its study plan was
saved before inspecting the pending current-availability geographical analysis.
Test60 ecological candidates versus the preceding20, retaining the same
nearest-five distance scale, prior, keys, receiving query, matched aggregate
inputs,37,900 parameters and30epochs/patience5. Actual and seasonal-only
wide source values,142/143/144 ×42/43/44,18neural fits/90reused references,
no forest. Original20 form the wider pool prefix. Three meaningful contracts
pass (prefix/scale, query/donor exclusion, variable-width attention/new nodes/
save-load/future inputs). The142/42 technical package completed and replayed bitwise; all9 fixed source
packages completed (18neural fits), retaining the first without refitting.
All-package bitwise replay,5,000 station analysis, inspected PNG and English
decision are complete.1005 tests/two skips, Ruff and audit passed.
Wide completeMAE1.726834 versus preceding20-candidate1.723002:
**-0.222%[-0.766%,0.302%]**,4/9 packages,2/3 partitions and1/3 seed means
improve. Native-only-0.273% is also uncertain. Usable current donors increase
to8.5-11.5 per query and supported cells to98.6-99.4%, but access does not
produce an isolated error gain. Actual values beat wider seasonal-only values
1.142% with supported interval; this does not establish widening value.
Close the60-candidate probe without geography or candidate-count tuning.
Retain the20-candidate current-availability model. Its accumulated advantages
over older baselines must not be attributed to widening. Do not restart this
completed runner or alter its saved execution sources.

Completed source-only version: `doc_source_value_key_attention_v1`.
Attention previously sees donor residuals only as values; append current source
OOF log1p residual and its calendar-month mean to the selection keys, scaled by
a frozen permitted-source RMS. Query fold A is excluded from each library;
donor references exclude both A and donor fold B. Twenty candidates, existing
query/aggregate inputs/12-month GRU/native objective/fusion remain unchanged.
New key coefficients start at zero;128 parameters are added,38,028 total.
Current/seasonal-only/zero key controls retain identical actual source values
and availability. Source roles142/143/144 × seeds42/43/44,27neural fits and
90reused references, no forest refit. The142/42 technical package completed
and replayed bitwise; continue the fixed nine-package matrix without refitting
the first. Four contracts passed before fitting (source-only RMS, receiving
label exclusion, zero-coefficient compatibility/gradients, arbitrary-node
save/load and causal fixed-library access). All9packages/27neural fits are now
complete;1009 tests/two skips, Ruff and historical audit passed. All-package
bitwise replay,5,000 station analysis, inspected final figure and English decision
are complete. Current-key completeMAE1.725912 vs preceding/zero-key1.723002:
**-0.169%[-0.525%,0.155%]**,5/9 packages,1/3 partitions and1/3 seed means
improve. Q90 gain0.172%[-0.172%,0.563%] is unresolved. Seasonal keys give
1.725698; current keys add no established benefit. New key coefficients do learn
(norm0.287-0.840); zero-key products reproduce the preceding model bitwise in
all9 packages. Close without geography or more key-capacity scans. Retain the
20-candidate current-availability model. Receiving DOC/pH/conductivity remain
unavailable. Do not restart this completed runner.

Completed deployment work: `doc_current_source_portable_v2`. Its saved study plan
fixes the geographically confirmed current-availability architecture; wider
candidates and source-state keys are excluded. The new named-site facade
`scripts/portable_current_source_doc_v2.py` reuses saved tree preprocessing and
ecological memory, adding the41-feature readout, native aggregate innovation
library, relative individual library and calendar-aligned source daily hydro.
It accepts arbitrary new station IDs, months and explicit K-support; K0 ignores
receiver DOC/pH/conductance. Components distinguish local readout, dynamic source
observations and static memory. Four contracts pass, including an actual
geographical-fit replay, hidden/future inputs, new N/calendar and save/load.
The25-fit geographical facade replay is complete (maximum error2.49e-14mg/L);
1013 tests/two skips, Ruff and historical audit passed. Preserve its initial failed
bitwise forest serialization log and precision repair. Actual parallel forest
reductions differ by~1.4e-14mg/L, so real save/load uses an absolute1e-12 tolerance
(tiny unit and saved neural replays remain bitwise). No fitted model changed.

V2 completed ten seed42 pair forests, then stopped before its first neural
epoch because unobserved source OOF cells supplied NaN attention references.
Its sources/log and all forests are retained. Recovery version
`doc_current_source_portable_v2_r1` zero-fills non-loss reference cells while
preserving observed OOF values; this matches the evaluated architecture.
Six repair/facade contracts pass. Reuse the ten verified forests unchanged;
only40 new pair forests remain for seeds43-46. No research setting changed.
The fixed source-only R1 refit is complete: five neural fits, forty new pair
forests and ten reused references, seeds 42--46. It retains the source deployment
bases and 303 train / 54 validation roles. Ensemble validation MAE is 1.791012
versus 1.828532 for the prior release, a 2.052% gain [0.481%, 3.693%]; validation
also selected the fitted states. All five select zero static-memory fusion.
Export save/load error is at most 2.14e-14 mg/L; training-to-receiver facade error
is at most 7.45e-8 mg/L. All completion identities and source snapshots verify.

`doc_current_source_external_v2` is complete: 130 disjoint stations, 6,514 DOC
cells, 5,864 fixed-query cells, five source members and 5,000 paired station
bootstrap draws. All preceding rows are preserved exactly. Current-source K0
MAE is 0.910577 versus preceding 0.914994: 0.483% [-0.393%, 1.560%], unresolved.
Against earlier full model the cumulative gain is 7.468%; strong trees retain
lower cell MAE 0.840835. Station-equal error is 0.821436 versus trees 0.904738,
a different estimand. Fixed-query K0/1/3/5 is 0.925335/0.925335/0.828634/0.746269.
Q90 contains only 16 cells and all procedures miss them. Coverage 83.267% and
width 2.737396 trade narrower intervals for lower coverage than the preceding
release. This case was inspected earlier; this is an updated replication.

`doc_current_source_temporal_v3_r1` is complete: six task/seed packages, six
neural fits, fifty new and ten reused complementary forests. The original V3
stopped before its first neural epoch on an API role-string mismatch; its
execution is preserved. Every R1 neural/fusion product replays bitwise. Complete
fusion MAE 0.784905/0.796930 retains the prior temporal performance; increments
are 0.107% and 0.086%, the latter unresolved. Against strong trees reductions are
12.506% and 14.882%. Native attention alone has no established tree advantage;
all six select log-affine validation fusion. Q90 groups of 12/7 remain unstable.
These are separate monitored-station temporal refits, not deployment weights.

The five-member portable facade, source support/interval policies and label-free
full-grid export are complete. Manuscript v3 incorporates the current method,
geographical, external, strata and temporal figures; compilation and visual
inspection are complete. No parameter scan follows this finished delivery.

Parallel new-data idea: `doc_precipitation_inputs_v1/data_status.md`. Official
NASA documentation is reviewed, but its public precipitation API failed DNS
through the current runtime and was inaccessible through the web tool. No new
weather data has been collected; metadata/units remain unverified. Existing daily
flow already retains120,541 grid months with0 masked-out months, so expanding
that footprint cannot provide new cached information. Keep this as an explicit
access limitation; do not invent weather data or block authorized offline
model development. Do not restart completed runners. The current-source-portable-v2-r1 root records completed deployment status. The source-only/current-availability
geographical, full-source-level geographical and monthly-source-hydro studies are complete.

## Objective and current research choice

Reconstruct DOC at new stations without local DOC, pH or conductivity history.
Retain environmental trees, ecological encoding, observation-aware GRU, daily
hydrology and the local residual framework. Main K0 includes every valid DOC
cell; K1/3/5 use a separate fixed query excluding all five support candidates.

The retained model fits a station-hidden environmental reference and its
station-blocked OOF native-concentration residual. Dynamic donor retrieval and
hydro pretraining did not add reliable source-validation value. The fixed
complete upgrade remains the primary procedure; native-only is an ablation.
Do not select a different procedure for each test region or K.

## Completed research

### Source development

- `doc_unmonitored_tasks_v1`: water-quality-free receiving-site inputs, five
  geographical role assignments and separate all-observation/fixed-query tasks.
- `doc_source_retrieval_v1/v2/v3`: donor attention, contrastive donor profiles,
  station-balanced donor errors and hydro pretraining. Relative source gains
  are approximately 0 or negative; preserve these experiments.
- `doc_unmonitored_trees_v1`: station-hidden fitting improves ordinary matched
  trees by4.97%, but remains2.01% worse than the preceding complete model.
- `doc_unmonitored_residual_v1`: nine source-role packages, splits142/143/144,
  seeds42/43/44. Integrated K0 MAE1.769053 vs preceding1.829579:3.31% gain
  [1.72%,5.03%]; vs strong station-hidden trees1.866362:5.21% [2.69%,8.02%].
  These scores are development validation, not geographical or external tests.
- `doc_station_balanced_residual_v2`: nine matched source-only refits complete.
  Equal-station training loss gives MAE1.771209 vs existing1.769053:
  -0.12% gain [-0.65%,0.39%],1/3 partition directions positive. Do not adopt it.
  Native-only also shows no established gain. V1 stopped before training on
  tree accumulation roundoff; its execution is retained, and V2 records the
  numerical replay repair. No geographical/external query selects this change.

### Five-region geographical confirmation

`doc_geographical_confirmation_v1`:25/25 packages, HUC4
1013/1019/0708/1030/1101, cyclic-next validation, seeds42–46. Saved-state and
OOF exclusion verification,5,000 paired station draws and inspected figures
are complete in `analysis_5seed_final/`, `verification_5seed_final/` and
`figures_5seed_final/`. Earlier three-seed analyses are retained.

Equal-region primary K0 MAE2.282266 vs preceding complete2.360283:3.31% gain
[0.72%,5.98%],3/5 regions improving. Strong station-hidden trees2.344885:
2.67% gain [-0.04%,5.15%],5/5 positive directions. The5% working goal over
both comparators is not achieved. Q90 benefit remains uncertain. This is
retrospective geographical withholding inside ST357, not external validation.
Read the final section of its English `research_decision.md` first.

### Source deployment and independent basin replication

`doc_portable_source_fit_v1`:five source fits42–46 and five portable exports
complete. Source fitting uses303 ST stations/20,288 DOC cells; fixed split142
validation uses54 stations/2,283 cells. Historical internal test stations are
legitimate source data for the independent deployment. Five-seed averaging,
support correction and empirical intervals are selected from source validation.
All deployment fits select memory weight gamma0; complete and native-only
therefore coincide. No source-fit runner remains active.

`doc_external_replication_v1`:selected HUC02040104 has130 disjoint stations,
520 calendar months,6,514 DOC cells and5,864 fixed adaptation queries. Its
label-free point products were saved before scoring. All25 source procedure
replays, source-policy recomputation, scored-product verification,5,000 paired
station draws and inspected figures are complete.

| External result | Preceding full | Strong station-hidden trees | Fixed upgrade |
|---|---:|---:|---:|
| K0 cell MAE, mg/L |0.984065|0.840835|0.914994|
| Station-equal MAE |0.907013|0.904738|0.841214|
| Fixed-query K5 MAE |0.790013|0.681116|0.746009|
| Nominal90% empirical coverage |0.809487|0.925084|0.849248|
| Median interval width, mg/L |3.030606|3.536959|2.889311|

Upgrade gain over preceding full:7.02% [4.04%,9.53%]. Gain over strong trees:
-8.82% [-19.43%,3.47%]. Preserve cell-weighted and station-equal summaries as
separate estimates. Own fixed-query K5 improves19.5% over K0. Source Q90=9.730
identifies only16 external cells at11 stations; all tools miss those events and
their intervals. This tail is unstable. Read the external `research_decision.md`.
Do not retune this version on external query results.

### Portable product and manuscript

Single-member `scripts/portable_doc_reconstructor_v1.py` exposes
fit/predict/predict_components/predict_with_support/save/load. The five-seed
`PortableDOCEnsemble` in `scripts/portable_doc_ensemble_v1.py` delivers the fixed
complete procedure with saved source preprocessing, context/experience and
source-selected support/interval policy. Bundle:
`doc_portable_source_fit_v1/ensemble/`. It accepts arbitrary new station IDs
and consecutive calendars, hides all new-site chemistry at K0 and preserves
causal history. Large fits remain local. Actual label-free external bundle
replay differs by at most7.2e-15, including interval products. Usage:
`doc_portable_source_fit_v1/portable_usage.md`. The standalone
`scripts/predict_portable_doc_v1.py` exports arbitrary new grids and explicit
support corrections. Its3-station/24-month output, row identities and sidecar
were verified against the saved external product within8.9e-16.

New manuscript: `docs/paper/latex/unmonitored_doc_draft_v2.tex`, compiled with
existing local TeX Live. It uses the requested importance → prior-method gap →
proposed method → comparisons → future research story. Code-generated vector
architecture, geographical, independent-basin, environmental-strata and temporal
figures are included. The earlier manuscript is preserved. The final12-page PDF
was compiled twice without unresolved references or overflow warnings. All12
rendered pages were inspected; numerical and figure sources are recorded in
`docs/paper/latex/unmonitored_doc_draft_v2_sources.json`.

## Completed temporal compatibility

Root: `experiments/phase4_transfer/doc_temporal_compatibility_v2`.
Entry: `run_ladder.py --experiment doc-temporal-compatibility-v2`.
All6/6 fresh role-specific fits are complete: strictly unobserved and
observation-assisted periods, each seeds42/43/44. The runner exited normally.
Saved-state replay of all eight procedures agrees within7.2e-14 mg/L. Exact
query identities and truth,5,000 paired station draws and inspected figures are
complete. The archived assisted-period test mask is unsorted; sorting expected
cell identities repaired verification/analysis without changing training,
predictions or endpoints. Its failed check and `analysis_repair.json` are retained.

| Temporal MAE, mg/L | Current full fusion | Strong station-hidden trees | Upgraded full fusion |
|---|---:|---:|---:|
| Unobserved periods |0.905180|0.897097|0.785746|
| Observation-assisted periods |0.853555|0.936269|0.797619|

Full-procedure gains over current fusion are13.19% [3.91%,20.69%] and6.55%
[-2.32%,13.45%]; gains over the raw strong trees are12.41% [8.60%,15.84%]
and14.81% [11.79%,17.55%]. Every seed is positive in both contrasts. Native
residual-only gains over strong trees are only0.24%/0.29%, with intervals
crossing zero. All six complete refits select log-affine validation fusion;
seed42 selects zero native residual scale. The large complete-procedure gain
includes concentration/bias calibration and is not an isolated neural-memory
contribution. Q90 samples are12/7 cells, unstable. Read the English temporal
`research_decision.md` for all comparators, station-equal error and bias.

This is a monitored-station temporal compatibility refit, with legitimate past
source/local history and partial-period context. It is distinct from K0 new-site
inference. Each task refits its original backbone and source trees; it never
imports deployment weights that have seen the old temporal test labels.
The spatial ecological-memory module is excluded because its donor-station
assumption does not fit overlapping temporal roles. The original temporal
fusion family remains available on validation. Eight fixed procedure outputs
include original hybrid, context trees, ordinary/strong daily trees, current
native/fusion and upgraded native/fusion. The comparison does not claim an
identical portable deployment model or equal total optimization budget.

The original V1 startup could not construct a spatial support schedule for some
sparse temporal validation stations and stopped before finishing a backbone.
V2 directly fits on all temporal validation cells, retaining the failed V1
snapshot/log. Its actual tiny forest/OOF/GRU integration test covers sparse val
stations. Completed backbone/tree/current stages are resumable. If repair is
needed, keep old execution sources and document the revised version.

## Prior authorized evaluation cycle complete

The authorized research cycle is complete: unmonitored-site tasks, source-only
mechanism development, five-region confirmation, independent-basin replication,
portable single-member/ensemble inference, empirical coverage and width,
temporal compatibility and the rewritten illustrated manuscript. Completed
experiments are not restarted. The5% performance objective against both fixed
comparators was not reached in geographical/external evaluation; this remains
an explicit research result, not a reason to alter retained tests.

The next scientific direction, documented as future work, is concentration-
and hydrology-conditioned residual/fusion transfer. A calibrated tree-only
reference should accompany that investigation. Further mechanism development
starts on source roles142/143/144 and retains the completed geographical/
external queries as results of their evaluated version.

## Verification and environment

Current full suite:1018passed,2skipped; three targeted release/temporal tests also
pass after inference export integration. Ruff:all checks passed. Historical
`audit_artifacts.py --verify`:exit0. Pytest now explicitly discovers `tests/`
so retained evaluation-source copies are not imported as current test modules;
Ruff likewise excludes both execution and evaluation snapshots. Training code
was not changed by this test-discovery repair.

Commands use `/tmp/river-graph-authoring-tools/bin/uv run --no-sync`, with
`UV_CACHE_DIR=/private/tmp/river-graph-uv-cache UV_OFFLINE=1`. Plotting additionally
sets `MPLCONFIGDIR=/private/tmp/river-graph-matplotlib` and the Agg backend.
Repository `.git` writes are denied in this execution environment. Related
work is saved but uncommitted; maintain
`doc_geographical_confirmation_v1/pending_commit_files.json`, excluding large
local fitting caches and unrelated historical untracked files. Do not work
around the filesystem restriction.

The existing DOC heartbeat treats this document as the authoritative execution
state. Its old prompt mentions geography first; that previous evaluation cycle
is finished. Follow the renewed source-development section at the top for the
user's subsequent continuation. A tool request to pause the `doc`
automation was rejected: the MCP tool requires approval and this environment's
approval policy is `never`. The schedule could not be changed through the
authorized tool. The renewed current-source evaluation and manuscript cycle is
now complete. If the unchanged schedule fires again, verify this completed state
and remain quiet; do not repeat source training, analysis or paper generation.
The remaining Git submission needs a permitted environment. No filesystem or UI
workaround was used. Git submission likewise remains pending under the existing
read-only Git policy; the explicit related-file list is ready for a permitted
environment.
