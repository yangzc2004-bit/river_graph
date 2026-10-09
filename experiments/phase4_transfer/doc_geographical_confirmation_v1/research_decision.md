# Final five-seed geographical research decision

All 25 packages completed: five fixed held-out HUC4 regions and seeds42–46.
Saved-state replay, stage identities, OOF exclusions and query alignment passed
before the 5,000-draw paired station bootstrap. Figures were inspected.
This is retrospective ST357 geographical replication, not external validation.

## Final K0 results

The primary population is7,262 DOC cells at104 stations. Average seeds within
region and then weight the five regions equally. Additional seeds repeat model
fits; they do not increase the number of ecological samples.

| Complete procedure | MAE (mg/L) | Station-equal MAE | Q90 MAE |
| --- | ---: | ---: | ---: |
| Current full model | 2.360283 | 2.556920 | 10.632267 |
| Ordinary daily-input trees | 2.440719 | 2.606200 | 10.473665 |
| Station-hidden strong trees | 2.344885 | 2.547670 | 10.520142 |
| Fixed upgraded full model | 2.282266 | 2.504735 | 10.519953 |
| Upgraded native residual without memory fusion | 2.265595 | 2.485495 | 10.494278 |

The fixed upgrade reduces MAE by **3.31% [0.72%,5.98%]** relative to the current
full model;3/5 regions improve. Relative to the strongest matched station-hidden
trees, the gain is **2.67% [-0.04%,5.15%]**, with5/5 directions positive. Thus the
paired interval supports a gain over the original full model, while the gain
over strong trees remains uncertain. The working5% target over both comparators
is not attained. The separate6.49% [3.11%,9.95%] gain over ordinary trees cannot
replace the stronger comparison.

Native-only has0.74% lower primary MAE than the fixed integrated candidate;
the paired interval for integrated worsening is0.23%–1.27%. Keep this ablation,
without replacing the fixed candidate after geographical scores are seen.
The comparison still tests a complete training recipe, including up to30 extra
residual epochs, rather than an isolated equal-budget architecture effect.

## Adaptation and high-DOC findings

Fixed-query K0/1/3/5 MAE is2.296495/2.160459/1.987946/1.909790 on6,742 identical
query cells. Relative to the current full model, K1/3/5 gains are3.56%/2.74%/2.05%,
with positive paired intervals. Against strong trees, their point gains are
0.97%/0.09%/-0.23%, and each interval crosses zero. Support is retrospective
station calibration using only designated support values and source-selected
shrinkage; it is not fed into the backbone.

Primary Q90 gain is1.06% [-0.70%,2.29%] against the current full model and
0.002% [-0.67%,0.55%] against strong trees. Tail detection and low predictions
remain unresolved. Q90 groups in HUC1030 and1101 contain15 and11 unique cells,
respectively, and are flagged unstable; seeds do not enlarge those populations.
Ecology/hydrology/source-distance strata retain contributing-region counts.

## Next research and deployment

The result supports aligning source training inputs with genuinely unmonitored
stations and refitting the existing ecology/GRU residual. Extra source-memory
fusion is not a reliable geographical improvement in this version. Future
mechanism changes return to142/143/144 source train/validation roles; retain the
complete geographical result rather than retuning on it.

Proceed with the fixed recipe's portable deployment and the already selected
02040104 external case, using no external DOC at K0. Define source fitting,
ensemble averaging and support calibration before opening external outcomes.
All comparator recipes use the same source roles and external query cells.
Source validation supplies empirical interval calibration; coverage must be
reported with interval width. Continue the existing time checks and manuscript.

## Reproduction

Run the analyzer with `--seeds 42 43 44 45 46 --bootstrap-draws 5000`, then the
plotter. Final outputs are preserved in `analysis_5seed_final/`,
`figures_5seed_final/` and `verification_5seed_final/`; reproduction sources are
copied into the analysis archive. The earlier three-seed history follows.
Repository Git writes remain unavailable in this execution environment; saved
research files are retained pending a permitted commit.

---

# Archived three-seed checkpoint (superseded by the final result above)

# Research decision: unmonitored DOC geographical replication

## Three-seed checkpoint

The initial 15 packages are complete: five held-out HUC4 regions, each with
training seeds 42/43/44. This experiment refits the full source model separately
for every geographical role assignment. It is retrospective geographical
replication within ST357, not validation in an independent external basin.

The primary candidate remains `unmonitored_integrated`, selected using source
validation in partitions 142/143/144 before this experiment. All test-region
water-quality history is hidden at K0. Source DOC context, environmental inputs,
hydrology, season and daily-flow features remain available to all matched arms.

## Primary K0 result

Each region first averages its three seeds; the five regions receive equal
weight. Primary K0 includes all 7,262 valid DOC station-months at 104 test
stations. Station clustering carries a station's months and seeds together in
5,000 paired bootstrap draws.

| Complete procedure | MAE (mg/L) | Station-equal MAE | Q90 MAE |
| --- | ---: | ---: | ---: |
| Current full model | 2.353589 | 2.551741 | 10.652596 |
| Ordinary matched daily-input trees | 2.434860 | 2.599510 | 10.478719 |
| Station-hidden strong trees | 2.343379 | 2.548005 | 10.506033 |
| Fixed upgraded full model | 2.293338 | 2.519433 | 10.510027 |
| Upgraded native residual without memory fusion | 2.273672 | 2.497767 | 10.489626 |

The fixed upgrade reduces MAE by **2.56%** relative to the current full model
(95% paired interval -0.19% to 5.36%; three of five regions improve), and by
**2.14%** relative to station-hidden strong trees (-0.30% to 4.29%; all five
regions improve). The positive direction is useful, but the intervals include
zero and the planned 5% improvement over both principal comparators has not
been demonstrated.

The 5.81% gain [2.63%,9.01%] over ordinary matched trees is a separate
comparison. It must not substitute for the stronger station-hidden comparator.

## Support and high-DOC findings

The fixed-query K curve contains 6,742 cells, reserving five candidate supports
at every K. Its population differs from primary K0. MAE for the upgrade changes
from 2.3080 (K0) to 2.1638 (K1), 1.9850 (K3) and 1.9071 (K5). Relative to the
current full model, the K1/3/5 gains are 3.53%,3.24%,2.31%, respectively, with
positive paired intervals. Against station-hidden trees these gains are small;
at K5 the point estimate is 0.12% worse, with an interval spanning zero.

High-DOC error remains large. Relative to the current full model the primary
Q90 MAE gain is 1.34% [-0.42%,2.60%]; four regions improve in this tail subset.
Relative to strong station-hidden trees it is -0.04% [-0.64%,0.44%]. Mean Q90
recall is approximately 0.20 and mean tail bias approximately -10 mg/L for the
compared procedures. This round does not solve high-DOC underprediction.

Ecology, hydro-availability and nearest-source-distance strata are exported
with their contributing region counts. Some groups contain fewer regions;
differences between their means must not be described as a monotonic response
without a matched-region comparison.

## Mechanism interpretation and next work

Matching source training inputs to stations without water-quality history and
refitting the existing neural residual gave useful source-validation gains.
Geographical transfer preserves a smaller positive direction. The comparison
tests a complete recipe: the upgrade changes the forest inputs/OOF targets and
adds up to 30 residual-training epochs after the current native model. It does
not isolate an equal-budget architecture effect.

Ecological-memory fusion is weaker than its native-only ablation here: the
fixed integrated candidate has 0.86% higher primary K0 MAE [0.03%,1.82%]. This
is evidence about this version's fusion under geographical shift. The primary
candidate is not changed after seeing these test scores, and no region-specific
or K-specific winner is assembled.

Continue the saved fixed recipe for seeds45/46 in all five regions, bringing
the confirmation to25 packages. Report this complete result before making a
new source-development decision. Further mechanism changes return to source
train/validation roles142/143/144. Portable inference development can proceed
with saved source preprocessing and donor profiles; external02040104 query
labels remain unused. The external result is a subsequent independent test,
not an opportunity to retune this geographical version.

## Reproduction and report repair

Run `scripts/analyze_doc_geographical_confirmation_v1.py --bootstrap-draws 5000`
and then `scripts/plot_doc_geographical_confirmation_v1.py` through `uv run`.
The analyzer verifies complete stages, label alignment, OOF station exclusion,
saved native-model replay (tolerance1e-6 for changed inference batches) and
bitwise integrated replay before evaluating predictions.

An initial reporting error reused overall improving-region counts for Q90
subsets. It was corrected without changing predictions, trained states,
estimands or bootstrap draws. The first report remains in `analysis_3seed/`;
the first corrected checkpoint is in `analysis_3seed_corrected/`, with matching
figures and verification. After adding the tail-direction regression test, a
fresh run of the final analyzer and plotter was archived in
`analysis_3seed_final/`, `figures_3seed_final/` and `verification_3seed_final/`.
It gives the same numerical results and includes reproduction-source copies.
Corrected figures were visually inspected. The five-seed report will be written
separately and will not erase these checkpoints.

Full software checks at this checkpoint:914 tests passed,2 skipped; Ruff passed;
the historical artifact audit returned exit0, retaining its documented legacy
conflict and missing-sidecar classifications. The portable inference prototype
replays all five procedures without passing new-station water-quality labels.

The current filesystem policy denies writes to repository `.git/index.lock`.
This checkpoint's files are saved but could not be staged or committed. The
related small artifacts are listed in `pending_commit_files.json`; large fits
and unrelated historical files are excluded. This does not change the fixed
training or subsequent research work.
