# Independent DOC replication: research decision

## Result and scientific interpretation

The fixed five-seed model has now been evaluated in the independently selected
HUC02040104 basin:130 stations,6,514 observed DOC station-months. These station
identities do not overlap ST357. External DOC was opened for scoring only after
point predictions and source-validation support/interval choices were saved.
This is an independent basin evaluation, following the retrospective ST357
geographical confirmation. It is one basin, not a population of replicated
external basins.

The complete upgrade reduces primary cell-weighted K0 MAE from0.984065 to
0.914994 mg/L versus the original complete model:7.02% improvement, with a
5,000-draw paired station-bootstrap95% interval of4.04–9.53%. No external
DOC, pH or conductivity values entered K0 prediction or calibration.

The strong matched station-hidden ExtraTrees model is better in point estimate:
0.840835 mg/L. The upgrade's relative gain against it is−8.82%, with interval
−19.43–3.47%. Ordinary matched daily trees score0.837384 mg/L. Thus the
upgrade transfers beyond ST357 and improves its preceding full model, but
has not established superiority over strong explicit-feature trees.

| Procedure | K0 MAE | Station-equal MAE | Mean bias | 90% coverage | Median interval width |
|---|---:|---:|---:|---:|---:|
| Original full model |0.984065|0.907013|−0.410810|0.809487|3.030606|
| Ordinary matched daily trees |0.837384|0.904946|−0.038288|0.934602|3.575854|
| Strong station-hidden trees |0.840835|0.904738|−0.034247|0.925084|3.536959|
| Native residual upgrade |0.914994|0.841214|−0.434428|0.849248|2.889311|
| Fixed complete upgrade |0.914994|0.841214|−0.434428|0.849248|2.889311|

Concentration errors, biases and widths are in mg/L. The point estimate of
station-equal MAE favors the upgrade over trees, whereas cell-weighted MAE
favors trees. These were separate planned summaries. They weight monitoring
effort differently; the station-equal result does not replace the primary
endpoint or demonstrate a cause for the difference. The next source-only
experiment will test equal-station residual training directly.

## What the model actually used

All five deployment fits independently selected ecological-memory mixing
gamma=0 on source validation. Consequently the fixed complete and native-only
predictions coincide in this deployment. This is the pre-specified model's
source-selected fallback, not a replacement chosen from external results.
There is no established additional memory contribution in this external fit.
The neural branch uses the retained ecology/self encoder and observation-aware
GRU, with daily hydrology and native-scale residual prediction. Empty graph
edges are retained. No external river graph or source-to-external river links
were available; directional context is zero. Calendar-aligned source DOC
context remains available. This result does not establish river-message gain.

## Few-observation adaptation

Every K curve uses the same5,864 queries, reserving five support candidates per
station even at K0. Support dates may follow query dates: these are retrospective
record corrections, not prospective forecasting.

| Procedure | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| Original full model |0.995417|0.931237|0.891065|0.790013|
| Strong station-hidden trees |0.836142|0.777944|0.717284|0.681116|
| Fixed complete upgrade |0.926606|0.926606|0.830499|0.746009|

The upgrade improves19.49% from its fixed-query K0 to K5. Its K1 curve is
unchanged because source-validation ensemble selection chose alpha=0 at K1;
no external result changes that choice. K3/K5 use alpha0.25/0.5. The upgrade
remains better than the original full model and worse in point estimate than
strong trees at K5. Do not present different K-specific winners as one model.

## High DOC and uncertainty

The source-training Q90 threshold identifies only16 external cells at11
stations. All five procedures have zero recall and zero interval coverage on
these cells. Q90 MAE is9.566088 mg/L for the upgrade,9.421085 for the original
model and9.392486 for strong trees. These are unstable small-sample tail
diagnostics. Even a nominally significant tail contrast in this small group
does not establish a broad ecological pattern. High DOC remains underpredicted.

Source-val-only empirical intervals give upgrade coverage84.92%
[80.02%,90.29%], with median width2.889311 mg/L. Strong trees give92.51%
[90.48%,94.29%], with median width3.536959 mg/L. Source calibration and
model selection share validation information; these are empirical intervals.
The fixed nominal90% target is not reliably reproduced by the upgrade under
external distribution shift. Wider tree intervals achieve greater coverage;
coverage and width must be reported together. Earlier uncertainty-ranking
findings remain unchanged; no monitoring-blind-spot claim is added.

## Verification and deliverables

- All25 procedure×seed portable replays match2,283 saved source-validation
  cells per replay; maximum discrepancy2.14e−14 mg/L.
- Source support selection and interval quantiles were independently recomputed
  from saved source-validation predictions. Source/external identity overlap
  is zero. Full-grid components, arithmetic ensembles, scoring truth, fixed
  queries, designated support corrections and intervals were verified.
- `analysis/` contains errors, paired station intervals, station effects,
  hydro/ecology/source-distance strata and individual-seed diagnostics.
- The PDF/SVG/PNG comparison figure was rendered and visually inspected.
  Its initial native GUI-backend abort was repaired by using the same Agg
  backend as the geographical plotter. Predictions and statistical results
  were unchanged.
- The source execution snapshots are contemporaneous training records.
  Any evaluation code copies retained now are post-execution review copies,
  not replacements for those records.

## Next research action

Keep this complete external result. Continue performance development only on
the142/143/144 source training/validation roles, retaining the same full model
and strong trees as comparators. Test station-balanced residual training using
the unchanged30-epoch budget and existing tail weight2, changing the training
weighting alone. This hypothesis was considered during source development and
is now motivated further by the differing planned cell/station summaries.
Do not retune this external model on its query outcomes.

Then finish the original time-scenario compatibility check, portable ensemble
product and manuscript integration. The paper should report a method that
improves the preceding full model across internal geography and an independent
basin, together with a strong tree competitor, instead of asserting that the
graph architecture alone explains the improvement.

## Subsequent source-only experiment and product completion

The proposed equal-station experiment has completed on142/143/144 source
roles, nine packages. It gives -0.12% gain [-0.65%,0.39%] against the existing
complete upgrade and is not adopted; the original external procedure remains
unchanged. See `doc_station_balanced_residual_v2/research_decision.md`.

The five-seed portable ensemble is saved in `doc_portable_source_fit_v1/ensemble`.
Its label-free external full-grid replay matches the numerical products within
7.2e-15, including source-calibrated intervals. Environmental stratification
has been plotted using saved source thresholds. The matched temporal refits
and manuscript integration continue independently of these external outcomes.
