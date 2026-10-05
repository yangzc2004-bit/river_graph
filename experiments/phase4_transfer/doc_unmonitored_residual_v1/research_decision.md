# Research decision: station-hidden DOC residual reconstruction

Nine source-validation packages (station partitions142/143/144, seeds42/43/44)
are complete. All native and integrated predictions replay bitwise from their
saved states, and station OOF records exclude the held stations from their fits.
Target station roles were not evaluated in this development study.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) |
|---|---:|---:|
| Current complete model | 1.829579 | 9.225863 |
| Original matched daily trees | 1.963925 | 9.618575 |
| Station-hidden daily trees | 1.866362 | 9.394458 |
| Station-hidden base + current encoder/GRU residual | 1.772035 | 9.053055 |
| Complete model with ecological-memory fusion | 1.769053 | 9.075420 |

The complete candidate reduces MAE by3.308% relative to the current complete
model (paired station95% interval1.718–5.031%), by5.214% relative to the stronger
station-hidden tree base (2.694–8.022%), and by9.923% relative to the original
matched trees (6.635–13.446%). All three development partitions improve versus
the two main comparators. High-DOC error improves1.631% versus the current
model (0.472–3.258%). Ecological memory adds only0.168% versus the neural
residual alone; its interval crosses zero. This small fusion gain is not evidence
for a new transfer mechanism.

The useful change is training the environmental base on the same absent-local-
history condition seen at new stations, and retraining the existing neural
residual against those new station-OOF errors. Earlier learned donor attention
and hydro pretraining remain recorded as development results with insufficient
incremental benefit. They are not folded into this candidate.

## Continue with geographical replication

Retain the complete integrated recipe as one fixed candidate. Refit it and the
current model from source roles for each of the five saved whole-HUC4 tasks.
Use the strong station-hidden trees as the primary tree comparator, with the
ordinary matched trees retained for diagnosis. Do not reuse a historical fitted
backbone that has already trained on a geographical test region.

The source-validation intervals are useful for selecting the candidate, not
independent estimates of geographical performance. The fixed HUC4 result and
the prepared external basin02040104 will answer the generalization question.

Analysis: `uv run python scripts/analyze_doc_unmonitored_residual_v1.py
--bootstrap-draws 5000`. Figures: `uv run python
scripts/plot_doc_unmonitored_residual_v1.py`. Large fitted/input caches remain
local; the configuration, runtime source copies, predictions and analyses
preserve the experiment.
