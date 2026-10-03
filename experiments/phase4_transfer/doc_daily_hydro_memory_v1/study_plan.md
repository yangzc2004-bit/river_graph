# Daily hydrologic state in the existing DOC recurrent model

The preceding daily-discharge head improved source-validation reconstruction
and support-assisted spatial transfer. Its daily history is absent from GRU
state updates. Test whether explicitly integrating that history improves the
zero-observation held-station prediction.

## Neural comparison

All three arms retain the same ecology/self encoder, observation-aware GRU,
source-OOF context base, 38 current-month head features and seven interaction
indices. Initialize from the same original source expert and a zero residual
head. Use no river messages in this local expert comparison.

- `off`: daily inputs only enter the head, matching the preceding daily model.
- `current_only`: a zero-initialized, bias-free Linear(8,64) adds current daily
  hydrologic information to the target month's encoded GRU input.
- `full_history`: the same projection operates at all valid months in the
  causal 12-month window.

Current-only and full-history add 512 trainable parameters and retain the GRU
dimensions, decay inputs and parameter learning rates. This comparison separates
better current-information integration from additional historical information.
Zero projection must reproduce the old hidden/prediction path exactly.

Run three station partitions (142–144) and three seeds (42–44): 27 neural fits.
Use the established 120 epoch ceiling, patience 5, native MAE loss with source
Q90 weight 2, unweighted source-validation K0 checkpoint/scale selection,
and learning rates 1e-5 for the final spatial self/ecology layers, 1e-4 for
GRU/decay/hydrology projection and 1e-3 for the head.

## Matched tree information probe

Clone each saved selected context ExtraTrees hyperparameter configuration.
Append twelve daily slots, each containing eight descriptors plus history-valid,
to its unchanged 39 context features. Current-only zeros the older eleven
daily slots; full-history retains them. Both have 147 features and identical
history-valid slots. Fit source labels only, without new hyperparameter search.
These eighteen independent benchmark fits do not replace the fixed neural
context or its OOF residual targets.

## Evaluation

Retain identical support representations, query cells, K=0/1/3/5 and source-
validation support fits. Neural arms also receive the existing ecological
integration; tree probes receive the same direct support adapters. Copy the
preceding daily direct/integrated and ecological-affine references unchanged.

Primary mechanism contrasts are full-history versus current-only and off,
current-only versus off, and tree history versus tree current. Compare neural
arms with the corresponding tree information probe and their frozen references.
Report native/log-space MAE, RMSE, R², Q90 and ordinary error, recall/false-high
rates, partition/seed consistency and fixed daily-availability/history-support
strata. Use the existing equal-partition/seed estimator and 5000 paired whole-
station bootstrap draws. Inspect source-validation fits before target analysis.
Do not select a model or support budget from the target contrast table.

Age groups are never-visible, 0, 1–3, 4–12 and >12 months since a visible target
reading. Never-visible is distinct from old; held-station queries may all fall
there. Hydrologic-history support groups are 0, 1–5, 6–11 and 12 usable months.
Empty groups are retained. Observation summaries use visibility and hydro flags,
not hidden DOC values. Positive K remains retrospective station adaptation.

This is a development iteration on previously examined station partitions.
Use existing frozen daily extraction/calendar/QC rules and source input views.
The model reconstructs month-end concentrations, not month-start forecasts.
Preserve old results, save all checkpoints/traces/components/adapters and verify
causal influence, old-path compatibility and inference replay.
