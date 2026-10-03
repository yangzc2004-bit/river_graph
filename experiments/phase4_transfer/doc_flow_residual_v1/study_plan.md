# Explicit discharge dynamics in the existing DOC residual model

## Question

Does explicitly supplying causal discharge anomalies and short-term changes
improve the existing native-DOC residual beyond its current GRU history and
environmental context? The previous native tail-weighted residual improved
zero-observation station prediction by 2.54%, but added little after five local
observations. The new experiment targets information about month-to-month
variation using the same model.

The source-validation hydro diagnosis motivates the inputs. The same station
partitions 142–144 and seeds 42–44 are retained for development. Previous results
have been seen; this is not a fresh external confirmation.

## Model and controls

Keep the frozen context ExtraTrees base and spatial encoder. Copy the original
GRU/observation-decay weights and train them with a zero-initialized scalar head.
The new head concatenates 10 discharge-derived channels with the existing
64-dimensional recurrent state. There is no new backbone or attention operator.

Two new arms have the same nominal head dimension and training ceiling:

1. `flow_values`: all 10 new channels.
2. `flow_freshness`: the same channels, with only the three numerical
   anomaly/change columns zeroed. The missingness, age and count channels remain.

The zeroed columns are inactive, so this does not equate effective capacity.
Both arms still receive the existing discharge-containing spatial/GRU inputs.
Their contrast isolates explicit numerical changes at the residual readout,
not the value of all discharge information.

Saved references: the previous native-tail residual, the unchanged context base,
and previous v4 fusion adapters. Both constant and frozen-v4-GRU support
adapters are retained, with the same K={0,1,3,5} and fixed queries.

## Ten causal, dimensionless flow channels

Only dataset discharge and its observation mask enter this block. Let Q be
signed flow. Previous-12 statistics exclude the current month and include only
genuinely observed flow values. For each station/month:

- Relative anomaly r=(current Q minus previous-12 mean Q)/previous-12 mean |Q|,
  compressed as r/(1+|r|), plus its valid flag. Requires current flow, at least
  three previous observations, and a positive denominator.
- Symmetric signed 1-month change (Q_t-Q_(t-1))/(|Q_t|+|Q_(t-1)|), plus a valid
  flag requiring both observed values and a positive denominator.
- The analogous 3-month change and valid flag.
- Number of observed flow months in the preceding 12 calendar months / 12.
- Age since the latest observed flow at or before t, capped at 12 and / 12.
- A flag indicating that such a current/past flow observation exists.
- Current-month flow visibility.

Undefined numerical features are zero with their valid flag zero. Unseen age
is zero with age-valid zero. Real zero flows count as observations; signed
negative values are retained. All features are invariant to a positive change
of flow units and require no fitted scaler. Future hydro values and all DOC
labels are irrelevant to these features. These bounded transforms differ from
some unbounded diagnostics; they are fixed before this experiment's training.

## Training

Use the same selected-context station-blocked OOF predictions for source
training and source-fold-hidden DOC input views. Full-source context predictions
serve validation/test. The neural representation is source-trained, not an
independently fitted OOF neural model.

Both new arms retain tail weight 2 at/above source-training Q90, native-unit
MAE, lookback 12, 30 maximum epochs, patience 5, batch 512, Adam rates 1e-4 for
GRU/decay and 1e-3 for the scalar head, and gradient clipping at 1. Source-
validation K0 overall MAE selects checkpoint/scale from {0,0.25,0.5,1}, including
the exact context fallback. Each arm uses the same frozen support representation
and source-validation alpha/ridge grid after correcting its base predictions.
No objective or feature combination is selected using outer-query outcomes.

## Evaluation

Report values versus freshness and each versus the previous native-tail model
at K0/K5; retain context and previous fusion references. Use paired MAE,
partition-equal seed means and 5,000 joint whole-station bootstrap draws. Read
Q90, non-tail error, signed bias and false high-DOC alarms alongside overall
error. Report all arms and station-level gains/losses. Support is retrospective,
so the complete K-shot model is not a prospective forecasting system.

Every package saves weights, objective trace, flow definition, full-grid
components, adapted queries and replayable sources. Existing experiment data
and model products remain unchanged.
