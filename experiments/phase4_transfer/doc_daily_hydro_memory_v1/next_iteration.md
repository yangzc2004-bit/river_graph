# Next iteration: align station adaptation with the updated recurrent state

The daily-history model adds a modest K0 benefit over the matched current-only
projection, but does not improve the retained daily model at K5. Its central
prediction uses the newly trained encoder/GRU while station adaptation still
uses the frozen v4 `gru_tuned_anchor` representation. Test this interface before
changing the recurrent mechanism again.

## Minimal representation refresh

Use the nine completed packages and each selected off/current-only/full-history
checkpoint. Keep its native predictions, context forest, ecological residual
profile, training parameters and support/query cells fixed. No neural or
forest refitting is needed.

Compare these station-support representations for each expert:

1. Constant support correction, as already reported.
2. The existing frozen v4 two-dimensional basis, copied exactly.
3. The selected current model's 64-dimensional recurrent state projected by
   the same frozen v4 64×2 readout, then normalized by the same 32 fixed calendar
   anchors and one station-level scalar RMS (floor 1e−4).

Do not fit a new PCA or projection in this comparison. The contrast changes
the recurrent state used by adaptation, while preserving projection dimension,
projection weights and normalization formula. Source-validation selection
of support alpha/ridge and ecological mixing remains unchanged. Report direct
and integrated predictions at K=0/1/3/5 for every expert and representation.
K0 must remain exactly invariant to the support-basis choice.

Save refreshed bases and normalization statistics, source-validation results,
matched old-basis products and independent prediction replay. Read the source
validation comparison before the target comparison. Evaluate with the same
equal-partition estimator and paired station bootstrap. All three expert
contrasts are reported; target performance does not choose a representation
or a support-budget-specific route.

Calendar-anchor normalization uses the station's label-free hydrologic record,
including anchors after individual query months. This follows the existing
retrospective support protocol; it is not a forecasting representation.

## Subsequent memory hypothesis

The label-free decay audit finds that full-history explicit decay averages
0.8682 on source-validation queries and has 11-step decay-only retention
0.2863. Setting only its DOC-age term to zero gives mean gamma 0.9996.
Source OOF, validation and target stations all have their local DOC series
hidden, and their clock is the same function of dataset calendar position.
This is a shared modeling choice, not a source-to-target input shift.

If basis refresh does not address the K5 plateau, compare the existing DOC
clock, a zero age term, and a causal hydrology-availability clock. Change only
the recurrent decay's age input, retain raw M1/head/support features, and use
matched budgets. This would test whole-state decay conditioning; it would
not by itself prove a separate hydrologic memory mechanism. The audit's
explicit decay products are not the full GRU sensitivity.
