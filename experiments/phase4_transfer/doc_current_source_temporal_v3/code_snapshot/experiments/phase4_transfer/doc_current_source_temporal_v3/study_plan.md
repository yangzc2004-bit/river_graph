# Current-source attention: temporal compatibility

Check the fixed current-availability attention architecture on the two
existing temporal DOC tasks (strict temporal extrapolation and observation-
assisted missing periods), three seeds 42/43/44 each. This is retrospective
compatibility, with temporal queries already inspected for the older recipe.
No architecture or parameter search follows these results.

Reuse each completed doc_temporal_compatibility_v2 source-fitted backbone,
station-hidden environmental tree, station-OOF reference, daily hydrology and
candidate initialization. Do not import the portable source fits, whose source
roles would include these temporal queries. Refit ten double-held station-fold
references and one twenty-candidate attention residual per task/seed. Keep the
41-feature readout, two32-dimensional heads, 12-month window, native MAE with
training-derived Q90 weight2,30epoch cap and patience5. This is the same
attention mechanism confirmed geographically; no extra water-quality input
is introduced by its source library.

The source library uses train cells only. Test and val labels are absent from
all input construction. Legitimate local train/context history remains
available at temporal inference. Training episodes retain the preceding
station-hidden residual condition. Query fold A is absent from its library;
donor references omit both A and donor fold B. Perturb held station labels
and all val/test cells individually before first pair fitting. Temporal
validation stations overlap source stations, so ecological-memory fusion
remains disabled; keep the same validation-selected concentration fusion as
the previous temporal recipe. Report native-only and complete fusion separately.

Save point/fusion choices before assembling the old score-only query truth.
Compare current-source native/fusion with prior upgraded native/fusion,
current full fusion and strong station-hidden trees on exactly the saved
queries. Use5,000 paired station-bootstrap draws with seeds averaged inside
each task. Report MAE, log error, RMSE, bias and Q90; Q90 counts12/7 remain
small and unstable. Evaluate causal/hidden-label saved-state replay and source
availability. Preserve old fits, predictions and manuscript evidence. This
checks compatibility with two time tasks; it does not make the portable
unmonitored-station protocol into prospective forecasting.
