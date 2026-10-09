# Research decision: joint source chemistry supervision

## Completed experiment

Nine source-development packages (partitions142/143/144 × seeds42/43/44)
are complete, including18 DOC fits with real and shuffled auxiliary labels.
The existing ecology encoder and observation-aware GRU were jointly supervised
by DOC and source pH/log-conductance, including source months without DOC.
The fixed auxiliary weight was0.1; original30-epoch/patience5 settings,
environmental reference, OOF residuals and backbone initialization were retained.
Receiving stations had no water-quality inputs or auxiliary targets.

| Procedure | K0 MAE (mg/L) | Q90 MAE (mg/L) | Signed bias (mg/L) |
|---|---:|---:|---:|
| Retained complete model |1.769053|9.075420|-0.638608|
| Joint real-source supervision, complete |1.773236|9.073502|-0.628029|
| Joint shuffled-source supervision, complete |1.769250|9.073485|-0.634496|
| Retained neural-only model |1.772035|9.053055|-0.625464|
| Joint real-source supervision, neural-only |1.776089|9.050676|-0.612959|
| Strong station-hidden trees |1.866362|9.394458|-0.611014|

## Performance and interpretation

Joint supervision worsens complete-model MAE by0.236% [0.027%,0.492%]
relative to the actual retained complete model. None of the three partition
averages improves; only two of nine packages improve. Its gain relative to
the matched shuffle is-0.225% [-0.512%,0.040%], with no positive partition
average. The matched neural-only gain is-0.229% [-0.541%,0.024%]. Intervals
are from5,000 paired whole-station draws, with seeds averaged within partition
and three partitions equally weighted. These are source-development comparisons.

Complete Q90 gain versus retained is0.021% [-0.092%,0.171%]; the small bias
change does not establish improved high-DOC reconstruction. The apparent4.99%
gain over strong trees is mostly the existing retained model's advantage and
must not be presented as a new4.99% contribution from auxiliary training.

Real-source auxiliary MSE falls in all nine packages for both targets. Shuffled
labels approach a standardized MSE of1, as expected for their mean predictor.
This confirms that the shared state learns source chemistry associations, but
that learning does not improve DOC transfer in this tested design. Source MSE
is an in-sample auxiliary diagnostic, not a new-station chemistry evaluation.
The experiment distinguishes simultaneous supervision from the earlier failed
sequential pretraining; neither is retained in the deployed DOC model.

## Decision and next research action

Keep the current complete model, source bundle and manuscript results. Close
this auxiliary-training branch without scanning weights or repeating
geographical/external confirmation for it. No new model is adopted.

Continue with one separate temporal question: the retained neural residual
resets its recurrent state every12 months. Test a24-month causal window on
the same source roles and same backbone, reference, loss and training budget.
This is a longer information horizon, unlike the completed extended-epoch
experiment. No chemistry auxiliary head, nonlinear head, new reference or
longer optimization schedule is carried forward. Its plan is saved before
fitting; a positive outcome would still require confirmation.

## Verification and reproducibility

All nine native and memory-integrated predictions replay bitwise. Initial
backbone states, source normalization and selected auxiliary-head MSE replay
bitwise; the source and receiving station populations are disjoint. The old
prediction frames are unchanged. The zero auxiliary weight matches the ordinary
DOC optimizer within1e-12. Hidden-label, future-input and save/load tests pass.
The full completed-version suite has967 passes and two skips; Ruff and the
historical provenance audit pass. Figures were generated from the saved CSVs
and the PNG actually inspected. Large fitting caches remain local. Related
files are listed for submission; Git index writes remain unavailable in this
execution environment. Previous ST357 geographical and independent-basin
evaluations are unchanged and were not used to tune this source experiment.
