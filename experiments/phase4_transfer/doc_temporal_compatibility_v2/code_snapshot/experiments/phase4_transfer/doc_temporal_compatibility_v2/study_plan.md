# Temporal compatibility of the new-station DOC residual recipe

Check the station-hidden tree/native-residual changes in the two existing
time-reconstruction tasks:unobserved periods (`e2a_strict`) and observation-
assisted periods (`e2b_partial`). Use the frozen ST357 masks and seeds42–44.
These splits have been used during prior development; this is a retrospective
compatibility check, not a fresh confirmation or the external evaluation.

Refit each task from scratch using its train/context input visibility. No
deployment or geographical checkpoint is reused: the deployment fits contain
labels in these temporal test periods. Retain val labels for selection only,
test labels for final scoring only. Only explicit context labels are visible
in the observation-assisted task. Keep causal12-month windows and daily
hydrology, where a current month's full covariates define retrospective monthly
reconstruction, not month-start forecasting.

Compare the freshly fitted original20-epoch log-residual hybrid, its selected
39-feature context trees, ordinary47-feature daily trees, station-hidden
47-feature daily trees, the current120-cap native residual/fusion and the
station-hidden30-extra-epoch native residual/fusion. Candidate source rows hide
their whole station fold's DOC, including context labels. Both inference views
retain legitimate past/local observed DOC in the temporal task, unlike K0
unmonitored-station prediction. Keep every model's input roles explicit.

Recurrent architecture, features and budgets match geography. Source OOF
residuals use five station folds, with nested station-hidden training for the
new tree. Checkpoints select all val-cell MAE; each native fusion uses the same
existing context/temporal/geometric/log-affine candidate family on all val
cells. No test score selects models. There is no geographical ecological-memory
fusion: train and val station identities overlap, violating that module's
station-disjoint design. Thus this check addresses the transferred native
recipe and matched temporal fusion, not the already-fitted portable new-site
ensemble. Keep the original hybrid as a separate control.

Report test-cell MAE, RMSE, bias, Q90 tail error, three-seed directions and5,000
paired station-bootstrap intervals separately for each scenario. Score exactly
the existing test masks. Test labels must not change input preparation or
training labels. All saved source execution and prediction products remain in
this independent version directory. If compatibility is poor, retain the
result and develop any remedy on source validation rather than the test query.

## V2 repair: temporal validation is not a K-shot episode

The initial v1 attempt stopped during backbone fitting: the general spatial
wrapper tried to reserve five K-support dates at every validation station,
which rejects temporal validation stations with fewer than six observations.
No completed fitted stage or query prediction was created. Its source snapshot,
configuration and failure log remain in v1. V2 fits the same forest, residual
and fusion components directly on all temporal validation cells, without a
spatial support builder. It preserves every sparse validation station and
keeps the original temporal train/val/test/context cells. The code never
reserves K supports for this compatibility experiment.
