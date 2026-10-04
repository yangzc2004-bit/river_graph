# Execution readiness

Date: 2026-10-04.

The source-validation development experiment is complete. Keeping the existing
temporal station calibration and ecological mix fixed, the separate chemical
increment lowered conditional station-validation MAE by 0.635% at K3 and
1.234% at K5 relative to legacy chemistry calibration. The K5 direction was
positive in all three partitions and all nine fitted packages. The current
joint procedure was still stronger on those validation curves, so this new
confirmation compares both procedures after full refitting on fresh station
roles rather than treating the development gain as an overall model win.

The complete one-package smoke run (partition 342, training seed 42) finished
in 61 seconds with reduced training budgets. It produced 233,478 full-grid
rows and 86,712 fixed-query rows across six models and four support counts.
Independent saved-state replay passed for the complete parent pipeline and
the separate chemical corrections. Smoke results are execution checks, not
the scientific confirmation.

Pre-launch checks passed: 889 pytest tests passed, 2 skipped; Ruff passed;
the historical artifact audit exited successfully with its existing explicitly
reported historical exceptions. A synthetic test was corrected to compare
floating-point replication within numerical tolerance; the estimator was not
changed. A replay-only string identity comparison was repaired before the
successful smoke verification.

Production uses the unchanged study plan: partitions 342/343/344, training
seeds 42/43/44, complete fresh fitting, six fixed comparison curves, and K
0/1/3/5. The primary comparison is the separate chemical increment versus the
current joint procedure at K5, with K3 and all other comparisons retained.
This is ST357 station-role replication. Large fitted caches stay local.

After the nine runs finish, replay all saved states, run the 5,000-draw paired
station-bootstrap analysis, inspect the final figure, and write the research
decision. The execution monitor handles completion without another approval
request. The training log and process record establish the actual launch time;
this readiness note does not itself assert that production has started.
