# Execution recovery, 2026-10-05

The active execution session was interrupted during a change in workspace
permissions. At10:24UTC, saved wrapper/training PIDs92345/92347 no longer
existed, and the training log had stopped updating at10:16UTC during
`huc4_1019_seed46`, source-backbone epoch9. No model exception was reported.

Eighteen of25 full packages were complete. Resume the same saved seeds45/46
matrix through `run_ladder.py --experiment doc-geographical-confirmation-v1`.
Completed packages and finished partial stages remain in place and are verified
by the existing runner. The incomplete backbone stage restarts deterministically
with its recorded seed and settings; no completed model is overwritten.

Training source modules, archived execution snapshot and study plan were not
edited. New inference/analysis scripts are separate from the executing training
modules. To accommodate the current filesystem policy, uv's operational cache
uses `/private/tmp/river-graph-uv-cache`, with offline execution in the same
existing managed `.venv`. This changes neither packages nor model settings.

A background-launch attempt failed under the current environment's process
priority restrictions and produced no new training output; its PID was absent
before retry. The runner is now launched as a tool-managed foreground process,
with its wrapper PID recorded in `training.pid`. Continue monitoring the log,
progress and complete packages, and do not launch a concurrent duplicate.
