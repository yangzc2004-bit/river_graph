# Graph upgrade continuation R3

R3 is the stable execution snapshot after the initial mechanism exploration.
It keeps prior M1/M2 files intact and separates new control arms by output
directory.

The new runner records a source snapshot once per batch, writes a training
trace and checkpoint for each completed run, and uses configuration schema V5.
M3 uses a full chronological causal trend state with 24-month long-history
metadata; short and seasonal paths are causal convolutions.

The current execution order is:

1. M2 static same-month control;
2. M2 no-message control;
3. M2 fixed lag control if the static/no-message comparison remains useful;
4. M3 matched pilot on the two holdout families.

The old exploratory pilot remains under the parent directories and is not
rewritten by R3.
