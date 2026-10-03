# Product roles

Each `runs/split*_seed*/predictions.parquet` contains the **new v2 target-query
predictions** for K = 0, 1, 3 and 5. Its `regional_gamma` column records the
source-validation mixing choice for that model and support budget.

`full_grid.parquet` is an unchanged copy of the **frozen v1 component cache**.
It stores context, interaction, ecological memory and fixed-v1 predictions.
Its `*_pred` columns are not the new K-dependent v2 final predictions. Use
`mixers.json` together with those components and the unchanged v4 support basis
to reconstruct the new selected bases and support-adapted predictions.

`profile_bindings.json` points to the frozen v1 ecological profiles. No forest,
recurrent encoder, support basis or ecological profile was refitted in v2.
Only validation-selected mixing and support-adapter parameters were fitted.

`verification/replay_checks.json` checks the component cache against v1 and
replays the selected bases and query predictions. The analysis keeps all fixed
v1 profiles as separately named references.
