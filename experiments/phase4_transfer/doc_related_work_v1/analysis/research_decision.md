# Published-backbone DOC comparison

Pinned official SAITS and GRIN code with documented covariate-only/inductive adapters. This is an internal development comparison, not the papers' original datasets or an independent SOTA confirmation. All source-side selections preceded new target scoring. Report both primary endpoints.

| model_name           |     mae |   log1p_mae |    rmse |       r2 |      bias |   station_macro_mae |   source_q90_mae |   query_count |   tail_query_count |   coverage |   basin_macro_mae |   basin_macro_log1p_mae |   training_seed_mae_std |
|:---------------------|--------:|------------:|--------:|---------:|----------:|--------------------:|-----------------:|--------------:|-------------------:|-----------:|------------------:|------------------------:|------------------------:|
| fixed_fusion         | 2.18388 |    0.30851  | 4.69912 | 0.497232 | -0.737229 |             2.84043 |          7.45052 |          7262 |               1039 |          1 |           2.30219 |                0.304504 |               0.0235063 |
| saits_doc_covariates | 2.24177 |    0.328758 | 4.56015 | 0.526252 | -0.589824 |             2.82617 |          7.1159  |          7262 |               1039 |          1 |           2.35373 |                0.324928 |               0.0877362 |
| grin_doc_covariates  | 2.26328 |    0.327541 | 4.63204 | 0.511054 | -0.517043 |             2.89802 |          7.48211 |          7262 |               1039 |          1 |           2.40831 |                0.32783  |               0.0740937 |

Five basin clusters give exploratory, unadjusted intervals. Additional recent competitors and reserved independent river systems are still required before stronger claims.
