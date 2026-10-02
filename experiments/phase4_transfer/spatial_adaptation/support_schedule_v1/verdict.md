# Support schedule pilot

Internal selection chose `fixed_five` with K=3 (MAE 1.9968) using source-pool 40 and the preselected mean-residual shrinkage factors. The complete outer table is in `outer_all.csv`; `outer_selected.csv` is the one selected row.

This pilot is a value-blind schedule diagnostic. It uses a common outer query obtained by excluding the union of each schedule's five candidate cells (1,844 cells per seed), so its MAE values are not directly comparable to the original 2,316-cell fixed-query result (about 2.02). On this common query, fixed-five K=5 is the best diagnostic (MAE 1.860), while the internally selected fixed-five K=3 scores 1.984. The schedule variants do not produce a credible reason to expand this line; retain the original fixed schedule and close the schedule search. No labels are used to construct schedules.
