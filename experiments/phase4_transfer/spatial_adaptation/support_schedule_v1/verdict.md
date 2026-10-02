# Support schedule pilot

Internal selection chose `fixed_five` with K=3 (MAE 1.9968) using source-pool 40 and the preselected mean-residual shrinkage factors. The complete outer table is in `outer_all.csv`; `outer_selected.csv` is the one selected row.

This pilot is a value-blind schedule diagnostic. It uses a common outer query obtained by excluding the union of each schedule's five candidate cells. If no schedule beats the existing fixed K=5 result (about MAE 2.02 on its original paired query), the schedule line is closed. No labels are used to construct schedules.
