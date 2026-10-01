# Station-profile mean/bias correction

Selected shrinkage alpha: **0**, chosen by the support-matched E3 validation split.

Five-seed E3 ET-context MAE: **2.5453**.
Five-seed E3 profile-corrected MAE: **2.5453** (0.00% lower).

The correction uses only source-station DOC means and target-station static/ecological/hydro/graph profile features. Target E3 DOC labels are used only for the final metric.

See `summary.csv` for comparison with density-ratio reweighting when that analysis is present.
