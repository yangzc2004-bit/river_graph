# Regional learner comparison

The nested E3 selection compares ExtraTrees, RandomForest and HistGradientBoosting with source pool 40 and fixed K=5 residual adaptation (alpha=.75). Internal validation selected random_forest, with MAE 1.692.

Outer K=5 mean MAE (three seeds):

               learner  mae_mean   mae_sd
           extra_trees  2.024014 0.009999
hist_gradient_boosting  2.086366 0.000000
         random_forest  2.048123 0.013330

The best outer learner is extra_trees (MAE 2.024), which does not improve the existing five-seed ExtraTrees residual product (MAE about 2.023). The learner search is therefore closed; ExtraTrees remains the production regressor.
