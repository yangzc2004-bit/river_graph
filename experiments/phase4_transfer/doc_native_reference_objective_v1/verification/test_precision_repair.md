# Forest reduction precision in the isolation test

The first full suite completed with958 passes, two skips and one failure in the
new native-tree isolation test. Training/inference features were bitwise equal;
parallel forest prediction sums differed by at most1.78e-15. The contract now
requires exact features and exact individual tree thresholds/leaf values, with
1e-12 tolerance only for the parallel sum. This distinguishes input/fitted-state
isolation from floating-point reduction order. The original test source remains
in the execution snapshot and the failed suite log is retained. No fitting,
predictions, model settings or scientific analysis were changed by this repair.
