# Descriptive daily-hydrology availability appendix

Strata are fixed from the three descriptor-validity flags (local daily columns 5, 6, 7):
all three flags equal one versus not all three. Coverage fractions at columns 3 and 4
are not validity flags. The same availability values are carried by every arm and reference;
these columns describe the data, not the zeroed inputs of the matched control arms.

This is an appendix, without a new primary endpoint, bootstrap test or model selection.
Cell errors are pooled within seed, seeds averaged within partition, and nonempty
partitions equally weighted. These conditional populations differ from the full query
population. Distinct ecological counts are deduplicated across repeated seeds and partitions.
Q90 is the existing source-training threshold; fewer than 20 true high cells in any partition
is marked unstable. Empty strata are recorded as undefined, not assigned zero error.

| Availability | Unique station-months | Unique stations | Partition-cell occurrences |
|---|---:|---:|---:|
| all_three_valid | 8141 | 142 | 10175 |
| not_all_three_valid | 2379 | 57 | 2690 |

## Direct and integrated residuals (GRU support basis)

| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Signed bias | Nonempty partitions | Q90 unstable |
|---|---:|---|---:|---:|---:|---:|---:|---|
| availability_gru_tuned_anchor | 0 | all_three_valid | 1.9875 | 7.1625 | 1.2713 | -0.3782 | 3 | False |
| availability_gru_tuned_anchor | 0 | not_all_three_valid | 1.1790 | 7.9845 | 0.9518 | -0.1558 | 3 | False |
| availability_gru_tuned_anchor | 5 | all_three_valid | 1.7534 | 6.6478 | 1.0695 | -0.4437 | 3 | False |
| availability_gru_tuned_anchor | 5 | not_all_three_valid | 0.9813 | 6.8213 | 0.7870 | -0.0453 | 3 | False |
| availability_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9767 | 7.1780 | 1.2564 | -0.4244 | 3 | False |
| availability_integrated_gru_tuned_anchor | 0 | not_all_three_valid | 1.1548 | 7.9826 | 0.9268 | -0.1960 | 3 | False |
| availability_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7460 | 6.6292 | 1.0631 | -0.4382 | 3 | False |
| availability_integrated_gru_tuned_anchor | 5 | not_all_three_valid | 0.9794 | 6.8128 | 0.7852 | -0.0419 | 3 | False |
| daily_gru_tuned_anchor | 0 | all_three_valid | 1.9587 | 7.0857 | 1.2502 | -0.3919 | 3 | False |
| daily_gru_tuned_anchor | 0 | not_all_three_valid | 1.2221 | 7.9892 | 0.9963 | -0.2058 | 3 | False |
| daily_gru_tuned_anchor | 5 | all_three_valid | 1.7474 | 6.6047 | 1.0690 | -0.4663 | 3 | False |
| daily_gru_tuned_anchor | 5 | not_all_three_valid | 0.9824 | 6.9114 | 0.7852 | -0.0751 | 3 | False |
| daily_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9578 | 7.0982 | 1.2473 | -0.4059 | 3 | False |
| daily_integrated_gru_tuned_anchor | 0 | not_all_three_valid | 1.2056 | 7.9888 | 0.9793 | -0.2281 | 3 | False |
| daily_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7168 | 6.5503 | 1.0403 | -0.4479 | 3 | False |
| daily_integrated_gru_tuned_anchor | 5 | not_all_three_valid | 0.9786 | 6.8434 | 0.7833 | -0.0411 | 3 | False |
| monthly_gru_tuned_anchor | 0 | all_three_valid | 1.9884 | 7.1617 | 1.2724 | -0.3731 | 3 | False |
| monthly_gru_tuned_anchor | 0 | not_all_three_valid | 1.1798 | 7.9754 | 0.9528 | -0.1522 | 3 | False |
| monthly_gru_tuned_anchor | 5 | all_three_valid | 1.7482 | 6.6176 | 1.0672 | -0.4337 | 3 | False |
| monthly_gru_tuned_anchor | 5 | not_all_three_valid | 0.9828 | 6.8071 | 0.7890 | -0.0352 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9776 | 7.1774 | 1.2574 | -0.4196 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 0 | not_all_three_valid | 1.1558 | 7.9743 | 0.9281 | -0.1929 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7464 | 6.6306 | 1.0634 | -0.4378 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 5 | not_all_three_valid | 0.9792 | 6.8117 | 0.7851 | -0.0422 | 3 | False |

All context and historical references and both support paths are included in the CSVs.
