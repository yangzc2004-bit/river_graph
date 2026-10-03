# Routing and daily availability: descriptive appendix

Groups are fixed by numeric descriptor-validity flags 5, 6, 7: no valid descriptor,
one or two valid descriptors, and all three valid. Coverage fractions do not determine
the route. Any valid descriptor selects daily; zero valid descriptors select monthly.
The routing rule was specified after the preceding daily study, before these new fits.

These are descriptive strata, without additional bootstrap endpoints or model selection.
Cell errors are pooled within seed, seed means averaged within partition, and nonempty
partitions equally weighted. Repeated predictions do not multiply ecological sample sizes.

| Availability | Unique station-months | Unique stations | Partition-cell occurrences |
|---|---:|---:|---:|
| all_three_valid | 8141 | 142 | 10175 |
| none_valid | 2379 | 57 | 2690 |
| partially_valid | 0 | 0 | 0 |

## Native versus direct/integrated K0

Routing guarantees the native and direct K0 hybrid equals monthly on zero-valid cells.
The integrated K0 output may differ there because ecological gamma is selected separately
for the complete hybrid pipeline. Such a difference is shown rather than treated as a routing failure.

| Stage | Arm | Availability | MAE | Q90 MAE | Ordinary MAE | Mean absolute prediction difference from monthly |
|---|---|---|---:|---:|---:|---:|
| direct_k0 | daily | all_three_valid | 1.9587 | 7.0857 | 1.2502 | 0.466192 |
| direct_k0 | daily | none_valid | 1.2221 | 7.9892 | 0.9963 | 0.240022 |
| direct_k0 | daily | partially_valid | nan | nan | nan | nan |
| direct_k0 | hybrid | all_three_valid | 1.9587 | 7.0857 | 1.2502 | 0.466192 |
| direct_k0 | hybrid | none_valid | 1.1798 | 7.9754 | 0.9528 | 0.000000 |
| direct_k0 | hybrid | partially_valid | nan | nan | nan | nan |
| direct_k0 | monthly | all_three_valid | 1.9884 | 7.1617 | 1.2724 | 0.000000 |
| direct_k0 | monthly | none_valid | 1.1798 | 7.9754 | 0.9528 | 0.000000 |
| direct_k0 | monthly | partially_valid | nan | nan | nan | nan |
| integrated_k0 | daily | all_three_valid | 1.9578 | 7.0982 | 1.2473 | 0.444344 |
| integrated_k0 | daily | none_valid | 1.2056 | 7.9888 | 0.9793 | 0.240193 |
| integrated_k0 | daily | partially_valid | nan | nan | nan | nan |
| integrated_k0 | hybrid | all_three_valid | 1.9578 | 7.0982 | 1.2473 | 0.444344 |
| integrated_k0 | hybrid | none_valid | 1.1647 | 7.9847 | 0.9369 | 0.026411 |
| integrated_k0 | hybrid | partially_valid | nan | nan | nan | nan |
| integrated_k0 | monthly | all_three_valid | 1.9776 | 7.1774 | 1.2574 | 0.000000 |
| integrated_k0 | monthly | none_valid | 1.1558 | 7.9743 | 0.9281 | 0.000000 |
| integrated_k0 | monthly | partially_valid | nan | nan | nan | nan |
| native | daily | all_three_valid | 1.9587 | 7.0857 | 1.2502 | 0.466192 |
| native | daily | none_valid | 1.2221 | 7.9892 | 0.9963 | 0.240022 |
| native | daily | partially_valid | nan | nan | nan | nan |
| native | hybrid | all_three_valid | 1.9587 | 7.0857 | 1.2502 | 0.466192 |
| native | hybrid | none_valid | 1.1798 | 7.9754 | 0.9528 | 0.000000 |
| native | hybrid | partially_valid | nan | nan | nan | nan |
| native | monthly | all_three_valid | 1.9884 | 7.1617 | 1.2724 | 0.000000 |
| native | monthly | none_valid | 1.1798 | 7.9754 | 0.9528 | 0.000000 |
| native | monthly | partially_valid | nan | nan | nan | nan |

## Direct and integrated products, GRU support basis

| Model | K | Availability | MAE | Q90 MAE | Ordinary MAE | Bias | Nonempty partitions | Q90 unstable |
|---|---:|---|---:|---:|---:|---:|---:|---|
| daily_gru_tuned_anchor | 0 | all_three_valid | 1.9587 | 7.0857 | 1.2502 | -0.3919 | 3 | False |
| daily_gru_tuned_anchor | 0 | none_valid | 1.2221 | 7.9892 | 0.9963 | -0.2058 | 3 | False |
| daily_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| daily_gru_tuned_anchor | 5 | all_three_valid | 1.7474 | 6.6047 | 1.0690 | -0.4663 | 3 | False |
| daily_gru_tuned_anchor | 5 | none_valid | 0.9824 | 6.9114 | 0.7852 | -0.0751 | 3 | False |
| daily_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |
| daily_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9578 | 7.0982 | 1.2473 | -0.4059 | 3 | False |
| daily_integrated_gru_tuned_anchor | 0 | none_valid | 1.2056 | 7.9888 | 0.9793 | -0.2281 | 3 | False |
| daily_integrated_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| daily_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7168 | 6.5503 | 1.0403 | -0.4479 | 3 | False |
| daily_integrated_gru_tuned_anchor | 5 | none_valid | 0.9786 | 6.8434 | 0.7833 | -0.0411 | 3 | False |
| daily_integrated_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |
| hybrid_gru_tuned_anchor | 0 | all_three_valid | 1.9587 | 7.0857 | 1.2502 | -0.3919 | 3 | False |
| hybrid_gru_tuned_anchor | 0 | none_valid | 1.1798 | 7.9754 | 0.9528 | -0.1522 | 3 | False |
| hybrid_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| hybrid_gru_tuned_anchor | 5 | all_three_valid | 1.7457 | 6.6033 | 1.0673 | -0.4670 | 3 | False |
| hybrid_gru_tuned_anchor | 5 | none_valid | 0.9818 | 6.8330 | 0.7871 | -0.0495 | 3 | False |
| hybrid_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |
| hybrid_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9578 | 7.0982 | 1.2473 | -0.4059 | 3 | False |
| hybrid_integrated_gru_tuned_anchor | 0 | none_valid | 1.1647 | 7.9847 | 0.9369 | -0.1750 | 3 | False |
| hybrid_integrated_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| hybrid_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7242 | 6.5821 | 1.0450 | -0.4544 | 3 | False |
| hybrid_integrated_gru_tuned_anchor | 5 | none_valid | 0.9800 | 6.8085 | 0.7860 | -0.0390 | 3 | False |
| hybrid_integrated_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |
| monthly_gru_tuned_anchor | 0 | all_three_valid | 1.9884 | 7.1617 | 1.2724 | -0.3731 | 3 | False |
| monthly_gru_tuned_anchor | 0 | none_valid | 1.1798 | 7.9754 | 0.9528 | -0.1522 | 3 | False |
| monthly_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| monthly_gru_tuned_anchor | 5 | all_three_valid | 1.7482 | 6.6176 | 1.0672 | -0.4337 | 3 | False |
| monthly_gru_tuned_anchor | 5 | none_valid | 0.9828 | 6.8071 | 0.7890 | -0.0352 | 3 | False |
| monthly_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |
| monthly_integrated_gru_tuned_anchor | 0 | all_three_valid | 1.9776 | 7.1774 | 1.2574 | -0.4196 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 0 | none_valid | 1.1558 | 7.9743 | 0.9281 | -0.1929 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 0 | partially_valid | nan | nan | nan | +nan | 0 | True |
| monthly_integrated_gru_tuned_anchor | 5 | all_three_valid | 1.7464 | 6.6306 | 1.0634 | -0.4378 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 5 | none_valid | 0.9792 | 6.8117 | 0.7851 | -0.0422 | 3 | False |
| monthly_integrated_gru_tuned_anchor | 5 | partially_valid | nan | nan | nan | +nan | 0 | True |

All context and historical references, both support paths and empty strata remain in the CSVs.
