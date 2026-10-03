# Observation age and hydrological support: descriptive appendix

Current daily validity uses the three numeric descriptor flags; it does not use descriptor magnitude.
DOC age is elapsed months since a visible target observation, with -1 for never visible.
The fixed age groups are never-visible, 0, 1–3, 4–12, and >12 months. Never-visible is not
interpreted as calendar age. Under complete station holdout this axis may contain only
never-visible cells; empty age strata are then not identifiable from this design.

Daily history support counts months with any valid daily descriptor within the causal last
12 calendar months. Groups are 0, 1–5, 6–11, and 12. The maximum available calendar history
is min(month index+1,12), recorded separately so early padding is not mistaken for missing data.
All arms carry the same data-availability descriptors, regardless of which inputs they zero.

These are descriptive K0/K5 profiles, not additional model-selection endpoints or bootstrap tests.
The estimator remains seed mean within partition, then equal mean of nonempty partitions.

| Axis | Stratum | Unique station-months | Unique stations | Partition-cell occurrences |
|---|---|---:|---:|---:|
| daily_current_validity | none_valid | 2379 | 57 | 2690 |
| daily_current_validity | partially_valid | 0 | 0 | 0 |
| daily_current_validity | all_three_valid | 8141 | 142 | 10175 |
| visible_target_age | never_visible | 10520 | 172 | 12865 |
| visible_target_age | age0 | 0 | 0 | 0 |
| visible_target_age | age1_3 | 0 | 0 | 0 |
| visible_target_age | age4_12 | 0 | 0 | 0 |
| visible_target_age | age_over12 | 0 | 0 | 0 |
| daily_history_support | history0 | 2352 | 49 | 2657 |
| daily_history_support | history1_5 | 141 | 46 | 177 |
| daily_history_support | history6_11 | 281 | 57 | 358 |
| daily_history_support | history12 | 7746 | 142 | 9673 |

| Model (GRU support basis) | K | Axis | Stratum | MAE | Log MAE | Q90 MAE | Ordinary MAE | Nonempty partitions |
|---|---:|---|---|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 2.0866 | 0.3027 | 7.4960 | 1.3415 | 3 |
| context_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1959 | 0.2211 | 8.0987 | 0.9651 | 3 |
| context_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1857 | 0.2209 | 8.1251 | 0.9661 | 3 |
| context_gru_tuned_anchor | 0 | daily_history_support | history12 | 2.0554 | 0.3018 | 7.7382 | 1.3164 | 3 |
| context_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7621 | 0.3342 | 5.0503 | 1.9566 | 3 |
| context_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5654 | 0.3031 | 5.4895 | 1.7654 | 3 |
| context_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.9028 | 0.2859 | 7.5586 | 1.2574 | 3 |
| context_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7576 | 0.2430 | 6.6738 | 1.0705 | 3 |
| context_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9665 | 0.1753 | 6.6938 | 0.7759 | 3 |
| context_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9547 | 0.1747 | 6.7282 | 0.7736 | 3 |
| context_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7239 | 0.2411 | 6.8941 | 1.0424 | 3 |
| context_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.5422 | 0.3084 | 4.4610 | 1.8756 | 3 |
| context_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2681 | 0.2572 | 4.9171 | 1.5404 | 3 |
| context_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| context_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5960 | 0.2294 | 6.6945 | 1.0084 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9650 | 0.2871 | 7.0697 | 1.2592 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2132 | 0.2232 | 7.9796 | 0.9874 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.2043 | 0.2231 | 8.0125 | 0.9898 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9261 | 0.2861 | 7.3024 | 1.2243 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7301 | 0.3130 | 4.8787 | 1.9709 | 3 |
| current_only_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5974 | 0.2932 | 5.1088 | 1.9242 | 3 |
| current_only_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8104 | 0.2742 | 7.1512 | 1.1996 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7444 | 0.2441 | 6.5581 | 1.0714 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9821 | 0.1782 | 6.8855 | 0.7857 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9712 | 0.1776 | 6.9348 | 0.7846 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7082 | 0.2424 | 6.7809 | 1.0393 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4931 | 0.2998 | 4.1913 | 1.9054 | 3 |
| current_only_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.3278 | 0.2583 | 4.8997 | 1.6361 | 3 |
| current_only_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| current_only_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5872 | 0.2307 | 6.5968 | 1.0099 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9636 | 0.2868 | 7.0762 | 1.2567 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2046 | 0.2215 | 7.9772 | 0.9786 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1956 | 0.2214 | 8.0120 | 0.9809 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9251 | 0.2858 | 7.3098 | 1.2222 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7293 | 0.3131 | 4.8839 | 1.9664 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5866 | 0.2922 | 5.1045 | 1.9111 | 3 |
| current_only_integrated_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8079 | 0.2737 | 7.1573 | 1.1960 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7203 | 0.2374 | 6.5487 | 1.0450 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9773 | 0.1764 | 6.8304 | 0.7823 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9666 | 0.1759 | 6.8749 | 0.7814 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.6847 | 0.2355 | 6.7689 | 1.0139 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4715 | 0.2982 | 4.2353 | 1.8614 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2942 | 0.2545 | 4.9035 | 1.5918 | 3 |
| current_only_integrated_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| current_only_integrated_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5677 | 0.2251 | 6.5842 | 0.9894 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9570 | 0.2856 | 7.0590 | 1.2517 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1986 | 0.2197 | 8.0680 | 0.9693 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1900 | 0.2197 | 8.1124 | 0.9716 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9176 | 0.2842 | 7.2836 | 1.2173 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7264 | 0.3185 | 4.9301 | 1.9279 | 3 |
| full_history_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.6031 | 0.2967 | 5.1587 | 1.9158 | 3 |
| full_history_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8009 | 0.2722 | 7.1475 | 1.1891 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7443 | 0.2438 | 6.5707 | 1.0701 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9785 | 0.1775 | 6.8912 | 0.7817 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9677 | 0.1769 | 6.9484 | 0.7805 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7086 | 0.2420 | 6.7871 | 1.0394 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4923 | 0.3032 | 4.2922 | 1.8508 | 3 |
| full_history_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.3198 | 0.2586 | 4.9214 | 1.6173 | 3 |
| full_history_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| full_history_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5864 | 0.2303 | 6.6087 | 1.0079 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9562 | 0.2854 | 7.0747 | 1.2485 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1816 | 0.2164 | 8.0448 | 0.9524 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1728 | 0.2164 | 8.0896 | 0.9547 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9181 | 0.2841 | 7.3029 | 1.2153 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7246 | 0.3186 | 4.9391 | 1.9193 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5676 | 0.2936 | 5.1337 | 1.8764 | 3 |
| full_history_integrated_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.7974 | 0.2715 | 7.1612 | 1.1835 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7202 | 0.2376 | 6.5596 | 1.0435 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9769 | 0.1765 | 6.8402 | 0.7816 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9661 | 0.1759 | 6.8920 | 0.7804 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.6849 | 0.2357 | 6.7743 | 1.0136 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4778 | 0.3010 | 4.3095 | 1.8253 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2845 | 0.2535 | 4.9361 | 1.5695 | 3 |
| full_history_integrated_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| full_history_integrated_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5676 | 0.2253 | 6.5949 | 0.9880 | 3 |
| off_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9587 | 0.2877 | 7.0857 | 1.2502 | 3 |
| off_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2221 | 0.2234 | 7.9892 | 0.9963 | 3 |
| off_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.2139 | 0.2234 | 8.0181 | 0.9995 | 3 |
| off_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9196 | 0.2867 | 7.3215 | 1.2149 | 3 |
| off_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7422 | 0.3146 | 4.8955 | 1.9789 | 3 |
| off_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5860 | 0.2923 | 5.0860 | 1.9165 | 3 |
| off_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8070 | 0.2746 | 7.1673 | 1.1941 | 3 |
| off_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7474 | 0.2455 | 6.6047 | 1.0690 | 3 |
| off_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9824 | 0.1782 | 6.9114 | 0.7852 | 3 |
| off_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9716 | 0.1776 | 6.9552 | 0.7844 | 3 |
| off_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7108 | 0.2437 | 6.8313 | 1.0361 | 3 |
| off_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.5326 | 0.3055 | 4.2424 | 1.9423 | 3 |
| off_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.3263 | 0.2593 | 4.8898 | 1.6359 | 3 |
| off_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| off_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5894 | 0.2318 | 6.6423 | 1.0075 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9578 | 0.2876 | 7.0982 | 1.2473 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2056 | 0.2203 | 7.9888 | 0.9793 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1972 | 0.2203 | 8.0203 | 0.9822 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9197 | 0.2867 | 7.3358 | 1.2130 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7369 | 0.3140 | 4.9111 | 1.9604 | 3 |
| off_integrated_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5611 | 0.2901 | 5.0729 | 1.8874 | 3 |
| off_integrated_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8035 | 0.2741 | 7.1792 | 1.1887 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7168 | 0.2371 | 6.5503 | 1.0403 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9786 | 0.1766 | 6.8434 | 0.7833 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9678 | 0.1761 | 6.8791 | 0.7826 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.6813 | 0.2351 | 6.7730 | 1.0093 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4816 | 0.3002 | 4.2414 | 1.8695 | 3 |
| off_integrated_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2804 | 0.2536 | 4.8848 | 1.5781 | 3 |
| off_integrated_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| off_integrated_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5650 | 0.2249 | 6.5870 | 0.9856 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9587 | 0.2877 | 7.0857 | 1.2502 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2221 | 0.2234 | 7.9892 | 0.9963 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.2139 | 0.2234 | 8.0181 | 0.9995 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9196 | 0.2867 | 7.3215 | 1.2149 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7422 | 0.3146 | 4.8955 | 1.9789 | 3 |
| prior_daily_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5860 | 0.2923 | 5.0860 | 1.9165 | 3 |
| prior_daily_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8070 | 0.2746 | 7.1673 | 1.1941 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7474 | 0.2455 | 6.6047 | 1.0690 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9824 | 0.1782 | 6.9114 | 0.7852 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9716 | 0.1776 | 6.9552 | 0.7844 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7108 | 0.2437 | 6.8313 | 1.0361 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.5326 | 0.3055 | 4.2424 | 1.9423 | 3 |
| prior_daily_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.3263 | 0.2593 | 4.8898 | 1.6359 | 3 |
| prior_daily_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_daily_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5894 | 0.2318 | 6.6423 | 1.0075 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9578 | 0.2876 | 7.0982 | 1.2473 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.2056 | 0.2203 | 7.9888 | 0.9793 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1972 | 0.2203 | 8.0203 | 0.9822 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9197 | 0.2867 | 7.3358 | 1.2130 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7369 | 0.3140 | 4.9111 | 1.9604 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5611 | 0.2901 | 5.0729 | 1.8874 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8035 | 0.2741 | 7.1792 | 1.1887 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7168 | 0.2371 | 6.5503 | 1.0403 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9786 | 0.1766 | 6.8434 | 0.7833 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9678 | 0.1761 | 6.8791 | 0.7826 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.6813 | 0.2351 | 6.7730 | 1.0093 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4816 | 0.3002 | 4.2414 | 1.8695 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2804 | 0.2536 | 4.8848 | 1.5781 | 3 |
| prior_daily_integrated_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_daily_integrated_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5650 | 0.2249 | 6.5870 | 0.9856 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 1.9830 | 0.2840 | 7.3595 | 1.2393 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1298 | 0.2055 | 8.0776 | 0.8975 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1202 | 0.2052 | 8.1074 | 0.8990 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9511 | 0.2828 | 7.6124 | 1.2121 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.6897 | 0.3269 | 4.7963 | 1.9128 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.4684 | 0.2896 | 5.2964 | 1.6946 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8092 | 0.2683 | 7.4318 | 1.1653 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7366 | 0.2416 | 6.6522 | 1.0488 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9703 | 0.1751 | 6.8075 | 0.7759 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9590 | 0.1745 | 6.8455 | 0.7744 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7022 | 0.2397 | 6.8768 | 1.0195 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.5431 | 0.3110 | 4.3446 | 1.9001 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2580 | 0.2544 | 4.9373 | 1.5268 | 3 |
| prior_ecological_affine_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| prior_ecological_affine_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5799 | 0.2283 | 6.6808 | 0.9913 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 2.0343 | 0.2926 | 7.4571 | 1.2882 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1624 | 0.2144 | 8.3162 | 0.9231 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1525 | 0.2141 | 8.3660 | 0.9243 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_history_support | history12 | 2.0021 | 0.2916 | 7.6974 | 1.2623 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7031 | 0.3245 | 5.0206 | 1.8864 | 3 |
| tree_current_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5389 | 0.2960 | 5.4663 | 1.7409 | 3 |
| tree_current_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8558 | 0.2768 | 7.5353 | 1.2081 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7309 | 0.2369 | 6.6011 | 1.0510 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9594 | 0.1737 | 6.7484 | 0.7667 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9475 | 0.1730 | 6.7618 | 0.7651 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.6972 | 0.2350 | 6.8183 | 1.0228 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.5104 | 0.3029 | 4.4163 | 1.8423 | 3 |
| tree_current_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.2441 | 0.2534 | 4.8780 | 1.5214 | 3 |
| tree_current_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| tree_current_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5733 | 0.2243 | 6.6299 | 0.9909 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_current_validity | all_three_valid | 2.0214 | 0.2903 | 7.4474 | 1.2744 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_current_validity | none_valid | 1.1790 | 0.2171 | 8.3093 | 0.9405 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 0 | daily_history_support | history0 | 1.1682 | 0.2167 | 8.3417 | 0.9413 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_history_support | history12 | 1.9893 | 0.2891 | 7.6818 | 1.2491 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_history_support | history1_5 | 2.7353 | 0.3336 | 5.1103 | 1.8952 | 3 |
| tree_history_gru_tuned_anchor | 0 | daily_history_support | history6_11 | 2.5217 | 0.2963 | 5.4946 | 1.7150 | 3 |
| tree_history_gru_tuned_anchor | 0 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 0 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 0 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 0 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 0 | visible_target_age | never_visible | 1.8489 | 0.2756 | 7.5252 | 1.2015 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_current_validity | all_three_valid | 1.7323 | 0.2385 | 6.6048 | 1.0516 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_current_validity | none_valid | 0.9746 | 0.1767 | 6.8520 | 0.7789 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_current_validity | partially_valid | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 5 | daily_history_support | history0 | 0.9635 | 0.1761 | 6.8841 | 0.7778 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_history_support | history12 | 1.7018 | 0.2367 | 6.8358 | 1.0254 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_history_support | history1_5 | 2.4419 | 0.3028 | 4.2804 | 1.8025 | 3 |
| tree_history_gru_tuned_anchor | 5 | daily_history_support | history6_11 | 2.1948 | 0.2515 | 4.7710 | 1.4863 | 3 |
| tree_history_gru_tuned_anchor | 5 | visible_target_age | age0 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 5 | visible_target_age | age1_3 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 5 | visible_target_age | age4_12 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 5 | visible_target_age | age_over12 | nan | nan | nan | nan | 0 |
| tree_history_gru_tuned_anchor | 5 | visible_target_age | never_visible | 1.5773 | 0.2261 | 6.6389 | 0.9940 | 3 |
