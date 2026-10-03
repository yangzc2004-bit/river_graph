# DOC-age attenuation of the shared hydrological memory

This diagnostic reads selected checkpoint weights, river edges and visibility masks. All DOC values are replaced with zero before the observation-statistics function is called. No prediction products or target labels are read. Source stations use their recorded station-fold-hidden input views; validation and target station inputs use the fixed train/context visibility. The latter two groups are input grids, not outcome evidence.

## Exact recurrence

For valid month $s$, $x_s=e_s+P d_s$ and $h_s=\mathrm{GRU}([x_s,1],\gamma_s\odot h_{s-1})$, where $\gamma_s=\exp[-\mathrm{ReLU}(W[a_s^{DOC},v_s^{local},v_s^{up},0]+b)]$. Current-only adds $Pd_s$ at the final step; full-history adds it throughout the 12-month causal window. Daily information enters unattenuated at its own step, but any part stored in the shared state subsequently encounters DOC-age decay.

At never-visible stations, $a_s^{DOC}=\log(1+s)/\log(13)$ for one-based calendar index $s$. It advances from the beginning of the dataset rather than measuring elapsed time since a discharge observation. Downstream support is zeroed in the actual KGML input path. The decay layer has 64 latent channels; these are not attention heads.

## Measured explicit retention

Values below average the nine saved models equally. Quantiles are calculated over station-month-channel values within each package, then averaged across packages; package and individual-channel distributions remain in the CSV files.

| Arm | Input group | Scope | Mean gamma | p10 / median / p90 | Mean 3-step | Mean 6-step | Mean 11-step | Age-zero mean gamma |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| off | source_fold_hidden | calendar_grid | 0.8775 | 0.8152 / 0.8670 / 0.9610 | 0.6843 | 0.4879 | 0.3051 | 0.9999 |
| off | source_fold_hidden | observed_cells | 0.8708 | 0.8116 / 0.8604 / 0.9500 | 0.6697 | 0.4690 | 0.2873 | 0.9996 |
| off | validation | calendar_grid | 0.8774 | 0.8150 / 0.8669 / 0.9610 | 0.6841 | 0.4877 | 0.3049 | 0.9999 |
| off | validation | fixed_query_cells | 0.8726 | 0.8138 / 0.8626 / 0.9498 | 0.6739 | 0.4746 | 0.2930 | 0.9997 |
| off | target | calendar_grid | 0.8776 | 0.8153 / 0.8672 / 0.9610 | 0.6846 | 0.4884 | 0.3056 | 0.9999 |
| off | target | fixed_query_cells | 0.8712 | 0.8126 / 0.8609 / 0.9493 | 0.6709 | 0.4709 | 0.2898 | 0.9998 |
| current_only | source_fold_hidden | calendar_grid | 0.8777 | 0.8141 / 0.8671 / 0.9648 | 0.6849 | 0.4888 | 0.3060 | 0.9999 |
| current_only | source_fold_hidden | observed_cells | 0.8709 | 0.8104 / 0.8605 / 0.9539 | 0.6702 | 0.4697 | 0.2880 | 0.9996 |
| current_only | validation | calendar_grid | 0.8776 | 0.8139 / 0.8670 / 0.9648 | 0.6847 | 0.4886 | 0.3059 | 0.9999 |
| current_only | validation | fixed_query_cells | 0.8727 | 0.8125 / 0.8628 / 0.9541 | 0.6743 | 0.4752 | 0.2936 | 0.9996 |
| current_only | target | calendar_grid | 0.8779 | 0.8143 / 0.8674 / 0.9648 | 0.6852 | 0.4893 | 0.3065 | 0.9999 |
| current_only | target | fixed_query_cells | 0.8714 | 0.8114 / 0.8610 / 0.9530 | 0.6714 | 0.4716 | 0.2906 | 0.9998 |
| full_history | source_fold_hidden | calendar_grid | 0.8733 | 0.8074 / 0.8616 / 0.9669 | 0.6755 | 0.4778 | 0.2980 | 0.9999 |
| full_history | source_fold_hidden | observed_cells | 0.8664 | 0.8036 / 0.8549 / 0.9542 | 0.6609 | 0.4592 | 0.2812 | 0.9996 |
| full_history | validation | calendar_grid | 0.8732 | 0.8072 / 0.8615 / 0.9669 | 0.6753 | 0.4776 | 0.2979 | 0.9999 |
| full_history | validation | fixed_query_cells | 0.8682 | 0.8060 / 0.8572 / 0.9541 | 0.6649 | 0.4644 | 0.2863 | 0.9996 |
| full_history | target | calendar_grid | 0.8735 | 0.8075 / 0.8618 / 0.9669 | 0.6759 | 0.4783 | 0.2986 | 0.9999 |
| full_history | target | fixed_query_cells | 0.8670 | 0.8048 / 0.8554 / 0.9530 | 0.6622 | 0.4613 | 0.2842 | 0.9997 |

The one-step distribution includes all valid calendar months. Multi-step products exclude the beginning-of-record months without the requested history; fixed-query summaries use only their eligible query months. A lag-L quantity multiplies the L explicit decay factors after an input at t−L. It is not an effective prediction coefficient or the full GRU Jacobian.

## Learned channels

- off: positive DOC-age coefficients in 576/576 saved channels; range 0.01926 to 0.12917. Hydro-projection row norms range 0.00000–0.00000.
- current_only: positive DOC-age coefficients in 576/576 saved channels; range 0.01983 to 0.13130. Hydro-projection row norms range 0.00820–0.16319.
- full_history: positive DOC-age coefficients in 576/576 saved channels; range 0.01875 to 0.13247. Hydro-projection row norms range 0.00725–0.12096.

## Interpretation and focused next comparison

The learned operators retain a real DOC-age-dependent attenuation. This is a semantic coupling between hydrological persistence and the target-observation clock, not a source-to-target clock mismatch: source-fold loss stations, held validation stations and target stations all have their entire DOC series hidden in these recurrent inputs. These multiplicative factors alone cannot establish that prediction accuracy is limited by decay: GRU update/reset gates, recurrent matrices and the head can compensate. Setting age to zero here changes only a diagnostic input; no predictions, model selection or performance claims are made from it.

The most focused next integration check is to refresh the station support-adaptation basis from the updated hydrological hidden representation. Current K0 predictions use the newly trained state, whereas K>0 shape adaptation still uses the older frozen v4 GRU basis. Compare old and refreshed representations under matched two-dimensional source-only basis construction, normalization and source-validation alpha/ridge selection, retaining both the off and full-history experts. This addresses an explicit representation disconnect without fitting another recurrent model. A refreshed basis must not be picked using target outcomes.

If a subsequent storage experiment is needed, retain the original DOC/ecology recurrent path and isolate daily hydrology in a separate storage path, with matched current-only and historical controls using the same added parameters. A whole-state age-zero replacement changes all ecological and DOC memory and deactivates 64 age coefficients, so equal allocated parameter count would not mean equal effective capacity. A new uncapped hydrological-age clock also adds a summary beyond the 12-month window and must be identified as extra information. The present diagnostic supports testing the coupling, not assuming that bypassing it will improve predictions.
