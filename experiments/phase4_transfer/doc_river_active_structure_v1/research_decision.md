# Water participation and DOC mixing on fixed river networks

Date: 2026-10-08

## Research conclusion

A fixed river arrangement can receive a different distribution of water at high
and low flow. Among the 17 eligible monitored frontiers, the season and time
adjusted effective contributor fraction increased by 4.15 percentage points at
high flow. This describes a more even distribution among measured incoming
branches, rather than an increase in the number of physical tributaries.

The result does not establish that broad or elongated networks have a distinct
DOC buffering response. The direct form contrasts remain unresolved, and the
positive participation average does not persist in the four networks with at
least 80% represented catchment area. Water fractions, branch DOC coordination
and branch fluctuation amplitudes together determine the mixing summary.

The useful advance is an explicit connection between mapped pathways and the
water and DOC signals using them. Whole-network form alone is insufficient to
predict the sign of the observed mixing response in this panel.

## Networks and measurements

The original 32 ST357 monitored frontiers, their source order, actual cropped
river paths and frozen form assignments were preserved. Seventeen frontiers
supported the hydro contrast: seven elongated and ten broad tributary-rich
networks, distributed across six overlapping river systems. The sparse form had
no eligible contrast and was retained in the availability ledger.

The hydro analysis contains 916 complete positive-flow network-months among the
17 eligible networks. The complete output contains 1,021 network-months across
eligible and ineligible frontiers. These counts retain repeated months and
nested receiving networks; they are not independent catchments. The secondary
DOC panel contains 531 common observed network-months at the same 17 networks.
Only the previously permitted source-role DOC cells were used.

Receiver flow terciles use all measured positive receiver discharge in each
original calendar span, including months without DOC. Participation requires
positive measured flow at every monitored source and the receiver. Missing or
nonpositive branch flows were not imputed as zero contributions. This panel
therefore measures balance among already flowing monitored branches and does
not measure activation of dry or unmonitored tributaries.

The effective fraction is `(1 / sum(w_j^2)) / J`, where `w_j` is the monthly
source discharge divided by total monitored-source discharge and `J` is the
fixed monitored-source count. It equals one for equal shares. Path dispersion
uses the fixed source-to-receiver distances and changing water fractions; its
reference mean distance stays fixed across flow states.

## Primary participation results

Network changes receive equal weight. The adjusted contrast uses annual
sine/cosine and a linear time term in each network. Intervals use 5,000 paired
whole-overlap-system bootstrap draws, preserving nested receivers, conditional
on the fitted network changes. They do not include a separate refit over sampled
years or discharge measurement uncertainty.

| Population | Networks | Raw high minus low effective fraction, pp | Adjusted high minus low, pp |
| --- | ---: | ---: | ---: |
| All eligible networks | 17 | 3.24 [−0.16, 6.40] | 4.15 [1.70, 6.40] |
| Elongated | 7 | 3.72 [−1.37, 9.83] | 5.35 [0.72, 9.31] |
| Broad and tributary-rich | 10 | 2.91 [0.24, 5.57] | 3.31 [2.06, 4.33] |
| Broad minus elongated | 17 | −0.82 [−6.68, 5.45] | −2.04 [−6.37, 3.43] |

Twelve of the 17 network point estimates were positive after adjustment. The
raw overall interval crosses zero, so calendar adjustment is consequential.
The direct broad-minus-elongated interval also crosses zero. Separate positive
within-form averages do not establish a difference between forms.

The adjusted normalized path SD change was 0.019 [−0.004, 0.050] overall. A more
even distribution of water can therefore occur without a resolved increase in
the spread of contributing path lengths. Actual travel times were not estimated
from these monthly records.

## Monitoring coverage changes the interpretation

These checks were added after inspecting the primary results and leave the
primary outputs and within-network fits unchanged.

| Minimum represented catchment area | Networks | Overlap systems | Adjusted effective fraction change, pp |
| --- | ---: | ---: | ---: |
| No additional restriction | 17 | 6 | 4.15 |
| 50% | 10 | 5 | 2.59 |
| 80% | 4 | 3 | −0.40 |
| 90% | 4 | 3 | −0.40 |

Removing each overlap system in turn leaves positive overall point estimates
between 2.93 and 4.57 pp. Thus one single system does not create the overall
sign. However, the four high-area-coverage networks have a slightly negative
mean. This small descriptive subset does not establish an opposite effect; it
shows that the full-panel result cannot be generalized to whole-network water
participation. Area thresholds retain different networks, rather than changing
the coverage of the same network.

Monitored-source discharge divided by receiver discharge was preserved without
clipping. Network low-flow means range from 0.17 to 2.26 and high-flow means from
0.059 to 0.98. The values above one and the large deficits preclude treating
these monthly frontiers as closed instantaneous water budgets. The locally
normalized monitored-source fractions remain the quantity analyzed.

## DOC mixing results

All source DOC series, the dynamically weighted mixture and the receiver use
the same seasonal and time projection. The fixed-state-mean-weight summary is
`100 * (1 - w' Sigma w / sum(w_j * Sigma_jj))`. It compares mixture variance
with the water-share-weighted source variances. It is a concentration
fluctuation summary, not DOC mass removal or a river reaction rate.

The high-minus-low change is decomposed into mean water fractions, the source
SD vector and the correlation matrix using all substitution orders. These
three contributions sum exactly; their substitution interpretation is analytic.
Dynamic mixture SD and observed receiver SD remain separate measured summaries.

| Population | Mixing potential change, pp | Water fraction contribution, pp | Coordination contribution, pp |
| --- | ---: | ---: | ---: |
| All networks | −0.59 [−6.83, 6.62] | 1.72 [−0.50, 4.25] | −2.97 [−9.71, 6.28] |
| Elongated | 3.66 [−10.30, 9.24] | 2.95 [−0.89, 5.60] | −0.29 [−12.15, 4.34] |
| Broad and tributary-rich | −3.57 [−8.29, 7.43] | 0.86 [−2.57, 5.13] | −4.85 [−12.91, 7.25] |
| Broad minus elongated | −7.22 [−13.02, 11.24] | −2.09 [−6.59, 5.96] | −4.56 [−14.31, 8.36] |

All these group intervals cross zero. The overall log change in receiver SD
relative to dynamic-mixture SD was −0.175 [−0.498, 0.394], also unresolved.
Individual networks display different combinations of water balance,
coordination and amplitudes; all 17 remain in the decomposition figure. The
field panel does not support ranking the two forms by DOC buffering, nor a
general claim that coordination always dominates the other contributions.

## Consequences for the river morphology study

The morphology question remains centered on the distribution of paths and
confluences. The controlled routing experiments isolate what those paths can
do when incoming signals and routing assumptions are held equal. The field
analyses show how the signals and water fractions using those paths change.
This distinction connects structure to DOC without substituting a source
land-cover classification for the original river-form question.

The next empirical priority is a comparison of independent elongated and broad
networks with near-complete tributary coverage and simultaneous branch flow and
DOC over rising and falling events. Match catchment size and regional setting,
retain source covariance, and measure arrival timing. That design can test
whether a shape contrast persists under comparable inputs. More reclassification
or repeated searches of the same sparse monthly archives will not answer it.

The current DOC predictor and prior results remain intact. No new prediction
model was trained and no spatial reconstruction improvement is attributed to
this mechanism analysis. The accompanying manuscript subsection can be used as
a mechanism result and discussion, without presenting a new field-confirmed
morphology coefficient.

## Deliverables and checks

Primary tables, exclusion reasons and source-flow weights are in `analysis/`.
Supplementary outputs are `coverage_sensitivity.csv` and `leave_system_out.csv`.
Three figure families have English and Chinese PNG and PDF exports in
`figures/`; rendered images were inspected and clipped English labels repaired.

The verifier reproduced every primary point table, checked fixed paths and
source ordering, and confirmed that perturbing DOC outside the permitted source
cells leaves the analysis DOC unchanged. Full tests: 1,358 passed, 2 skipped;
Ruff passed. The historical artifact audit exited zero, retaining the known G0
conflict, one zero-coverage case and 85 historical no-sidecar files explicitly.
Those historical qualifications are not evidence for new prediction identities.
