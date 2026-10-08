# River form changes the timing of information, with scale-dependent buffering

Date: 2026-10-08

## Scientific result

The useful structural question is more specific than whether a broad river is
always smoother. **Different path lengths change which upstream fluctuations
arrive together; the shared downstream corridor changes how sharply their
combined signal arrives.** These two effects depend on the duration and timing
of the inputs. No new source land-cover contrast is introduced.

This is a controlled mechanism follow-up on 32 actual monitored receiving
networks, 89 cropped source paths and seven catchment-overlap systems. Preserve
their original whole-network classes and the fixed 12 high-coverage receivers
(four elongated, eight broad). Each route definition has 6,912 scenarios. All
previous observational results remain intact.

## 1. Short and slow fluctuations have different structural responses

Give every source the same unit-variance exponential autocorrelation process,
with a common component and independent components. Use an imposed uniform
speed to express time relative to each network's mean cropped path. A shared
unit-gain exponential kernel redistributes a declared part of common-corridor
translation into spreading without changing individual path means.

At source coherence 0.5, the receiver-equal SD ratios below compare outlet
fluctuation with the same instantaneous upstream mixture:

| Input correlation time / mean nominal travel | Unequal paths only | Paths plus 0.5 common-corridor allocation |
|---|---:|---:|
| 0.025 | 0.873 [0.838, 0.916] | 0.352 [0.337, 0.384] |
| 8 | 0.993 [0.991, 0.996] | 0.983 [0.981, 0.986] |

Intervals resample entire overlapping-catchment systems 5,000 times. These are
controlled responses on the fixed geometry sample, not measured DOC reduction
percentages. A long, slowly changing signal is almost unchanged by the imposed
route spreading; a short fluctuation is much more sensitive to it. The complete
time, coherence and allocation grid is retained rather than selecting those two
endpoints as favourable field effects.

At the frozen middle working point (time 0.5, coherence 0.5, allocation 0.5),
mean SD ratio is **0.813 [0.801, 0.834]** across all 32 networks, and **0.819
[0.793, 0.856]** in the high-coverage 12. The mean log changes from path
differences and additional common memory are -0.0707 and -0.1390. They sum to
the total log change exactly. Their positive interaction of +0.0171
[+0.0128, +0.0197] means that counting each independently would double-count
some smoothing; this interaction is not an increase over the original mixture.

## 2. Arrival separation matters when sources share fluctuations

When sources are independent stationary processes, changing their delays does
not change aggregate variance. Source timing affects covariance, not each
individual source's marginal variance. Shared-corridor spreading still reduces
the fluctuations of each source.

With positive common fluctuations, differences in arrival paths separate some
of that common signal and reduce the instantaneous-mixture variance. A common
translation alone changes neither variance nor the input/output SD ratio.
These are analytic results for the declared forcing family, independently
checked by frequency integration. They do not establish that real tributaries
have equal variance or fixed positive coherence.

## 3. Passive paths can also align previously offset signals

Use the existing two unequal Loch Vale paths, approximately 1.060 and 1.422 km,
with a common downstream corridor of 1.033 km. These are actual cropped NHD
lengths. The source phases in this illustration are constructed, not measured.

Keep the geometry, flow proxies, individual input amplitudes and period fixed:

| Constructed source timing | Actual-path outlet / instantaneous-mixture amplitude | Add the declared common dispersion |
|---|---:|---:|
| Synchronous source waves | 0.460 | 0.131 |
| Longer-path source advances by 3*pi/4 | 2.174 | 0.621 |

In the second construction, travel delay cancels the source phase offset. The
outlet waves coincide, while the original instantaneous mixture partly
cancels. The 2.174 ratio is **relative to that instantaneous mixture**, not an
increase in source concentration, carbon production or a measured Loch Vale
peak. Outlet amplitude remains one, equal to each unit-amplitude input. Common
dispersion reduces it to 0.285, which is 0.621 of the instantaneous mixture.
All path means and unit gains remain unchanged.

This example establishes why receiving amplification cannot, on its own,
identify additional local DOC production. Input timing and route timing can
also cause it. It does not identify the cause of the previously observed Loch
Vale receiving amplification.

## 4. What this adds to the whole-form question

Under the same imposed input family and the same relative kernel rule, the
high-coverage broad-minus-elongated SD ratio difference at the working point is
**-0.0225 [-0.0690, +0.0255]**. It does not establish a form ranking. The
shortest-route version gives **-0.0208 [-0.0663, +0.0265]**, retaining that reading.

The earlier observed broad/elongated contrast of 0.403 [0.194, 0.940] remains a
separate field lead. It is a geometric-mean receiving/mixture comparison under
actual, regionally different source histories. It must not be compared as the
same estimand with this arithmetic-mean controlled difference, or fitted by
choosing a memory strength. The same-region comparison gap remains unchanged.

Sixteen of the 32 observed receiving ratios exceed one. None of the 6,912
stationary positive-coherence controls does so. This is a limit of that declared
forcing family, not a rejection of passive river transport: the separate
phase-offset construction demonstrates passive receiving amplification.

The structural profile to carry forward is:

**source fluctuation duration and synchronization -> path-length and entry
arrangement -> arriving overlap -> common-corridor spreading -> receiving DOC.**

Whole-network form provides a context for these quantities. A fixed class label
alone does not supply their joint response.

## Next scientific work

1. Use actual upstream observations to measure whether the sources rise together
   or successively, retaining the river form and complete path arrangement.
   Compare different observed periods within the same network before invoking
   source-region composition as an explanation.
2. Relate this observed input coordination to receiving fluctuation on identical
   sampled dates, with common-corridor and waterbody context kept explicit.
   Monthly DOC cannot directly identify event-scale transit time; the existing
   finer Loch Vale records support a separate resolution check.
3. Continue seeking same-region, similar-area whole-form pairs with observed
   tributaries and outlets. The 105 existing candidates define where such
   observations would most directly test the broad-versus-elongated field lead.

No new speed, lag, residence time, lake reaction or model hyperparameter is fitted
to observed DOC in this version. A targeted literature search returned HTTP 429;
no new article or inferred literature result is used. The covariance derivation,
independent spectral checks and preceding mapped data are the evidence here.
