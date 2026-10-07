# River morphology and DOC: restoring the scientific focus

## Research decision

The focal question is **how the geometric form and internal organization of a
river network shape DOC concentration and variability**. Vegetation and climate
provide the comparison context. The previous source-placement work supplies
one possible explanation, rather than replacing the morphology question.

This follow-up compares measured river forms in matched environments, changes
channel routing while holding the supplied concentration signal constant, and
separates the predictive information of footprint, branching and channel paths.
No neural reconstruction model was retrained.

The strongest current lead is **branch organization**, supplemented by the
distribution of channel paths. The coarse elongated/broad labels alone do not
establish a concentration ranking. Keep the three actual form classes for
communication, while studying the measured structures inside each class.

## 1. Different forms in comparable environments

The source-role cohort has 297 eligible, location/coverage-screened stations
across 62 HUC4s: 102 elongated tributary-rich, 30 mainstem-dominated sparse,
and 165 broad tributary-rich networks. Matching uses 296 stations with complete
comparison covariates. No concentration or morphology-response result selects
a station pair.

After excluding identical/nested receiving networks and requiring the same
HUC4, the primary comparison contains **22 elongated/broad pairs in 12 HUC4s**.
Matching holds basin size, land cover, climate, location and observation era
approximately comparable. Maximum matched area ratio is 1.780. Mean standardized
imbalances are near zero for area, forest, climate and observation year;
agriculture is−0.117 and urban cover +0.170 SD, so balance is improved but not exact.

| Response: broad minus elongated | Equal-pair difference | Paired HUC4 interval | Pairs / HUC4 |
|---|---:|---|---:|
| Station median DOC | +0.406 mg/L | −0.371 to+1.159 | 22 /12 |
| log1p station median DOC | +0.0683 | −0.0186 to+0.1512 | 22 /12 |
| Harmonic seasonal amplitude | +0.0343 | −0.0729 to+0.1199 | 22 /12 |
| DOC coefficient of variation | +0.0264 | −0.0582 to+0.0916 | 22 /12 |
| Observed DOC >=10 mg/L frequency | +1.81 percentage points | −2.48 to+5.26 | 22 /12 |
| Detrended within-station flow response | +0.0157 log1p DOC | −0.0860 to+0.0868 | 13 /8 |

These direct comparisons do not establish that broad networks have higher
concentrations or lower variation. This is not evidence of equivalence. The
small mainstem-dominated comparisons contain only four pairs/two HUC4s and
three pairs/one HUC4. Their large descriptive concentration differences should
not determine the class narrative. A one-HUC4 bootstrap cannot estimate
geographic variation; its interval is explicitly unavailable rather than
stored as an apparently exact point interval.

## 2. The forms differ in their channel-path organization

Path descriptors use all upstream catchment areas and actual directed channel
routes. Wetland and forest positions are not used as weights. In the same 22
matched pairs, broad networks have:

- Lower area-normalized mean path length: **−0.277**, interval[−0.394,−0.184].
- Lower relative path-length variation (distance CV): **−0.0432**,
  interval[−0.0851,−0.0164].
- A less certain mainstem-sinuosity difference: +0.146,
  interval[−0.0665,+0.2958].

Thus the form classes capture differences in the arrangement of flow paths,
even though their observed DOC concentrations overlap. A footprint label is
not a complete description of the transport geometry.

Distances are existing shortest primary/secondary routes from reach midpoints
to the receiving-reach outlet. They omit hillslope travel and are not measured
water residence times. Area coverage is effectively100% for this cohort.

## 3. Identical input, different routing

Every actual river network receives the same Gaussian concentration-anomaly
pulse. Total flow and supplied concentration strength are held constant;
catchment flows are proportional to their areas. Maximum channel-path delay
is normalized to one for each network, isolating relative path organization
from absolute network size. No vegetation-dependent source signal is supplied.

Among the matched elongated/broad pairs, broad-network routing produces:

- **+0.0231** output/input peak transmission, interval[+0.0078,+0.0501].
- **−0.0233** pulse temporal spread, interval[−0.0345,−0.0144], in normalized
  delay units.

In this controlled scenario, more concentrated arrival pathways produce a
sharper outlet signal. All networks conserve the time-integrated anomaly:
output/input mass ratios equal one numerically. Conservative routing changes
timing and concentration peaks; it cannot create a persistent mean DOC gain
from identical constant inputs. Mean-concentration differences require a
further process, such as channel/storage transformation, flow-dependent
exchange or different supplied material.

The simulated responses are mechanism demonstrations, not measured DOC effects
or fitted physical travel times. Catchment inputs are represented at reach
midpoints; sparse networks are sensitive to this discretization. The mean
class curve and mean of individual-network peaks are different statistics;
the routing figure labels them separately. Class2's larger individual pulse
peaks do not establish greater field DOC-event peaks.

## 4. Which morphological information matters for DOC?

Fixed Ridge models predict station median DOC in five HUC4-blocked folds on
the identical 297-station cohort. The baseline already contains measured area,
environment, observed local-hydro summaries, coordinates, HUC2 and sampling
covariates. Preprocessing is fitted within each training fold. Every fold has
the same stations for every information block.

| Added information | MAE, mg/L | Reduction vs environment+area | HUC4 interval | Positive folds |
|---|---:|---:|---|---:|
| Environment+area | 1.8971 | reference | — | — |
| Footprint: basin aspect, channel-axis ratio | 1.8616 | 1.87% | −0.52 to+4.63% | 4/5 |
| Branches: drainage density, mainstem share | 1.7911 | **5.59%** | **+2.02 to+8.42%** | **5/5** |
| Paths: sinuosity, normalized mean distance, distance CV | 1.8113 | 4.52% | −0.67 to+8.95% | 4/5 |
| All morphology | 1.7218 | **9.24%** | **+4.02 to+13.83%** | **5/5** |

Removing one block from the full model gives a complementary comparison:

| Information added while retaining the other blocks | Additional MAE reduction | HUC4 interval | Positive folds |
|---|---:|---|---:|
| Footprint | −0.01% | −1.28 to+1.47% | 3/5 |
| Branch organization | **3.23%** | **+0.86 to+5.14%** | **5/5** |
| Path organization | 3.33% | −0.46 to+6.87% | 5/5 |

Branch organization retains positive joint information after footprint and
paths are supplied. The log1p comparison gives the same lead: 4.78% above the
environment baseline and 2.63% when other morphology blocks are retained,
with both geographic intervals above zero. Keep the paths' positive point
estimates alongside their wider intervals.

These are source-cohort station-median diagnostics. **9.24% is not a gain of
the released monthly DOC reconstruction model**, and predictive improvement
does not identify a physical causal effect. Sampling and hydro summaries use
the source observation calendar. The forms are fixed; no class, feature block
or Ridge setting was selected from these results. The study examines several
related outcomes and blocks without multiplicity adjustment.

## Research narrative and next experiment

The developing explanation is:

> River form arranges tributaries and channel paths. That arrangement changes
> the distribution of arrivals, mixing and exposure to in-channel processing.
> These properties can add DOC information beyond environment and basin size.

The next scientific work should follow this pathway:

1. **Tributary integration:** test whether branch density and mainstem dominance
   modify how complementary tributary DOC signals combine, using the existing
   site-screened upstream/receiver observations. Retain equal averaging as a
   comparison so arithmetic smoothing is not credited to topology alone.
2. **Along-channel processing:** compare source/receiver DOC changes against
   actual path organization, waterbody exposure, flow and temperature, while
   accounting for added drainage. This addresses how a timing effect might
   accompany concentration-level differences.
3. **Model use of river structure:** condition the current upstream branch on
   branch organization and routed-path dispersion. Compare it with the same
   local morphology inputs and no messages, plus a tree with the same inputs.
   Develop on source roles before a new fixed geographic confirmation.

Source landscapes remain explanatory context for these steps. The next study
is not another survey of which vegetation type has higher DOC. It asks how
the river's structure combines and transforms the DOC supplied to it.

## Validation and preservation

The verifier recomputes matching, all paired DOC intervals and all geographic
score intervals; checks unchanged pairs after response and shape perturbations;
and checks profile normalization, conservative pulse mass, non-nesting and
matched model populations. Scientific figures are saved in English and Chinese.
See `validation.md` for repository checks. Historical experiments, graphs,
model products and paper endpoints remain intact.
