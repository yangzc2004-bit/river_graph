# Conclusion: river form organizes the timing and spreading of DOC fluctuations

Date: 2026-10-08.

## Central conclusion

The useful result is a **timing-and-spreading explanation of river structure**:
different upstream paths determine which fluctuations arrive together, while
transport through the common downstream corridor can broaden their combined
response. Whole-network form summarizes this arrangement but does not, by
itself, prescribe one DOC peak or buffering strength.

This version completes the planned bounded record recovery, structural
separation and observational robustness checks. It produces a mechanism result
with replicated observational support for the overlap-to-excursion link. It does
not complete a measured broad-versus-elongated timing comparison: the targeted
archive still lacks the required paired observations.

## 1. Targeted recovery was performed, rather than proposed again

The first 15 saved non-nested, same-HUC4, similar-area form opportunities contain
26 distinct receiving networks (14 elongated, 12 broad). We inventoried every
already mapped ST357 upstream gauge, including gauges omitted in the previous
source-role audit because they had no common dates. The fixed collection list
contains 44 stations.

The official WQP discrete-result service supplied 43 complete station downloads;
one failed request retains its explicitly recorded older-cache fallback. Native
DOC fraction, unit and censoring filters yield **5,640 sampling activities**.
Result replicates within one activity do not create additional observations.
These are complete public activity histories in a new observational scope,
including months outside the predictive model's source roles. They have not
entered model fitting, feature selection or prediction-model evaluation.

All 39 non-nested candidate upstream pairs were checked against exact calendar
dates. One elongated receiver has enough common dates for the established
same-day analysis. No selected broad receiver does, and neither form has a
usable dense within-month receiver. **Zero of the 15 form opportunities has
adequate co-sampling on both sides.**

The broad-form gap is concrete: some catchments have fewer than two already
mapped upstream DOC gauges; others have records on different dates. A richer
concentration history alone does not create a two-branch/receiver timing record.
Same-day observations also retain their actual, sometimes different clocks.

### Additional stations and current continuous-series service

Station-catalogue discovery was attempted in all seven HUC4 regions of the fixed
shortlist. Six regional catalogue requests timed out. The returned 1019 catalogue
contained 639 organic-carbon stations, including 49 additional river/stream
locations inside two selected broad catchments. A first parser expected obsolete
coordinate column names; its v0 products/code are retained. Replaying the saved
source with the actual WQP3 standardized coordinates corrected that error before
the final interpretation.

All **49 additional station archives were retrieved**. Forty-eight supplied
accepted DOC, yielding **1,047 further sampling activities**. Provisional
coordinate matching to real mapped channels was evaluated at both 100 m and
300 m, retaining unique reach matches and excluding receiving-reach aliases.
The expanded record still has no eligible broad receiver: the largest two-source
and receiver intersection in those catchments is one date. This supplementary
geometric mapping is not represented as a verified NLDI hydrolocation.

Together, the primary recovery and this extension inspected **6,687 accepted
activities across 93 requested station identities** (92 fresh downloads and one
explicit older-cache fallback). The count describes activities, not independent
rivers or replicated events. Six catalogue access failures leave those regions
incompletely inventoried beyond the existing stations; they do not prove that
all possible coordinated archives are absent.

The legacy continuous-service request returned HTTP 404. We then followed the
current official USGS OGC catalogue and its actual queryable schema. Its complete
parameter-00681 time-series metadata response returned zero series; the 26 target
receivers therefore have no match in that retrieved catalogue. This is an
availability result for that USGS parameter catalogue, not a statement that
continuous optical carbon proxies or other providers' DOC records do not exist.

This version closes the bounded archive search. We do not relax the date rules
or replace a missing paired field case with a constructed input.

## 2. The structural effects are separated on actual mapped paths

The mechanism experiment uses the previous 32 receiving networks, seven
overlapping-catchment systems and their actual cropped source paths. The same
unit-amplitude Gaussian fluctuations enter all sources. Source area shares and
individual mean path times remain explicit. Time is relative to each network's
mean path under an imposed uniform speed; it is not a measured transit time.
Four input durations and ten structural/clock scenarios give **1,280 responses**
for each of the original and saved shortest route definitions.

At input SD / mean nominal path time = 0.1:

| Operation | Reference | Mean peak change, 95% system-bootstrap interval | Central 80% response-duration change |
|---|---|---:|---:|
| Actual unequal source paths | Equalized arrival paths | −21.58% [−26.94%, −15.07%] | +155.26% [+79.48%, +207.04%] |
| Additional common-corridor spreading | Actual unequal paths without spreading | −32.35% [−35.13%, −27.56%] | +60.42% [+43.95%, +76.15%] |
| Source clocks compensated to align arrivals | Synchronous source clocks on unequal paths | +33.16% [+20.21%, +42.67%] | −43.04% [−52.84%, −31.80%] |

These are prescribed-input responses, not measured DOC reduction percentages.
The rows have different reference responses; their percentages cannot be added.
Shorter pulses are more sensitive. For slow inputs (SD / mean path time = 2),
the corresponding path and common-spreading peak changes shrink to −0.92% and
−0.61%. The complete duration grid and high-coverage subset are retained.

The width has an exact decomposition:

\[
\operatorname{Var}(\text{outlet pulse})=
\operatorname{Var}(\text{input pulse})+
\operatorname{Var}_{w}(\text{arrival clocks})+
\tau^2_{\text{common spreading}}.
\]

The common exponential kernel has unit gain and replaces an equal mean amount
of pure translation. Mean arrival time and total anomaly input are preserved.
A lower peak therefore describes redistribution in time, not assumed DOC loss.
Independent numerical integration checks the analytical gain and moments.

### What the junction experiment resolves

Move the junction while preserving each complete source path and constant-speed
translation. Outlet peak, width and duration are unchanged to numerical
precision (maximum checked metric difference below 9e−16). A common translation
changes arrival clock but not peak or width.

Junction position gains an effect when it changes the amount or kind of
transport experienced after sources meet. Under the explicitly imposed
spreading-per-common-length rule, increasing the shared segment from 20% to 80%
of the shortest complete source path lowers the short-pulse peak. That is a
process-dependent common-corridor effect; changing a graph partition alone is
not sufficient. Real branch/common velocities, lateral mixing and storage can
provide further operations, but were not estimated from DOC in this version.

### Whole-form comparison

At the 0.1 duration, the broad-minus-elongated unit-input peak difference is
−0.0608 [−0.2024, +0.0982] for actual paths and −0.0114 [−0.1002, +0.0734]
after the declared common spreading. The ≥80% represented-area subset likewise
does not establish a form ranking. Full shortest-route results preserve the
mechanistic interpretation. We retain the earlier observed morphology lead as
a field lead rather than forcing these controls to reproduce it.

## 3. The overlap-to-excursion link survives actual-case checks

In the preceding fixed 32-network observational panel, 17 receivers have enough
coincident and solo source excursions for comparison. Thirteen show a positive
difference and four show a negative difference. A high value here is a
season/year-adjusted value above that site's 75th percentile, not a regulatory
threshold or a measured maximum of a continuously sampled event.

Receiver-high probability averages **44.43% when at least two observed sources
are high**, versus **24.81% when only one is high**. The receiver-equal difference
is **+19.62 percentage points [ +10.73, +40.58 ]**, based on 5,000 bootstrap draws
of six eligible overlapping systems.

Omitting each eligible system in turn leaves differences of **+16.68 to +28.30
percentage points**. Every corresponding system-bootstrap interval remains
above zero. The observed association is therefore not supplied by a single
monitored river system. Shared receiving observations and nested river segments
are not counted as independent rivers.

The separate same-day comparison gives +17.03 points [ +1.04, +28.61 ]; the
dense Loch Vale case gives 26/42 versus 12/48 receiver excursions (+36.90 points,
year-bootstrap interval [ +16.9, +59.2 ]). Loch Vale is one mainstem-dominated
case, not an independently replicated broad-form experiment. Some upstream
samples were taken after the receiver on the same calendar date, so these
records do not identify actual transit lag.

The earlier real salt/labelled-carbon additions provide an independent measured
transport/process check in one stream: downstream conservative centroids are
60–67 minutes later and central response durations increase by 9–32%. The two
glucose additions show additional carbon-response decline relative to salt,
while the leaf-leachate addition does not. These quantify different measured
operations, not a universal carbon removal rate or a whole-form contrast.

## 4. What is now resolved

| Research question | Conclusion |
|---|---|
| Can path differences reshape a common short DOC-like input? | Yes, in controlled experiments on actual mapped paths; arrival separation lowers and widens the combined pulse. |
| Can source timing change the same network's response? | Yes; offset source clocks can compensate for unequal paths and concentrate arrivals. |
| Does merely moving a junction guarantee buffering? | No under constant-speed translation with unchanged complete paths; an additional transport/process change is required. |
| Can the shared corridor broaden a combined signal without losing input mass? | Yes under the declared causal unit-gain spreading rule, with unchanged mean path times. |
| Are joint source excursions associated with outlet excursions in measured DOC? | Yes in the fixed observational population, with negative cases retained and leave-system-out robustness. |
| Is broad form universally more buffered than elongated form? | Not established. The class contrast overlaps zero in the identical-input controls and the targeted paired field records remain missing. |

## Research closeout

The resulting paper section can state that **river-network structure organizes
the temporal overlap and spreading of DOC signals**, supported by real geometry,
controlled structural operations and replicated observational associations.
It can explain why a long thin river and a tributary-rich broad river may behave
differently, while locating the operative quantities in their actual paths and
shared downstream corridors.

The broad/elongated class label is not the completed mechanistic result. Direct
field confirmation of that contrast requires new coordinated observations: two
or more non-nested tributaries, their junction and the receiving reach; DOC
with actual timestamps, discharge and a conservative tracer/reference; repeat
the same design in comparable elongated and broad catchments. This is future
empirical work, not another iteration of the existing monthly archive.

No new GNN or DOC predictive model was trained. Previous model results and
source-role analyses remain intact. `manuscript_section.md` supplies the completed
results/discussion text; `claim_evidence_matrix.csv` keeps the exact scope of the
scientific conclusions visible.
