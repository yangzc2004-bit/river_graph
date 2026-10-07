# River organization and downstream DOC signal transmission

## Research decision

Keep the river network at the centre of the study. The productive question is
how tributary integration and channel paths reshape a DOC signal, rather than
whether one broad shape category always has a higher DOC concentration.

Three connected observations motivate the next mechanism experiment:

1. Downstream DOC anomalies are generally smaller than the monitored upstream
   anomalies in the source-role monthly record.
2. Path dispersion and normalized path length show different associations:
   dispersion with variability buffering, length with weaker upstream/downstream
   synchrony.
3. Dense channel organization does not simply mean stronger buffering. In the
   confluence population, density is associated with larger downstream/source
   anomaly ratios. Additional unmonitored tributaries and distributed inflows
   are one explanation to investigate, not a demonstrated cause.

Ecology remains background information and a covariate. This study does not
restart the terrestrial-source comparison as its main research topic.

## Population and measurement

This follow-up uses permitted source observations from the deduplicated source-
train union of role splits 142/143/144. It reuses the earlier location-screened
independent-tributary and connected-path inventory. The exploratory morphology
panel is screened in the same way as the preceding morphology study.

| Population | Receivers | HUC4 | Shared monitored-station systems | Shape counts: elongated / sparse / broad |
|---|---:|---:|---:|---|
| Monthly area-proxy mixing | 38 | 23 | 17 | 11 / 1 / 26 |
| Positive-flow mixing sensitivity | 35 | 21 | 17 | 10 / 0 / 25 |
| Connected monthly paths | 99 | 39 | 36 | 35 / 3 / 61 |
| Same paths/months with complete local hydro | 53 | 28 | 21 | 21 / 2 / 30 |

The inclusion ledger retains unmatched receivers (two mixing and four path
receivers in the broad monthly populations). Multiple incoming paths or pairs
are averaged within receiving stations. Receivers, not pair-month rows, get
equal weight. Shared-station-system bootstrap is primary; HUC4 intervals are a
sensitivity. Unobserved shared drainage can still connect nominally separate
systems. Sparse mainstem-dominated shape has too few observations for a general
class interpretation.

The primary variability measure is the log downstream/source standard-deviation
ratio after calendar and year adjustment of **log1p(DOC)**. For two branches,
source SD is the root mean of their anomaly variances. This separates signal
amplitude from mean concentration. A smaller amplitude does not measure DOC
consumption, retention, or a load budget.

## Findings

### 1. Actual downstream buffering is present

Area-proxy mixing receivers have mean log anomaly SD ratio **−0.1858**, connected-
system 95% CI **[−0.2967, −0.0660]**. Exponentiating the receiver-mean log ratio
gives a geometric-mean ratio of approximately **0.831**: downstream anomalies
are about **17% smaller** on this particular scale. This is neither a native-
concentration reduction nor an arithmetic-mean SD reduction.

The corresponding two-source mixture ratio is −0.1684. Downstream-to-mixture
ratio is −0.0174, CI [−0.1281, +0.0800]. Thus, at the aggregate level the observed
buffering is compatible with the simple mixture's attenuation; it does not
establish extra in-river removal. Source anomaly synchrony is 0.2774, rather
than identical forcing at both branches.

Connected path receivers show log anomaly SD ratio **−0.1354**, CI
**[−0.2050, −0.0469]**, geometric-mean ratio 0.873. Signal rank correlation is
**0.3625**, CI [0.3077, 0.4127]. The average signed log concentration difference
is −0.0156, CI [−0.0959, +0.0518]. Buffering is clearer than a consistent average
concentration decline.

### 2. River organization supplies specific process hypotheses

Separate morphology terms are adjusted for the fixed connection background;
coefficients are per receiver SD of the respective morphology term.

| Setting and association | Estimate | Connected-system 95% CI | HUC4 95% CI |
|---|---:|---|---|
| Mixing: channel density → downstream/source anomaly SD ratio | +0.1438 | [+0.0661, +0.3202] | [+0.0465, +0.2954] |
| Path: path dispersion → downstream/source anomaly SD ratio | −0.0842 | [−0.1637, −0.0093] | [−0.1739, −0.0080] |
| Path: normalized path length → signal rank correlation | −0.0653 | [−0.1270, −0.0108] | [−0.1378, −0.0035] |

All three estimates retain their signs when each shared monitored-station system
is omitted in turn. These are exploratory pointwise associations across several
correlated morphology terms, not independent discoveries or causal effects.
Every term and outcome is retained in the results and figures.

Controlling source synchrony leaves the mixing-density coefficient almost
unchanged: +0.1426, CI [+0.0753, +0.3248]. In the positive-flow monthly population
the estimate is also positive (+0.1522), but its primary interval spans zero
[−0.0062, +0.2732]. HUC4 resampling alone would look more conclusive. The two
populations differ, so this is not evidence that one weighting scheme is better.

Longer normalized paths accompany lower source/downstream synchrony; this can
motivate distributed routing and intermediate-input models. Path dispersion
accompanies lower relative anomaly amplitude; it motivates testing whether
different path timings distribute or cancel a coherent input signal. Monthly
samples cannot identify a travel-time distribution or measure event lag.

### 3. Same-population hydro sensitivity changes the strength of the evidence

The hydro-complete analysis retains precisely the same station connections and
months before and after removing measured temperature and log1p(flow).

| Quantity | Full monthly population | Hydro-complete, calendar-only | Same months, hydro-adjusted |
|---|---:|---:|---:|
| Receiver mean anomaly SD ratio (log) | −0.1354 | −0.1841 | −0.1642 |
| Receiver mean signal rank correlation | 0.3625 | 0.3685 | 0.3004 |
| Dispersion coefficient for SD ratio | −0.0842 | −0.0407 | −0.0162 |
| Normalized path coefficient for correlation | −0.0653 | −0.0563 | −0.0462 |

In the hydro-complete subset, both focal morphology intervals already span zero
**before** hydro adjustment. The weaker evidence cannot be attributed solely to
removing hydrologic variation. The reduced population and its morphology overlap
matter. After adjustment, the overall buffering ratio remains below zero, but
the morphology-specific coefficient is not independently established.

### 4. Broad versus elongated is not a universal DOC ordering

The broad-minus-elongated contrasts for downstream/source SD ratio and signal
correlation span zero in both the mixing and path populations. Their SD-ratio
point estimates even have different signs: mixing −0.1401, path +0.1039.

The area-proxy downstream/mixture SD ratio contrast is negative (−0.1991,
CI [−0.5518, −0.0756]), but the positive-flow sensitivity spans zero. This is a
descriptive follow-up, not a confirmed difference in class-specific retention.

A negative mixing mainstem-share coefficient for mean log concentration
departure has a primary interval below zero. However, one sparse-shape receiver
(`05357245`) has design leverage approximately **0.878**. This result must not
become a general claim that mainstem-dominated networks consume more DOC.
Footprint, internal organization, upstream sampling coverage and flow regime
should be interpreted together.

## Mechanism-focused next work

The next experiment should change **channel organization**, holding landscape
inputs fixed:

1. On real network paths, compare identical input signals with the same total
   flow and source concentration budget. Vary source synchrony, junction
   location, branch balance and path-delay spread separately. Match basin scale
   and report discretization. Start with conservative routing, then add an
   explicitly labelled processing scenario.
2. Test whether the observed dense-network excess variability is associated with
   the number and size of unmonitored tributaries joining between the last
   monitored upstream stations and the receiver. This is a channel-integration
   question; vegetation amounts remain controls.
3. Use the resulting organization descriptors to construct a morphology-aware
   transmission operator. Its scientific role is to represent buffering,
   intermediate contributions and loss of upstream synchrony. Compare against
   the existing same-month river message and no-message controls on source
   development roles before attempting a reconstruction performance claim.

This gives the research a connected story: **real river form → organization of
tributary merging and channel paths → DOC variability transmission → a better
river-information operator**. It avoids assigning a fixed high/low DOC label to
each shape or substituting a landcover study for river structure.

## Delivery and verification

- Five receiver panels, connection records, class summaries/contrasts, all
  conditional associations and leave-one-block-out results are reproducible.
- 5,000 bootstrap draws; all tables recomputed by the verifier.
- 8,779 monthly mixing record rows checked against the permitted source DOC
  cells; changing all other DOC labels leaves the source view unchanged.
- EN/CN scientific figures rendered and visually inspected; a row-heading
  overlap and an English annotation overlap were repaired.
- Full pytest: **1,090 passed, 2 skipped**, 8 existing warnings; Ruff passes.
- Historical artifact audit exits 0; its known G0 conflict and legacy no-sidecar
  records remain reported. That audit does not certify these new mechanism
  tables; the study-specific verifier does.
- No neural models were trained and no geographical or external predictions
  were used in the mechanism analysis.
