# River form and observed low-to-high flow DOC responses

## Scientific question

Do elongated and broad tributary-rich river networks respond differently when
flow changes within the same river? The main comparison concerns **changes** in
DOC, rather than a ranking of long-term concentrations. Basin area and ecology
provide matching/adjustment context; river form remains the focus.

The earlier form, routing, hydrologic-activation and same-month results have
been seen. This source-role exploratory extension reuses the fixed geometry
classes and exact covariate-selected pairs. It does not repeat the earlier
source-placement moderation analysis or modify any existing result/model.

## Flow states, defined before reading DOC response results

At each of the existing 297 stations, use all positive, finite, measured
monthly discharge in the calendar span of its permitted source DOC record.
Include months without DOC. Define low/middle/high flow by that station's
one-third/two-thirds log1p discharge quantiles. Thresholds depend only on flow,
its visibility and the observation calendar, never on DOC values. Reference
records require at least 24 measured-flow months and distinct thresholds.
Flow is the existing monthly aggregate in cfs. States mean low/high relative
to that station; they do not imply the same absolute discharge or an event.

DOC remains restricted to the union of permitted source-training roles
142/143/144. No geographic or external target scores enter this study.
The fixed high-DOC threshold remains 10 mg/L.

## Populations

1. Station response: at least 24 permitted DOC+positive-flow months, six
   calendar months, two calendar years, and six DOC observations in each low
   and high state. Availability-only inspection found 204 candidate stations.
2. Matched response: reuse all 29 original pairs; restrict both members to the
   **same** DOC+positive-flow calendar months. Apply the criteria above to
   this common record and both members' states. This yields ten elongated/
   broad pairs in six HUC4 regions and one pair for each sparse-class contrast.
   Retain all exclusions. No rematching or DOC-based reclassification.
3. Observed-temperature sensitivity: apply the same criteria to months with
   measured temperature; pairs require temperature at both members.
4. Joint states: among eligible primary pairs, compare months when both
   members are low and months when both are high, requiring four months of
   each joint state. This compares the B-minus-A concentration difference on
   exactly the same dates. Its seasonal distribution is reported separately.

## Responses

For each station/member, retain low/middle/high DOC means, log1p means, SD,
high-DOC frequency and numerator/denominator, discharge, temperature and season.
Report high-minus-low native mean, log1p mean and high-value frequency. These
are observed monthly contrasts, not measured event buffering or hysteresis.

Primary adjusted response is the high-versus-low coefficient in an unpenalized
within-station regression of log1p DOC on:

- intercept, middle and high state indicators;
- annual sine/cosine and centered calendar year;
- centered observed temperature in the sensitivity only.

All fits use the same fixed design. Save coefficients, ranks and residual
degrees of freedom. Rank-deficient designs have no adjusted coefficient; raw
responses remain visible. A second coefficient on native DOC is reported as
a sensitivity, not substituted for the primary log response.

For each matched pair compare **response B minus response A**, giving pairs
equal weight after their within-river fits. Flow-state changes can occur in
different months even within the common calendar: the joint-state diagnostic
and season summaries make that distinction visible.

## Which internal structures explain response differences?

Use all eligible station adjusted log responses as descriptive outcomes.
Fit fixed station-equal linear association models with log basin area, wetland,
forest, agriculture, urban, precipitation, climate temperature, latitude,
longitude, observed high-minus-low log-flow contrast, and HUC2 indicators.
Compare form labels and three separately added feature blocks: footprint,
branch organization, and paths. Numeric modifiers are standardized by the
eligible station population SD; constant terms are omitted and rank is saved.
No source-placement descriptors, feature selection or parameter search.

This is an explanatory association diagnostic conditional on the measured
station responses, not a new reconstruction model or a causal effect estimate.

## Uncertainty and outputs

Use 5,000 whole-HUC4 bootstrap draws (seed42). Station summaries give each
station equal weight; matched summaries give each pair equal weight. Retain
stations/pairs from resampled regions together. Refit the association models
in each geographic draw; report estimable draw counts. Single-region pair
contrasts have unavailable geographic intervals. Also report omitted-region
influence. Intervals are exploratory, with no multiple-comparison adjustment.

Save flow-only references and thresholds, all inclusion ledgers, observed
station/month records, state summaries, fitted station coefficients, paired
responses, joint-state/calendar evidence, structure associations, English
research decision and English/Chinese scientific figures. Necessary tests
cover threshold independence from DOC, exclusion of nonpermitted DOC,
shared-calendar alignment, hand-calculated differences and bootstrap grouping.

Scientific reading: separate a general DOC-flow response from an additional
river-form difference, and determine whether branch/path measurements offer
more explanation than a coarse long-versus-broad label. Keep this version fixed
after reading its responses; follow-up mechanisms receive a new study version.
