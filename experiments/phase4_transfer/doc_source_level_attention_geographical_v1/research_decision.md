# Research decision: seasonal source bias in geographical DOC transfer

## Completed study

All25 fixed packages are complete: HUC4 regions1013/1019/0708/1030/1101,
each with seeds42–46. Four arms fit full current source values, full earlier-
season values, fixed-prior allocation and individual seasonal means. The100
neural fits reuse250 double-held-fold forest references; no new forest is fitted.
Receiving K0 stations supply no DOC, pH or conductance, including their history
and derived visibility features. Every package replays bitwise through source
libraries, candidates, initial states, neural/complete predictions, diagnostics
and fixed-query support products.

The source-development study found an additional0.637% complete MAE improvement
from preserving donor seasonal bias alongside current relative departures.
This geographical study replicates that fixed change without enlarging the
37,900-parameter model, changing the30-epoch/patience5 budget, or selecting a
different model for a region or K. The seasonal-only arm retains current
aggregate readout features; it removes current departures from individual donor
values, not from every input.

These are previously evaluated ST357 geographical roles: retrospective
geographical replication, not independent external-basin validation.

## Main geographical result

The primary K0 population contains every valid test DOC cell. Seeds are averaged
within region, then the five regions are weighted equally. Intervals use5,000
paired station resamples, keeping each station's months and model outputs
together. Station-equal MAE is reported as a separate estimand.

| Procedure | MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Full current source complete |2.266236|2.492171|10.431318|
| Preceding anomaly-only complete |2.258278|2.486052|10.409544|
| Actual retained complete |2.282266|2.504735|10.519953|
| Full fixed-prior complete |2.268826|2.494689|10.463045|
| Full earlier-season complete |2.283633|2.510105|10.526034|
| Individual seasonal means complete |2.274233|2.506712|10.484878|
| Strong station-hidden trees |2.344885|2.547670|10.520142|
| Aggregate-source trees |2.314529|2.516639|10.481605|
| Older complete model |2.360283|2.556920|10.632267|

**The extra source-development improvement did not reproduce geographically.**
Relative to the immediately preceding anomaly-only procedure, full current
source values worsen complete MAE by0.352%: gain-0.352%
[-0.996%,0.364%], with only one of five region averages improving. Q90 gain is
-0.209% [-0.476%,0.039%], with two improving regions. These intervals do not
establish a systematic overall deterioration, but they provide no additional
geographical performance benefit for retaining donor seasonal bias.

Against the actual retained complete model, overall gain is0.702%
[-0.336%,1.883%], four positive regions. Q90 gain is0.843%
[0.332%,1.395%], all five positive.67 of104 station averages improve and37
worsen versus retained;45 improve and59 worsen versus anomaly-only. Signed
bias is-0.675490mg/L, compared with-0.655575 for anomaly-only and-0.649449
for retained. Lower point MAE does not remove concentration underestimation.

Gain over strong trees is3.354% [0.263%,6.448%], positive in four regions.
Gain over aggregate-source trees is2.087% [-1.302%,5.461%], also four positive.
The3.985% gain over the older complete model includes earlier upgrades. These
are full-procedure comparisons: the bare tree does not receive the attention
branch's individual source keys. None is the isolated seasonal-bias increment.
The5% spatial improvement objective remains unmet.

## What the controls show

Complete current-source gain over earlier-season is0.762%
[0.038%,1.498%], four positive regions; its Q90 gain is0.900%
[0.380%,1.454%], all five positive. Thus current source departures retain value
relative to older departures within this full-value model.

Gain over fixed-prior complete is0.114% [-0.295%,0.605%], only two positive
regions. Gain over seasonal-only complete is0.352% [-0.229%,0.986%], four
positive regions. Their Q90 gains are0.303% [0.099%,0.536%] and0.511%
[0.287%,0.777%]. Tail allocation improves these controls, while the overall
advantage of learned selection or individual current departures remains uncertain.

Native-only MAE is2.265472, versus anomaly-only2.259501: gain-0.264%
[-0.993%,0.622%], only one positive region. Native-only Q90 gain against
anomaly-only is-0.308% [-0.607%,-0.009%]. Against retained native-only,
overall gain is0.005% [-0.839%,0.881%] and Q90 gain0.726%
[0.217%,1.196%]. The full fusion benefit should not be attributed entirely to
the attention branch.

## Regional and few-observation behaviour

| Held-out HUC4 | Full source MAE | Anomaly-only MAE | Retained MAE |
|---|---:|---:|---:|
|1013|5.592494|5.586720|5.626815|
|1019|1.666321|1.637584|1.619163|
|0708|1.690753|1.728603|1.745890|
|1030|1.257548|1.249417|1.260627|
|1101|1.124062|1.089066|1.158835|

Only0708 improves over anomaly-only. All regions and their failures remain in
the analysis. Do not combine0708's full-source model with other regions'
anomaly-only models.

The K curve uses a separate identical query at all K after reserving all five
support candidates. Its K0 therefore differs from primary all-observed K0.

| K | Full source complete | Anomaly-only complete | Retained complete |
|---|---:|---:|---:|
|0|2.279522|2.271570|2.296495|
|1|2.158084|2.148383|2.160459|
|3|1.984496|1.966872|1.987946|
|5|1.902983|1.887012|1.909790|

Overall gains against retained are0.739%,0.110%,0.174%,0.356%; all intervals
cross zero. Against anomaly-only, K3/K5 gains are-0.896%
[-1.451%,-0.242%] and-0.846% [-1.249%,-0.316%]. K5 worsens in all five
regions. This full-source change does not preserve the preceding model's
few-observation improvement.

Source-derived Q90 yields904 unique tail cells in1013, mean recall0.9852;
0708/1019 have66/43 cells with zero recall.1030/1101 have15/11 cells with
zero recall and unstable flags. Seed repetitions do not increase the number
of ecological observations. Lower Q90 MAE is distinct from event detection.

Mean zero-innovation prior mass is0.496 for learned full-current attention,
0.761 for fixed prior,0.669 for earlier-season and0.516 for seasonal-only.
Corresponding per-head Shannon entropies are0.668/0.659/0.610/0.647 in raw
natural-log units. These describe allocation among ecological neighbours;
they are not river transport coefficients. Full current improves the missing-
hydro stratum versus anomaly-only but worsens its1–3-donor and no-donor strata.
Environmental strata have differing contributing-region counts and remain
descriptive comparisons, not rules for selecting a regional model.

## Decision and continuing research

Close this geographical matrix. Keep the preceding anomaly-only relative
attention as the stronger complete candidate in this fixed series. The source
seasonal mean is useful in development roles but is not a reliable extra
geographical correction by itself. Do not change this evaluated model using
its geographical results or update the retained external scores.

The next source study, `doc_source_monthly_hydro_attention_v1`, was specified
and started before opening these geographical numerical results. It adds four
source temperature/discharge/visibility columns to the existing keys; the
receiving GRU already contains local hydrology. New coefficients start at zero.
Current, source-mean and zero-key controls retain the same source values and
38,156 parameters. Its technical package replayed bitwise; all nine source
packages have completed27 fits, with their source-only analysis now running.
This asks whether actual source water conditions improve experience selection,
without adding a backbone or refitting forests. Its model choices use source
roles142/143/144, not geographical or external outcomes.

All figures and5,000-draw tables are reproducible; the actual PNG was inspected.
The fitted version passes995 tests with two skips. The new source-key workspace
passes999 with two skips; Ruff and the historical artifact audit pass. Executed
source copies, large local caches and prior model products are retained. Git
submission remains pending because this environment prohibits index writes.
