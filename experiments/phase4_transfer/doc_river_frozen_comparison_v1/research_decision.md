# River structure information in DOC reconstruction

## Research conclusion

The four-procedure comparison did not produce a useful river correction to the
complete DOC model. Structural conditioning was rejected by calibration in both
the whole-region study and the supplemental nested station study. The small
simple-upstream correction selected in one station partition increased mean
error by 0.162%, with a paired interval crossing zero. Keep the complete model
as the prediction product. River form remains a process hypothesis to test
against observed responses; these results do not establish an additional
predictive contribution from morphology.

## What was compared

The complete ecological and temporal predictor was fixed. Three regularized
readouts added simple upstream residual aggregation, actual upstream path and
whole-network morphology interactions, or matched non-ancestor source values.
Actual river relations were derived from mapped directed paths, not ecological
similarity. The non-ancestor control had exactly the same availability, age,
lag slots and feature layout. It replaced source values, retaining fictitious
path associations solely for the control.

This is an information experiment on an existing deep model with new small
graph readouts, not 168 newly trained neural networks. No backbone, ecological
fusion, lag set or test-dependent regional winner was changed.

## Whole-region results

Five HUC4 regions and three seeds completed 15 packages and 60 procedure
scores. All 104 test stations and 7,262 observed station-months were retained.
Every added readout selected zero, so predictions were exactly the saved base.

| Procedure | Region-equal K0 MAE mg/L | Q90 MAE mg/L | Station-equal MAE mg/L |
|---|---:|---:|---:|
| Current complete | 2.249670 | 10.454788 | 2.479406 |
| Simple upstream | 2.249670 | 10.454788 | 2.479406 |
| Structure upstream | 2.249670 | 10.454788 | 2.479406 |
| Matched non-upstream | 2.249670 | 10.454788 | 2.479406 |

Matching retained upstream support in 606 cells, 8.34% of this population,
at five stations. HUC4 1019 and 1101 had no supported test station. The cyclic
calibration regions provided 0, 1, 1, 0 and 3 supported stations for HUC4
1013, 1019, 0708, 1030 and 1101 respectively. With only one supported station,
station CV cannot learn that station's signal from another supported station:
all nonzero-regularization candidates can tie the unchanged predictor. Thus
the identical geographical predictions are partly a design-identifiability
result, not evidence of architecture equivalence or an absent river process.

On the separate fixed-query support curve, all four procedures remain identical:
K0, K1, K3 and K5 MAE are 2.262714, 2.158930, 1.988985 and 1.893021 mg/L.
K5 improves that fixed-query K0 by 16.34%. This is local-support adaptation,
not a river-structure gain, and uses a different population from all-cell K0.

## Additional station comparison

After recognizing the calibration degeneracy, a documented supplemental study
kept the same four procedures and split the saved receiving stations into three
outer folds, balanced by upstream availability without reading receiving DOC.
Each correction was fitted on the other two folds and selected by inner station
CV. Every receiving station remained excluded from the source bank.

Splits 142, 143 and 144 with three seeds completed nine packages and 27 outer
blocks, yielding 108 procedure scores. They cover 140 distinct stations and
7,897 distinct station-months; 50 stations and 2,548 cells have matched upstream
support in at least one partition. Libraries vary by partition. This increased
support makes the correction experiment more informative than the regional
case, while the base's previous selection on these validation labels means the
study is retrospective source development, not independent confirmation.

| Procedure | Split-equal K0 MAE mg/L | MAE reduction versus current | Q90 MAE mg/L |
|---|---:|---:|---:|
| Current complete | 1.723002 | Reference | 8.895024 |
| Simple upstream | 1.725797 | −0.162% | 8.895248 |
| Structure upstream | 1.723002 | 0, correction not selected | 8.895024 |
| Matched non-upstream | 1.723002 | 0, correction not selected | 8.895024 |

For simple upstream, 5,000 paired station draws give a MAE-reduction interval
of [−0.458%, +0.013%]. Its supported-cell point estimate is −0.857%, with interval
[−1.866%, +0.109%]. This is a slight point-estimate deterioration, not an
established harmful effect. Three of 27 simple readouts were selected; all
27 structural and all 27 non-ancestor readouts selected zero. Consequently,
structure versus non-ancestor cannot establish the value of real connections:
neither branch supplied a deployed correction. A structural advantage over
the slightly worse simple procedure is likewise not a positive river gain.

Zero paired intervals for identical predictions describe the selected
procedure's exact output. They are not uncertainty bounds on all possible
river architectures or an equivalence test for physical networks. Station
bootstrap retains whole monthly series, joins overlapping partitions, averages
training seeds and weights represented partitions equally.

## What the morphology analysis now tells us

The original measured forms were retained: elongated and tributary-rich,
mainstem-dominated with sparse tributaries, and broad and tributary-rich.
Upstream monitoring is uneven across them. In the station study, matched
support occurs in 436 of 2,400 elongated-network cells, 14 of 1,029 sparse
mainstem cells, and 1,974 of 4,116 broad-network cells: 18.2%, 1.4% and 48.0%.
No form group acquired a positive structural correction in this experiment.

These are monitoring-availability differences, not measured DOC effects of
form. They explain why an apparently stronger signal in broad networks could
be confounded with having more upstream information. The source-matched
comparison prevents crediting that availability advantage to structure alone.

Previous controlled pulse experiments showed how arrival separation and
spreading can alter peaks. The present observation-based comparison did not
show that adding measured morphology improves new-station DOC reconstruction.
These are different scientific statements and should remain separate in the
paper. The current results support neither a universal broad-versus-elongated
DOC ranking nor a causal interpretation of network-shape coefficients.

## Research decision

Close this four-procedure modelling round. Retain the current complete model;
do not merge the river readout or advertise a morphology-performance gain.
Do not continue adding geometry interactions to the same monthly observations
and call the repeats confirmation. The next structural question should compare
observed upstream-to-downstream responses under comparable hydrological
events across forms, with enough simultaneous sampling to distinguish arrival
timing, spreading and source magnitude. That would test the proposed mechanism
directly. More reliable high-DOC prediction remains a separate model objective.

## Deliverables and checks

The original geographical and supplemental station designs, all fitted
readouts, matched donors, bound predictions and calibration scores are saved.
Baseline and correction replay is bitwise; receiving stations are excluded from
donor banks and outer correction fits; unsupported predictions remain exactly
unchanged. Figures were rendered and visually inspected. Analysis can be
regenerated with the verifier, two analyzers and shared plotter in `scripts/`.
These are ST357 studies, not external-basin validation.

The full workspace suite passed 1,337 tests with two explicit skips. Ruff and
the historical artifact audit passed. Historical audit limitations, including
old missing sidecars, are preserved in its log. Old model results and large
local source fits were retained.
