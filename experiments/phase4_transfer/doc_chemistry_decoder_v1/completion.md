# Nonlinear chemistry-conditioned DOC decoder: completed

Twenty-seven heads were fitted across partitions142/143/144 and seeds42/43/44.
The current DOC backbone stays fixed. A learned eight-dimensional chemical
embedding interacts with its hidden state, adding1111 trainable parameters.
The no-auxiliary, mask-only and chemistry-value modes share architecture,
initialization and auxiliary-availability fallback. Existing chemistry trees
and linear chemical corrections remain fixed controls.

## Main results

MAE is mg/L, seed-averaged within partition and partition-equally averaged.
K is the number of reserved DOC observations used to calibrate each target
station; all K values share the same query cells.

| Model | K0 | K1 | K3 | K5 | K0 Q90 MAE |
|---|---:|---:|---:|---:|---:|
| Retained integrated model | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 7.179187 |
| Nonlinear no auxiliary, integrated | 1.801879 | 1.752870 | 1.619850 | 1.563227 | 7.163501 |
| Nonlinear masks, integrated | 1.800286 | 1.752642 | 1.620066 | 1.564150 | 7.165428 |
| Nonlinear chemistry, integrated | 1.764947 | 1.738600 | 1.614240 | 1.568820 | 7.019544 |
| Previous linear chemistry, integrated | 1.792258 | 1.746092 | 1.619351 | 1.571309 | 7.105102 |
| Fixed chemistry tree | 1.802261 | 1.768522 | 1.565090 | 1.537905 | 7.372282 |

The nonlinear chemistry decoder lowers integrated K0 MAE by2.140% versus
the retained model. The paired station-bootstrap change is−.038591,
95% interval[−.066865,−.015185]. It also improves against its matched no-auxiliary
control (−.036933[−.063709,−.015899]), mask control
(−.035339[−.062586,−.014613]) and preceding linear chemical correction
(−.027311[−.046387,−.012595]). All three partitions improve in these K0
comparisons. All nine packages improve versus the mask control; against the
retained model seven improve, one worsens and one falls back.

Both ordinary-DOC and Q90 error improve at K0 against the retained model:
ordinary MAE1.188732→1.163428; Q90 MAE7.179187→7.019544, with tail difference
−.159643[−.316980,−.018144]. High-value recall increases.831 percentage points,
but false-high rate also rises.0895 points. The added measurements improve
prediction without eliminating the high-DOC underprediction/detection tradeoff.

At K5 the new integrated MAE1.568820 is numerically worse than the retained
1.564984: change+.003836[−.009955,+.016600]. Its small improvement versus
the preceding linear decoder is unresolved. The fixed chemistry tree remains
better at K5 by.030915[.002488,.062056], with lower MAE in all nine packages.
K1/K3 curves are reported descriptively; no target-selected K switch is made.

## Model decision

Preserve the nonlinear decoder as an improved chemistry-informed K0 candidate.
Keep the retained integrated model as the general reference across K. The
matched controls show measured chemical information provides useful cross-station
prediction signal. The nonlinear-versus-linear comparison also changes feature
form and parameter count, so it does not isolate nonlinearity as the sole cause.

The next model question is why the improvement contracts when target-station
DOC support is added. The current support basis was built before chemistry
entered the model. A chemistry-aware calibration representation is a focused
next hypothesis; the present results do not establish that it is the cause.

## Training and replay

All27 fits completed in79.25 summed seconds,643 training epochs. Chemistry
checkpoints select epochs0–73; no fit reaches120. Direct K0 source-validation
MAE changes from1.804428 at initialization to1.769764 for chemistry, compared
with1.799160/1.799376 for no-auxiliary/mask controls. One chemical package keeps
epoch0. Source means and standard deviations precede the learned chemical basis.
Only the source forest base is OOF; its frozen neural correction is source-trained.

Independent replay verified27 head states, the seeded chemical embedding,
source554-feature views, the explicit SiLU/1070-feature native formula,
81 direct adapters and36 ecological/support mixers. All540 query panels,
2315700 rows and324 copied reference panels match exactly. Full-grid output
is9×233478=2101302 rows. Absent-chemistry final component fallback is exact.
The full suite has806 passed,2 skipped and three existing warnings; ruff passes.
The historical provenance audit exits0.

Either auxiliary measurement is known for98.43% of unique observed-DOC queries
but19.735% of genuinely DOC-missing cells. Monthly auxiliary observations can
come from different sample times. This is retrospective reconstruction with
known chemistry, not prediction before these measurements arrive. Station
partitions remain reused development data. See analysis/findings.md,
analysis/interpretation.md and verification/replay_checks.json.
