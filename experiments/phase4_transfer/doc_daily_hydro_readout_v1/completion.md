# DOC episodic support readout: completed

Nine128-parameter readout fits are complete. The selected daily-head prediction
expert, context forest and ecological memory remain fixed. Five fits select
epochs1–5 and four retain epoch0. The nine fits run63 epochs in total; none
approaches the30-epoch ceiling. No forest or recurrent expert is refitted.

## Result and model decision

Native MAE in mg/L; seed means within partition, then the three partitions
equally weighted.

| Integrated support basis | K0 | K1 | K3 | K5 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 1.803538 | 1.755368 | 1.620090 | **1.564984** |
| Refreshed fixed | 1.803538 | 1.755368 | **1.599087** | 1.572464 |
| Refreshed learned | 1.803538 | 1.755368 | 1.604441 | 1.570974 |

Learning improves integrated K5 only0.095% relative to fixed refresh:
delta−0.001490, station95% interval[−0.003791,+0.000481]. K3 instead changes
by+0.005354[−0.004141,+0.015427]. None of the four direct/integrated learned
versus fixed overall intervals excludes zero. Against the existing legacy
integrated candidate, learned K5 is0.383% worse:
+0.005990[−0.001234,+0.014262], with worse point estimates in all three
partitions. Retain the existing daily-head/legacy-support integrated model.

Direct learned K3 improves1.10% over legacy, but fixed refresh already
improved1.45%; additional readout training does not establish incremental
value. The change concerns ordinary DOC. Learned integrated K5 Q90 MAE
is6.634382, compared with6.586994 for legacy; its signed tail bias is
−5.045860 mg/L. Tail/detection contrasts do not establish improvement.
All curves and unfavorable comparisons remain in the analysis; no
target-selected K-specific route is introduced.

## What was learned

Mean source-validation checkpoint loss changes only1.731654→1.730017.
After downstream calibration, source K3/K5 changes are also small. Readout
training does not repair the refreshed representation's five-observation
transfer disadvantage. Extending the same optimization has little evidence
in its favor; four fits already prefer no update. This result tests a global
two-dimensional support map, not all information in the64-dimensional state.

The next experiment compares recurrent decay clocks in
`../doc_recurrent_clock_v1/study_plan.md`, keeping the old support basis fixed.
It tests the advancing DOC-age scalar at receiving stations without DOC
history against an unseen-neutral scalar and within-window discharge age.
The existing main candidate remains available throughout.

## Reproduction and checks

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-readout-v1
uv run python scripts/verify_doc_daily_hydro_readout_v1.py
uv run python scripts/analyze_doc_daily_hydro_readout_v1.py
```

All nine packages pass independent replay, including deterministic
source/validation readout refitting, initial/selected validation objectives,
source native baselines, hidden-state identities, episode schedules,
normalization, downstream selection and query predictions. Constant, legacy
and fixed-refresh controls reproduce their parents exactly; K0/K1 and the
native full grid remain unchanged. The eight specified contrasts use5000
paired whole-station bootstrap draws. The test suite at this point passes
751 tests with two explicit skips; the historical artifact audit passes.

See `PRODUCTS.md`, `analysis/interpretation.md`, `analysis/findings.md` and
`verification/replay_checks.json`. Source forest predictions are OOF; the
fixed neural expert is source-trained. Raw recurrent states are causal,
while record-wide anchor normalization and positive-K reconstruction are
retrospective. These are previously seen development partitions.
