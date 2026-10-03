# State-dependent flow corrections: completed DOC iteration

## Model change

The current DOC reconstructor now supports an interaction between its existing
64-dimensional recurrent state and three causal discharge anomaly/change
features. The residual head gains 192 weights. This lets the same flow change
produce different corrections in different learned station states.

Eighteen residual fits completed: three station partitions, three seeds and two
arms. One arm updates the GRU, observation decay and expanded head; the other
keeps the original GRU/decay fixed and trains only the same head. Environmental
forests, spatial encoders and the existing K-shot support representation are
reused. These are extensions of the existing model, not eighteen complete
retrainings of its forest and spatial backbone.

## Performance

MAE in mg/L; seeds are averaged within partition and the three partitions have
equal weight. K0 uses no target-station DOC observations; K5 uses five reserved
observations and the existing frozen GRU support adapter.

| Model | K0 MAE | K5 MAE | K0 Q90 MAE |
|---|---:|---:|---:|
| Environmental context base | 1.902823 | 1.596001 | 7.5586 |
| Previous additive flow readout | 1.838594 | 1.583868 | 7.4889 |
| Interaction with frozen recurrent memory | 1.852352 | 1.584619 | 7.6236 |
| Interaction with updated recurrent memory | **1.835240** | **1.581605** | **7.4371** |

The updated interaction model improves overall MAE by **0.18% at K0** and
**0.14% at K5** relative to the additive model. Their station-bootstrap
intervals cross zero: -0.20% to 0.53% and -0.53% to 0.71%, respectively.
It reduces K0 MAE by **3.55%** relative to the environmental base, with an
interval of **1.92% to 5.01%**. Thus this iteration adds a modest increment
to the previous model rather than a large overall accuracy change.

The clearest interaction increment is at high DOC without target observations:
Q90 MAE falls by **0.0518 mg/L (0.692%)**, with a paired interval of
**0.0352 to 0.0670 mg/L improvement**. All three partition means and all nine
fits improve. Non-tail MAE changes by +0.0022 mg/L, with an interval crossing
zero. The false-high-DOC rate changes by +0.024 percentage points, also with
an interval crossing zero. K5 Q90 MAE falls from 6.6845 to 6.6746 mg/L, but
that incremental difference is not resolved by its interval.

Updating recurrent memory matters within this interaction architecture:
compared with the frozen-memory arm, Q90 MAE improves by 0.1865 mg/L at K0
and 0.0334 mg/L at K5, with both intervals favoring the updated model.
Freezing the old representation and adding a larger head alone does not
recover the same high-DOC performance.

The constant-only K5 adapter also benefits from the interaction, reducing MAE
by 0.47% versus its matched additive control (interval 0.17% to 0.73%). The
stronger GRU support adapter absorbs much of that extra benefit. Both adapter
results are retained; the simpler one is not substituted for the main comparison.

## What this resolves and what comes next

The result supports state-dependent flow correction as a compact way to improve
high-DOC reconstruction. It does not support further head widening as the
next priority. In two partitions, source-training loss continues to decrease
after the best validation epoch while validation MAE deteriorates. Simply
running every fit longer is therefore unlikely to resolve the main weakness.

The follow-on source/validation diagnostic is also complete. Ordinary DOC
concentrations are closely matched: source versus validation means are 3.895
and 3.896 mg/L, with context residual means of -0.380 and -0.362 mg/L.
High-DOC underprediction is more heterogeneous. In partition 144, the source
tail has mean DOC 17.071 mg/L and mean context prediction 11.418 mg/L;
validation has mean DOC 23.383 mg/L but mean prediction only 7.855 mg/L.
The corresponding tail residual increases from 5.653 to 15.528 mg/L. The other
two partitions show slightly smaller validation tail residuals than source OOF.

This identifies station-specific tail transfer as the next problem; there is
no consistent global OOF-to-full-fit residual offset. OOF forests use fewer
stations, and the populations differ, so this comparison cannot isolate a
single cause. The next model investigation should target how ecological and
historical station information represents high-DOC regimes at unseen sites,
with the current compact interaction retained as a candidate. Increasing the
tail weight uniformly or widening the head would not directly address the
observed heterogeneity.

The readout inspection is complete. In the updated model, the interaction
component has a mean absolute contribution of **0.1326 mg/L**, compared with
0.0154 mg/L from the three numerical additive flow terms. Components can cancel,
so these magnitudes are not attribution percentages. Flow-anomaly coefficients
are mostly negative in partition 142 and positive in partitions 143/144;
one- and three-month flow-change coefficients are nearly all positive.
The model is using the interaction, with a response that depends on the learned
states and training partition. This diagnostic uses validation queries with
all non-training DOC labels removed before feature reconstruction.

The station partitions have been examined in previous iterations; these remain
model-development results. K-shot observations span the historical record, so
the complete adapted product is retrospective reconstruction. Flow interactions
are predictive relationships in learned states, not physical or causal coefficients.

## Reproduction

Both arms use the same 30-epoch ceiling, patience 5, native MAE with tail weight
2, and validation-only checkpoint/scale selection. All eighteen selected scales
are nonzero. There were **255 executed epochs** and **74.2 seconds** of recorded
residual training/product time, excluding earlier expert training and analysis.
The frozen arm trains 267 parameters; the updated arm trains 25,739.

All nine packages replay from saved weights: 2,101,302 full-grid rows across
the packages and all 216 model-by-K query groups. Feature, neural residual and
query outputs are bitwise identical; independent forest recomputation differs
by at most 4.97e-14 and is checked separately by tolerance. Frozen GRU/decay
weights remain exactly unchanged.

Full suite: **598 passed, 2 skipped**. Ruff passes. The historical artifact
audit exits zero; its existing historical provenance qualifications remain.

```bash
uv run python scripts/run_ladder.py --experiment doc-flow-interaction-v1
uv run python scripts/verify_doc_flow_interaction_v1.py
uv run python scripts/analyze_doc_flow_interaction_v1.py
uv run python scripts/diagnose_doc_flow_interactions.py
uv run python scripts/diagnose_doc_residual_shift.py
```

Detailed results: [analysis/findings.md](analysis/findings.md).
Saved models and products: [runs](runs/).
Replay: [verification/replay_checks.json](verification/replay_checks.json).
Readout inspection: [analysis/interaction_diagnostics.md](analysis/interaction_diagnostics.md).
Residual distributions: [analysis/residual_shift_findings.md](analysis/residual_shift_findings.md).
