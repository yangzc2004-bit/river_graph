# Explicit flow dynamics: completed DOC model iteration

## What this iteration adds

The existing DOC model now has a residual readout that can directly use relative
discharge anomalies, one- and three-month flow changes, and flow observation
age/availability. The environmental forest, spatial encoder, GRU, observation
decay and support-adaptation machinery are retained. Ten bounded features are
concatenated with the 64-dimensional recurrent state; the scalar head gains
only ten weights.

All eighteen neural fits completed across the three existing station partitions
and three training seeds. The comparison separates full numerical flow features
from an otherwise identical input block containing only age, count and validity
information. Existing hydro inputs remain present in both models.

## Performance

MAE in mg/L, with seeds averaged within each partition and partitions weighted
equally. K0 means no target-station DOC observations; K5 uses five fixed target
observations and the same frozen GRU support representation.

| Model | K0 | K5 |
|---|---:|---:|
| Environmental context base with existing support adapter | 1.902823 | 1.596001 |
| Previous tail-weighted native residual | 1.854491 | 1.595924 |
| Flow age, count and availability readout | 1.850776 | 1.588835 |
| Full flow anomaly/change readout | **1.838594** | **1.583868** |
| Previous v4 fusion with GRU adapter | 1.913634 | 1.595209 |

The full flow model reduces K0 MAE by **3.38%** relative to the environmental
base (95% paired station interval **1.69% to 4.93%**). Relative to the previous
native residual, its additional K0 reduction is **0.857%**, with an interval of
**-0.087% to 1.871%**; eight of nine fits and all three partition means improve.

The direct numerical-input comparison is more conclusive at K0: full values
improve on flow availability/age alone by **0.658%** (interval **0.310% to
1.039%**), with improvement in all nine fits. This supports the incremental
predictive value of explicit short-term discharge changes in this readout.

At K5, the complete flow model improves on the previous native residual by
**0.755%** (interval **0.270% to 1.209%**), again improving all three partition
means and eight of nine fits. The direct values-versus-availability difference
is smaller, **0.313%**, with an interval across zero. Thus the K5 gain belongs
to the complete input/readout and subsequent support adaptation; it cannot be
assigned entirely to the three numerical columns.

The full model's K5 error is approximately 0.71% lower than the old v4 fusion
adapters, but those intervals cross zero. All versions remain available for
comparison. These results come from previously studied ST357 station splits
and therefore constitute continued model development.

## High DOC remains the main unfinished component

The new gain primarily comes from concentrations below the source-training
Q90. Relative to the previous native residual:

- K0 non-tail MAE falls from **1.2148 to 1.1909 mg/L**, while Q90 MAE increases
  from **7.4446 to 7.4889 mg/L**. The Q90 increase is 0.0443 mg/L, with an
  interval of 0.0090 to 0.0793 mg/L.
- K5 non-tail MAE falls from **1.0095 to 0.9953 mg/L**. Q90 MAE is nearly
  unchanged, **6.6822 to 6.6845 mg/L**, with the paired difference interval
  spanning -0.0252 to 0.0264 mg/L.
- False Q90 rates decrease from 1.978% to 1.926% at K0 and from 2.203% to
  2.157% at K5. The lower overall error is not obtained by creating more high
  value alarms.

Across distinct target stations, the full flow model improves 104 and worsens
68 at K0 relative to the previous native residual; at K5 those counts are 100
and 72. The mean gain does not imply improvement at every station.

## Implementation and verification

Discharge features use only current/past observed hydro inputs. Undefined
ratios carry zero values and explicit validity flags. Signed flows are retained,
and the features are invariant to a positive conversion of flow units. No
DOC labels or fitted whole-record statistics enter the new feature block.

Both models use tail weight 2 and the same maximum 30-epoch/patience-5 budget.
Source-validation K0 MAE chooses checkpoints and residual scale. All eighteen
fits selected nonzero scales. There were **255 executed epochs** and **25,547
nominal trainable parameters** per fit. The availability-only control has three
inactive numerical columns. Recorded training/product time totals **86.4 s**,
excluding previous expert training, analysis and replay.

- All nine packages, all ten flow columns, full-grid predictions and every
  adapted query group replayed from saved weights; neural/feature/query arrays
  are bitwise identical. Forest recomputation is checked separately within
  floating-point tolerance.
- Old zero-extra-feature checkpoints retain their exact predictions and
  serialized summaries.
- Full suite: **593 passed, 2 skipped**. Ruff passes. Historical artifact audit
  exits 0. The existing historical evidence qualifications remain unchanged.

```bash
uv run python scripts/run_ladder.py --experiment doc-flow-residual-v1
uv run python scripts/verify_doc_flow_residual_v1.py
uv run python scripts/analyze_doc_flow_residual_v1.py
```

## Next scientific step

Retain the added flow information and focus on how its effect changes with the
local ecological and recurrent state. The current scalar head adds one shared
linear flow correction across stations. A compact interaction between the
existing recurrent state and the three flow-change values is the next natural
comparison: can it improve high DOC without losing the ordinary-concentration
gain? Keep the additive readout as its direct control and change this one
modeling choice before adding more mechanisms.

Detailed tables and paired intervals: [analysis/findings.md](analysis/findings.md).
Weights and products: [runs](runs/).
Replay: [verification/replay_checks.json](verification/replay_checks.json).
