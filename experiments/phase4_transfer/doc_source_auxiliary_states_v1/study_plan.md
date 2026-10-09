# Source water-quality supervision for the existing DOC representation

## Question and starting evidence

Recent ecological and hydro-state extensions leave overall source K0 MAE close
to the retained1.769 mg/L. Rare high-DOC observations remain the main error
burden. The next mechanism tests a different information source: learn the
existing ecology/GRU state from richer source-station pH and conductance labels,
then fit the same DOC residual. Output and new-site inference remain DOC-only.

The source audit aligns both auxiliary datasets to the357-station/654-month
DOC grid. Depending on partition, source stations supply34,296–36,671 pH
labels and39,467–41,719 conductance labels. Receiving station chemistry is
neither inspected nor used. The older cross-analyte station-profile experiment
required receiving pH/EC and therefore does not answer this K0 question.

## Implementation and experiment

Use the retained environmental tree and station-blocked OOF reference, existing
ecological encoder, observation-aware GRU and native linear DOC residual head.
Do not combine the unconfirmed nonlinear, composition or hydro-state candidates.

Pretrain the same trainable backbone parameters with two disposable auxiliary
heads: source pH and source log1p conductance. Fit label centering/scaling on
source auxiliary training stations only. The first existing source station fold
is internal auxiliary validation; remaining source folds train. No DOC validation
or geographical/external labels select pretraining. K0 input views continue to
hide each receiving site's entire water-quality history and availability.

Use30 auxiliary epochs and the existing30-epoch DOC phase, patience5 for DOC.
Compare true source supervision and a source-label-shuffle control with the same
auxiliary training budget; choose each auxiliary checkpoint from its internal
source validation. Shuffling preserves per-analyte training label distributions
and valid cells. Pretraining heads are discarded, and the ordinary existing DOC
residual/portable architecture receives the warm backbone weights.

Auxiliary fitting uses every available label in the four fitting station folds.
The two standardized MSE terms have equal target weight. The disposable head
learning rate is1e-3, GRU/decay1e-4 and the existing trainable ecological/self
encoder parameters1e-5. The backbone remains64-dimensional with12-month causal
windows. Both arms complete30 auxiliary epochs and select the lowest equal-target
MSE across all labels in the first source fold, including epoch zero. Heads do
not consume water-quality values or masks. No new feature or network is added
to DOC inference. A fresh ordinary DOC residual constructor records the warm
backbone as its initial weights so DOC fitting cannot silently reset pretraining.

Development remains142/143/144 ×42/43/44. Preserve the current complete model
and strong trees. Report true versus shuffle and actual complete-model gains,
Q90, bias, partition consistency, auxiliary validation loss and backbone change.
Verify auxiliary receiving-label isolation, source-only scaling, causal input
windows and saved-state replay. Do not claim biochemical process states or
cross-analyte transfer from a lower auxiliary loss alone.

If a complete candidate adds useful DOC performance, fix it before geographical
replication. Current old geographical/external queries remain historical results,
not development targets. This plan and data audit authorize the next source-only
implementation. The runner is `run_ladder.py --experiment
doc-source-auxiliary-states-v1`; all nine source packages use the same code and
the two matched auxiliary arms. No fitting had started when this design froze.
