# DOC source-retrieval development, version 1

All nine source-validation packages completed: partitions 142/143/144 and seeds
42/43/44. The full 10-procedure panel covers every valid validation DOC cell.
Target-station DOC, pH and conductance channels are absent. No geographical or
external target evaluation was performed in this development experiment.

## Results

The current complete predictor has equal-partition K0 MAE **1.829579 mg/L**.
The matched daily-input ExtraTrees comparator has **1.963925 mg/L**. The retained
predictor reduces MAE by **6.841%** versus these trees (paired station-bootstrap
95% interval **3.503–10.350%**). This is existing model performance in a source
validation panel, not a gain caused by the new retrieval module.

Retrieval, uniform attention and zero-source-value retrieval predictions are
identical to the current predictor. Every retrieval checkpoint selects epoch 0:
the new projection remains zero. Source bank access is therefore not sufficient
evidence that learned retrieval contributes useful information.

Hydro pretraining followed by 30-epoch DOC training yields MAE **1.835878 mg/L**:
**0.344% worse** than the retained model (gain interval **−1.553–0.770%**).
Q90 MAE is 9.242435 versus 9.225863 mg/L. Keep this pretraining experiment as a
completed negative development result; do not promote it to geographical tests.
The retained DOC expert had a longer historical training budget, so these
comparisons establish candidate performance, not an isolated matched-budget
effect of hydro pretraining.

## Next iteration

Version 1 optimizes the donor memory against the forest residual alone, while
validation mixes that memory with a strong temporal correction. The selected
projection does not replace the temporal correction. The next version will
instead learn **donor minus current local residual**, with zero initialization
reproducing the current complete prediction. Forest, donors, candidate count,
attention size and data roles remain fixed. Shared query-design columns will
not serve as prediction values. Uniform-weight and donor-value removal tests
will distinguish source selection from simple rescaling of the local correction.

The current model stays the reference. Whole-HUC4 confirmation follows a useful
source-validation candidate. The five HUC4 roles and DOC-only external availability
screen have been prepared independently of target prediction results.

## Reproduction

Run `verify_doc_source_retrieval_v1.py`, `analyze_doc_source_retrieval_v1.py
--bootstrap-draws 5000`, then `plot_doc_source_retrieval_v1.py` through `uv run`.
Nine packages passed nested station exclusions, sidecar identity checks and
bitwise attention checkpoint replay. The original execution source is retained
in `code_snapshot/`. After completion, a cache-reload key typo was corrected in
the working script (`cells{i}` to `query_cells{i}`); no fitted outputs changed.
Exact execution recovery uses the saved snapshot, not a claim that the modified
working script is the historical code.
