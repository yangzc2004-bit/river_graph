# DOC source-retrieval residual contrast

This development version is designed after seeing version 1. Its single main
change is the **fusion objective**: train donor-minus-local residual contrast
against the existing complete predictor, rather than training donor memory as
an alternative standalone forest-residual predictor.

Use source training/validation roles in partitions 142/143/144, seeds 42/43/44.
Do not evaluate their outer target roles. Retain the same nested OOF donor banks,
20 ecological candidates, two 32-dimensional heads, frozen ecology/GRU states,
30 epochs, patience 5, station-balanced native MAE and zero-initialized projection.
New station water-quality inputs remain entirely hidden.

The donor value is its predicted forest residual minus the current local forest
residual. Query design columns cannot directly serve as value channels; only
source-residual contrast is projected. Zero projection reproduces the complete
current prediction, including its fixed static-memory fusion. This contrast
reallocates an existing forest residual rather than adding it twice.

Source neural weights remain source-trained with station-fold-hidden inputs;
they are not independently trained OOF neural checkpoints. Donor profiles and
their underlying forest fits exclude the whole outer pseudo-target fold.

Keep current model, matched daily-input trees and static memory as comparisons.
Report uniform weights and removal of the donor value while retaining the local
subtraction: the latter can expose gains caused merely by local rescaling.
Use all valid validation cells for K0, equal partition means and 5,000 paired
station-bootstrap draws. These are selected development results. A useful
candidate proceeds to the already prepared whole-HUC4 tasks.
