# Support-matched spatial validation

The frozen E3 test stations are unchanged. This validation block holds out 20
source stations selected using only the graph and observation masks. Its
station-month fraction with at least one visible upstream source is matched to
the frozen E3 test fraction (0.403), and the block contains both supported and
unsupported stations. The selection is deterministic (`seed=20260930`) and
records 18 train-to-validation directed edges.

The block is used to choose the spatial residual arm and shrinkage settings.
Validation labels are hidden from fitting. The original E3 test remains a
terminal evaluation and is not used for selection. This is a targeted
structure-matched diagnostic, not an estimate of performance over all possible
spatial splits.
