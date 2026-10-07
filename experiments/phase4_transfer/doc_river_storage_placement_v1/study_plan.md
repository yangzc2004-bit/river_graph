# Where storage acts in a real confluence structure

## Question

Why can different routes' conservative storage responses either lower or raise
the combined DOC peak? Compare storage on the shorter-path branch, longer-path
branch and common downstream trunk, with identical forcing and unchanged path
means. This is an exploratory follow-up to the whole-network study; its results
have been seen. It uses no new DOC outcome to select geometry or parameters.

## Real geometry and input scope

Retain the original 297 station-network instances and three planform labels.
For each selected directed tree, calculate unique incremental area upstream
of each reach. At every junction, take its two largest positive-area tributary
subtrees and score 2*A*B/(A+B). Select the largest score, breaking ties by COMID,
among junctions with positive independent branch and common path lengths.
This chooses a major real confluence using geometry alone. If none exists,
record an explicit exclusion; do not substitute a synthetic junction.

In each selected subtree, choose the actual positive-area reach midpoint nearest
its area-weighted mean source distance; ties use COMID. These are two real
representative input locations. Flow shares use the two subtrees' incremental
area sums. Retain the pair's share of total basin area in the outputs.
This two-input structural response is distinct from the preceding uniform input
at every local catchment. It represents the selected tributary pair, not a
complete basin DOC budget. Also repeat on all 59 existing measured monitored
footprints; those endpoints and area-proxy shares remain unchanged.

Normalize lengths by the pair's weighted mean total path. Both tributaries
receive the same unit-height Gaussian pulse. Use SDs 0.075, 0.15, 0.30;
fractions 0.25, 0.5, 1 of a common feasible capacity. The early/late names
refer to path length at conceptual common velocity, not observed arrival time.

## Two matched-strength experiments

Let branch lengths be b_e,b_l, common length c, flow shares w_e,w_l and original
complete delays d_e=b_e+c,d_l=b_l+c. Preserve both d values in every scenario.
For affected paths, replace tau of deterministic translation with a causal
unit-gain exponential response of mean tau. All sources keep their mean delay;
integrated anomaly and steady concentration remain unchanged.

Primary, matched additional variance:

    R = min(sqrt(w_e)*b_e, sqrt(w_l)*b_l, c)
    V = (f*R)^2
    tau_e = sqrt(V/w_e); tau_l = sqrt(V/w_l); tau_c = sqrt(V)

Thus the three placements have the SAME total extra variance and pulse SD.
Remaining differences involve response shape and pulse overlap, not greater
average delay or greater variance. Mean allocated storage is allowed to differ
and is reported explicitly.

Sensitivity, matched allocated mean-time budget:

    Bmax = min(w_e*b_e, w_l*b_l, c)
    B = f*Bmax
    tau_e = B/w_e; tau_l = B/w_l; tau_c = B

This holds flow-weighted storage allocation equal, while total extra variance
can differ. Report both designs; do not select one from outcomes. Repeat the
middle-pulse experiment with equal branch flow shares to separate area balance
from arrival differences. Every translation remainder must be nonnegative.

## Outcomes and decomposition

Record peak, central 80% duration, centroid, variance, source peak times and
peak-time gap. Also record E = sum(w_j*max(source_response_j)), the envelope
of the two individual weighted peaks, and A = combined_peak/E. Then:

    combined_peak = individual_peak_envelope * alignment_ratio
    log(peak/peak0) = log(E/E0) + log(A/A0)

E cannot increase under these conservative kernels. If the combined peak rises,
increased overlap must compensate for attenuation of individual peaks. A common
post-mixing kernel cannot increase the original combined maximum. The overlap
ratio measures waveform superposition, not observed DOC source synchrony.

## Summaries and products

Within each geometry compare late-minus-early and common-minus-branch placement.
For the whole-network-derived cohort, give instances equal weight and use HUC4
block bootstrap (5000 draws); report all three existing classes and within-class
effects. For monitored footprints, average within receiver then give receivers
equal weight, with shared-monitoring-system bootstrap. Quantify pair coverage
and feasible intervention capacity so weak responses are interpretable.

Use the original geometry-selected shape representatives where eligible and
one extra case chosen by maximum arrival-CV times feasible variance capacity,
before computing outcomes. Produce real path maps, weighted source/combined
response plots, and bilingual distribution/overlap figures. Save replayable
tables, source manifests and an English research decision. Verify conservation,
fixed path means, equal variance or equal allocated budget, analytical moments,
finer-grid convergence, graph/source selection and figures. Run pytest, Ruff
and the historical audit. No neural training or edits to previous studies.
