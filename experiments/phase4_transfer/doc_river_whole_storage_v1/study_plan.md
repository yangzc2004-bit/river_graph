# Whole river form, path dispersion and mapped storage

## Question and retained design

Extend the branch/storage experiment to all 297 station-network instances
(295 receiving reaches) in the existing morphology/DOC panel. Retain the
original elongated/tributary-rich, mainstem-dominated/sparse, and
broad/tributary-rich classes. Neither DOC nor the simulated response selects
classes, representatives, matching pairs or storage parameters.

Every positive-area incremental catchment supplies the same unit-height DOC
anomaly. Its constant flow share is proportional to incremental area. Use
the existing shortest directed midpoint-to-outlet paths, including secondary
links, and normalize channel length by square-root basin area. These units
are not measured days. Reconstruct one downstream successor consistent with
each saved distance; prefer the primary link only when shortest paths tie.

## Structural operation

Use only mapped LakePond/Reservoir reaches as storage opportunities. Consecutive
reaches with the same positive NHD waterbody COMID on a selected path form ONE
storage element. Entering a waterbody at different points changes the traversed
length. Separate waterbodies remain serial elements. Thus subdividing a single
physical waterbody does not create extra reservoirs. Audit missing IDs and
require exact agreement with the original saved route distances.

For each contiguous waterbody segment of normalized traversed length l,
allocate f*l of its existing mean-time budget to a causal unit-gain exponential
kernel and retain (1-f)*l as deterministic translation. Report f=0, 0.25, 0.5,
1 and Gaussian input SD=0.075, 0.15, 0.30. This is a fixed-mean shape experiment,
not an estimate of actual residence time or a calibrated reservoir addition.
No source-cover contrast, DOC chemical loss or hydraulic flow splitting is used.

For source p, mean delay d_p is unchanged and additional within-path variance
is f^2 * sum(l_pg^2), with g indexing contiguous physical waterbodies. Exact
mixture moments are:

    mean = sum(w_p*d_p)
    variance = input_SD^2 + Var_w(d_p) + f^2*sum(w_p*sum(l_pg^2))

Compute the full response by summing exact serial exponential characteristic
functions, then inverse Fourier transform with the analytic Gaussian forcing.
Use a long tail domain and negligible forcing-frequency truncation; verify
moments, nonnegative response, unit gain and finer-grid convergence. Save
representative traces, all scenario metrics and path/storage descriptors.

## Comparisons

Report baseline peaks and durations, within-network storage changes, weighted
storage exposure, distinct serial storage variance, and within-class ranges.
Use equal station-network weights and HUC4-block bootstrap for descriptive
class contrasts. Retain the original covariate-selected elongated/broad pairs;
do not rematch on new outcomes. Repeat the moderate storage comparison with
five uniform input locations per reach for the complete cohort, because sparse
networks were sensitive to source placement in the preceding experiment.

Between-class means retain the previous basin-size normalization. Only the
within-network storage comparison holds average arrival fixed. A class label
does not by itself identify storage opportunity or its response. Distinguish
the three classes' between-group signal from their within-group heterogeneity.

## Deliverables

English and Chinese figures with the fixed real mapped representatives;
replayable scenario and matched-pair tables; an English research decision;
focused physical-identity tests, pytest, Ruff and historical artifact audit.
Old analyses remain untouched. No neural training is required. The output
answers how real geometry can reshape an identical DOC input, while observed
DOC event data remain insufficient to calibrate physical storage times.
