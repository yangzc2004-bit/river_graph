# Branch arrival and common-trunk storage at a fixed mean arrival time

## Scientific question

How do independent tributary paths and a shared downstream storage response
change the height and duration of a DOC pulse? This controlled follow-up uses
the 59 measured station-to-station river footprints, 22 receivers and 11
monitoring systems already retained in the river-form study. It follows the
sampling-resolution findings; it is an exploratory mechanism experiment.

## Controls and structural operations

Both tributaries receive the same unit-height Gaussian concentration anomaly,
centred at relative time zero. Keep the existing drainage-area proxy shares
fixed and assume constant positive discharge. Their shares sum to one. No
landscape source contrast, chemical loss, extra lateral source or model fit
is introduced. Use pulse SDs 0.075, 0.15 and 0.30; report every duration. The
middle duration matches the preceding controlled experiment.

Let measured independent branches be B_a and B_b, the shared trunk be C, and
M = w B_a + (1-w) B_b + C. Normalize these lengths by M, corresponding to the
same-velocity conceptual routing experiment. All complete responses have
weighted mean arrival one; relative time is not measured days.

Cross two branch conditions with four storage conditions:

- Actual independent branch delays versus equal branch delays at their existing
  weighted mean. The common trunk and mixture shares stay unchanged.
- Allocate fractions f = 0, 0.25, 0.5 and 1 of the common mean-time budget c=C/M
  to a causal linear-reservoir response k(u)=exp(-u/tau)/tau for u>=0,
  tau=f*c. Its remaining deterministic translation is c-tau. The reservoir
  mean plus translation is always c. f=0 is pure translation, not storage.

This is a redistribution of a fixed time budget, not adding a reservoir to a
previously calibrated real river. Mapped lake/reservoir segments are recorded
separately and never used to infer f or residence time. Both branches traverse
the same common-trunk response after mixing. The kernel is nonnegative and
unit gain: constant concentration and integrated anomaly are preserved at
constant flow, although peak concentration can fall.

## Measures and exact identities

Record peak height, peak time, pulse centroid, SD, t10/t50/t90, central 80%
duration and integrated anomaly fraction. Centroid and variance also have
closed-form checks:

    centroid = 1
    variance = input_SD^2 + w(1-w)*(b_a-b_b)^2 + tau^2

Equal branches remove only the branch-variance term. Compare actual versus
equal branches within the same f, and storage versus pure translation within
the same branch condition. Report their interaction without assuming that
percentage peak reductions add. Strongly split peaks can affect percentile
duration differently from SD. Repeat the complete grid at half the time step
to establish numerical convergence.

## Summaries and figures

Each pair is a deterministic geometry scenario, not a new field observation.
Average pairs within receiver, then give the 22 receivers equal weight for
cohort summaries; also retain the distribution over all 59 connections. Do
not attach population confidence intervals to chosen storage parameters.
Retain the three original whole-network outline classes for interpretation.
Their monitored footprint counts are uneven; do not use these examples to
claim a validated class ranking or invent a new morphology taxonomy.

Reuse the four geometry-selected river examples for response curves and one
additional map selected solely for the largest mapped common-storage fraction.
Produce English and Chinese scientific figures, replayable scenario tables
and a research decision relating these mechanisms to the observed sampling
clock. State the directed temporal graph operation suggested by the findings
and the event observations needed to estimate actual hydraulic parameters.

## Process basis

The separation of translation and temporary-storage attenuation is motivated
by the USACE HEC-HMS Clark routing description. Its runoff model is not a DOC
calibration: the concentration routing experiment here uses a separately
derived unit-gain conservative-anomaly response under constant discharge.

https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/clark-unit-hydrograph-model

## Execution

No neural training and no edits to prior studies. Run the new analysis and
plot scripts through the existing uv environment, inspect both figure languages,
replay the results, and run pytest, Ruff and the historical artifact audit.
