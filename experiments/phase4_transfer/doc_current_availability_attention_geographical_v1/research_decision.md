# Current source availability: geographical replication

## Research decision

Using observed current source DOC without requiring an earlier-year source
observation improves the matched full-source model in geographical withholding.
Keep the evaluated current-availability procedure intact as a confirmed
candidate. Continue source-only performance development; a wider ecological
candidate study was designed before inspecting these geographical results.

The working target of a 5% K0 improvement over both the retained complete model
and strong trees has not yet been reached. The candidate is also not established
as superior to the preceding anomaly-only attention across all outcomes. Keep
that complete version and these results as separate procedures, without mixing
the winners at different regions or K values. Existing portable/external scores
remain unchanged.

## Experiment and execution

All 25 fixed HUC4/seed packages completed: 1013, 1019, 0708, 1030, 1101 ×
42–46, 50 neural fits, 250 reused double-held reference fits and no new forest.
The source-role study selected the mechanism and settings before this fitting.
Receiving K0 uses ecology, hydrology, season and source experience, with no
local DOC, pH or conductivity. The model still has 37,900 parameters, 20
ecological candidates, the original daily-flow keys and matched aggregate
inputs, a 12-month GRU query, 30 epochs and patience 5.

Every saved candidate, fold exclusion, initial state, full-grid component,
ecological fusion, support curve and diagnostic replayed bitwise. Point states
and validation selection were saved before opening scoring/support labels.
The analysis uses 5,000 paired station draws, seed means within region and
equal weights across the five regions. The final PNG was inspected. A plot
filename inherited from the preceding study was corrected in the plotter;
prediction, fitting and numerical analysis were unchanged.

This is **retrospective geographical role withholding inside ST357**, not an
independent external-basin evaluation. These geographical tasks have been
evaluated for earlier fixed versions. No region or support-count result chose
this candidate's mechanism or settings.

## Primary all-observation K0 result

| Complete procedure | Equal-region MAE, mg/L | Station-equal MAE | Q90 MAE, mg/L |
|---|---:|---:|---:|
| Older complete model | 2.360283 | see analysis | 10.632267 |
| Strong station-hidden trees | 2.344885 | see analysis | 10.520142 |
| Aggregate-source trees | 2.314529 | see analysis | 10.481605 |
| Retained complete | 2.282266 | 2.504735 | 10.519953 |
| Matched full-source attention | 2.266236 | 2.492171 | 10.431318 |
| Matched anomaly-only attention | 2.258278 | 2.486052 | 10.409544 |
| Expanded seasonal values only | 2.270518 | see analysis | 10.480669 |
| **Actual current source availability** | **2.248532** | **2.476483** | **10.439944** |

The isolated availability gain versus matched full-source attention is
**0.781% [0.282%, 1.341%]**, with four of five regions improving. Actual
current values beat individual seasonal means under the same expanded
validity by **0.968% [0.455%, 1.547%]**, all five regions improving. Q90 also
improves against this seasonal control by 0.389% [0.028%, 0.706%]. The
seasonal control retains the original matched aggregate descriptors, so it
does not remove every form of current source information.

Accumulated gain versus the retained complete model is
**1.478% [0.416%, 2.672%]**, four regions positive. Against strong station-hidden
trees it is **4.109% [1.113%, 7.109%]**, all five regions positive. Against
the older complete model it is 4.735% [2.141%, 7.548%]. These comparisons
include changes introduced before the current availability experiment.
Against aggregate-source trees, the 2.851% gain has an interval crossing zero.

Compared with the preceding anomaly-only attention, K0 improves 0.432%
[-0.259%, 1.271%], four regions positive: a lower point estimate, but no
established incremental superiority. Complete Q90 is 0.292% worse than that
version, with an interval crossing zero. Q90 versus matched full-source is
also unresolved (-0.083% gain); versus retained complete, Q90 improves
0.761% [0.112%, 1.438%], all five regions positive. Report these distinct
comparisons alongside the overall MAE result.

The native-only candidate has MAE 2.246979. Its isolated gain versus native
matched full-source is 0.816% [0.266%, 1.408%], four regions positive,
confirming that the availability increment exists before fusion. Native-only
gain versus the original native residual is 0.822% [-0.031%, 1.779%]. Native
Q90 is 0.438% worse than anomaly-only native attention, with the interval
supporting that loss. Do not substitute the native result for the complete
procedure or hide this tail trade-off.

## Regional heterogeneity and remaining error

| Held-out HUC4 | Current-availability complete MAE |
|---|---:|
| 1013 | 5.583000 |
| 1019 | 1.635860 |
| 0708 | 1.683737 |
| 1030 | 1.258189 |
| 1101 | 1.081874 |

There are 7,262 unique station-months at 104 stations in the primary task.
61/104 stations improve versus retained complete, 65/104 versus matched
full-source and 71/104 versus strong trees. Station-equal MAE is a separate
estimate, not a replacement endpoint. Overall signed bias is -0.720759 mg/L
versus -0.649449 for retained complete; lower MAE has not removed systematic
underprediction. The large 1013 error remains a priority for source-role
research rather than adjustment to its scored labels.

Q90 uses source-training thresholds. Unique tail counts are 904, 43, 66, 15
and 11 for 1013/1019/0708/1030/1101. Five seeds repeat the same cells. Mean
recall in 1013 is approximately 0.983; it remains zero in the other four
regions. The last two tails are unstable (n<20). An average Q90 MAE benefit
does not establish successful rare-event detection everywhere.

## Fixed-query adaptation

| K | Current availability | Retained complete | Anomaly-only attention |
|---|---:|---:|---:|
| 0 | 2.261259 | 2.296495 | 2.271570 |
| 1 | 2.149632 | 2.160459 | 2.148383 |
| 3 | 1.982968 | 1.987946 | 1.966872 |
| 5 | 1.893004 | 1.909790 | 1.887012 |

These 6,742 query cells always exclude all five possible support cells;
their K0 is distinct from the primary 7,262-cell K0. K5 gain versus retained
complete is 0.879% [0.181%, 1.561%], four regions positive. K1/K3 gains
versus retained complete are uncertain. Against anomaly-only attention, K3
is 0.818% worse [-1.675%, -0.069% gain]; K5 is 0.318% worse, uncertain.
Keep the single fixed procedure for the entire curve.

## What expanded access supplies

Mean current donor counts in 1013/1019/0708/1030/1101 are
4.478/1.615/4.156/2.396/2.003, versus matched counts
2.748/1.324/3.687/2.277/1.743. The fraction with at least one source donor
rises from 65.5% to 86.7% in 1013 and from 57.6% to 62.6% in 1019.
There are 255/108/12/32/75 newly supported cells across the five regions;
seeds do not multiply these counts. Mean actual-source prior mass is 0.431
and raw Shannon entropy 0.770; these are allocation diagnostics, not physical
transport or causal coefficients. The ecology/hydro/input strata remain in
`analysis/strata_summary.csv` with all regions retained.

## Continuation and reproduction

Continue `doc_wide_source_attention_v1`, already designed from source-role
support sparsity before these results were inspected. Increase only the
individual ecological pool from 20 to 60, keeping parameter count and prior
scale fixed. Source validation determines whether this new mechanism merits
its own fixed replication. No geographically selected regional or K mixture.

Reproduce with `scripts/analyze_doc_current_availability_attention_geographical_v1.py
--bootstrap-draws 5000` and its plotter. Exact replay is included. Fitting-version
validation: 1,002 tests passed, two skipped, Ruff and historical audit passed.
The subsequent wider-source implementation passed 1,005 tests, two skipped,
Ruff and audit without changing the fitted model files above. Related small
artifacts are saved in the existing pending-submission whitelist; large
fitting caches stay local under the documented Git write limitation.
