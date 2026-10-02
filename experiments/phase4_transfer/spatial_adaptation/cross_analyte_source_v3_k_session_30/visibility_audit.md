# Cross-analyte visibility audit (v3 K-session)

## Policies

The same DOC outer split is evaluated under three explicit auxiliary-data
policies.  They are different scientific questions and are not silently mixed:

* `all_analyte_unmonitored`: no target-station pH or specific-conductance
  profile is available.  Distances use the label-free station descriptors.
* `doc_unmonitored_aux_observed`: pH and specific-conductance profiles at a
  target station are treated as independently observed auxiliary information.
  This is the policy implicit in the historical v1 experiment; it is not
  called leakage unless the study disallows auxiliary observations.
* `k_session`: only pH/EC cells in the same months as the first K DOC support
  cells are visible at a target station.  Unavailable profile dimensions are
  omitted pairwise from source distances; they are never encoded as zero.

The active rerun is `k_session`.  Source profile scaling is fitted on unique
training stations (`split.train // T`) only.  Outer-test stations are excluded
from internal source scaling and from neighbour candidates.

## Perturbation and matching contracts

`tests/test_cross_analyte_visibility_v3.py` checks that (i) K-session masks are
nested and expose only support months, (ii) an unobserved auxiliary cell cannot
be reopened by the support policy, and (iii) changing an unavailable profile
dimension cannot change a matched source station.  These are synthetic tests
and do not use WQP artifacts.

## Results

The validation grid selected `profile_k=0, source_k=160` (mean validation MAE
1.8083).  The corresponding outer result is mean MAE 2.5022 across seeds
42/43/44.  Thus, under validation-based selection, the legal K-session
auxiliary profile did not replace the no-auxiliary route.

The outer grid contains exploratory, unselected cells (for example K=1 with
40 source stations, mean MAE 2.4301).  They are retained for transparency but
are not treated as a selected test improvement because validation selected a
different cell.  The historical v1 selected setting (full target pH/EC
profiles, source-k=160) had outer mean MAE 2.5080; the difference is not an
apples-to-apples claim because v1 used a different profile policy and estimator
configuration.

## Counts and interpretation

On outer target stations, full pH/EC profiles would expose 6,367/7,125 cells.
The K-session policy exposes only 43/43 cells at K=1, 129/129 at K=3, and
214/215 at K=5 (pH/EC respectively).  Internal validation exposes 30/31,
91/92, and 151/152 cells.  This confirms that the K-session policy is a small,
same-month auxiliary channel rather than a full target profile.

The v2 all-hidden rerun should not be interpreted as evidence that auxiliary
analytes are ineffective: it zero-filled unavailable dimensions while still
using them in distances, which creates a missing-as-zero matching artifact.
