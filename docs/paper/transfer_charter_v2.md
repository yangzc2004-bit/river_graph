# Cross-basin multi-analyte transfer charter (v2)

Status: **versioned route freeze, 2026-09-24**. This charter defines a new
route after the target-unseen K=0 scale blocker was identified. It does not
modify `transfer_charter_v1.md`, the v1 endpoints, or any Phase 0--3 frozen
artifact. No result from the earlier exploratory route is promoted to a
confirmatory claim.

## Why this version exists

The original leave-one-analyte route hid every target-analyte label outside
the K support cells. At K=0, source tasks did not identify a legal native-unit
output scale for an unseen analyte. A target mean, variance, or inverse
mapping would have been an unregistered target prior. The blocker is recorded
in `experiments/phase4_transfer/target_unseen_k0_decision.md`.

The Stage-2A same-analyte diagnostic was already exposed before its protocol
corrections. Its v2 result is reproducible but descriptive only: it uses
privileged same-analyte source labels, is not held-out-analyte evidence, and
does not unlock model development. This prior exposure is acknowledged here
and is not treated as a preregistered result.

## Main idea

**Ecology-aware cross-basin transfer for reconstructing sparse water-quality
observations across analytes.**

The target analyte is observed in source HUC6 basins, which makes its native
output scale identifiable at K=0. A target HUC6 basin is then held out, and
only K target-analyte support cells are revealed during adaptation. The study
asks how much shared ecology--time information and local support transfer to a
new basin, and how that value changes across DOC, pH, and specific
conductance.

Target venue: *Patterns*.

## Scientific question

**When a target analyte is observed in source basins but a target HUC6 basin
is missing most of its observations, can a shared ecology--time
representation reconstruct that analyte in the held-out basin, and how much
do K local observations add?**

## Scope frozen for this route

- Analytes: DOC, pH, and specific conductance.
- Primary cohort: ST357 with the five existing HUC6 leave-out tasks
  `{103001, 510020, 102701, 101302, 101900}`; `510020` retains its canonical
  `051002` alias.
- Primary task: same-month target-HUC6 leave-out with a fixed query and nested
  support sets. This is a spatial cross-basin task, not a future-month claim.
- K ladder: `{0, 1, 3, 5}`. K=10 remains exploratory and cannot enter a
  primary endpoint.
- Visibility: the target analyte may be used for fitting and validation in
  source HUC6 basins. Every target-HUC6 outcome label is hidden from fitting,
  early stopping, normalization, threshold selection, and model selection;
  only K target-analyte support cells are available during target adaptation.
  Query labels are evaluation-only.
- Inputs in the target HUC6 may include the frozen ecological, hydroclimatic,
  temporal, and graph features. Outcome labels from the target HUC6 are not
  inputs outside the declared K support cells.
- The independent external basin remains future work. It is not selected,
  tuned, or used to change endpoints in this version.

## Claims this route may test

1. Ecological information has missingness- and basin-dependent predictive
   value.
2. A shared ecology--time representation can transfer a target analyte from
   source HUC6 basins to some held-out HUC6 basins.
3. A small target-analyte support set can improve reconstruction, with gains
   that depend on analyte, basin, and K.
4. Matched no-graph and no-ecology controls are required before assigning an
   incremental value to topology or ecological features.
5. Empirical uncertainty calibration has to be reported jointly with interval
   width; coverage alone is not a success claim.

## Non-claims

- No causal, mechanistic, or universal interpretation.
- No claim that every analyte, HUC6, external basin, or missingness regime
  transfers.
- No target-unseen-analyte K=0 claim under this route; that is a separate
  blocked protocol and remains documented as such.
- No stable ecological blind-spot, active-sampling, or management claim
  unless a new ranking gate is passed by at least two independent tools or
  indicators.
- The failed `support_encoder_v2` result and the corrected Stage-2A
  same-analyte diagnostic are not positive evidence for this route.

## Stage gates

| stage | gate | failure action |
|---|---|---|
| 0 | this charter, v2 endpoints, label roles, and task construction are frozen | no transfer training; issue a new version with a recorded reason |
| 1 | ST357 three-analyte provenance and primary HUC6 task data pass; external case remains separately pending | continue ST357 internal work only, or return to the DOC manuscript |
| 2 | complete baselines and information decomposition run under the v2 visibility rules; at least two analytes show measurable support information | stop architecture expansion and report the information decomposition |
| 3 | feasibility: K=5 improves the best source-only-selected simple baseline by >=10% for >=2/3 analytes and >=3/5 target HUC6 tasks, with no >5% primary harm | stop new architecture development and retain a bounded transfer result |
| 4 | five-seed ST357 confirmation reproduces the feasibility direction without a single-task explanation | downgrade the transfer claim or return to the DOC route |
| 5 | a future independently selected external basin reproduces DOC and at least one secondary analyte without retuning | restrict the paper to ST357 internal transfer and report the boundary |
| 6 | uncertainty is reported with width and passes no unsupported ranking claim | keep uncertainty as a calibration diagnostic |

Every gate must pass before the next stage is authorized. The existing
DOC-only reconstruction and uncertainty work remains the fallback manuscript
if the internal transfer gate fails.

