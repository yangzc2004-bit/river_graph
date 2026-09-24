# Transfer paper charter (v1)

Status: **Phase-0 freeze, 2026-09-24**. This charter starts a new paper route
without modifying the frozen DOC-only Phase-0--3 artifacts. It authorizes data
audits and protocol tests; it does not authorize model training before the
Stage-1 data gate is passed.

## Main idea

Study which ecological, temporal, local-support, and river-network signals
remain useful when water-quality observations are missing across analytes and
river basins. The model is an instrument for measuring information transfer,
not the paper contribution by itself.

## Paper question

**When monitoring is sparse, can a shared ecology--time representation use a
few observations of a target analyte to reconstruct its missing values in a
new river basin?**

Target venue: *Patterns*.

## Scope frozen before new training

- Targets: DOC, pH, and specific conductance.
- Primary cohort: the existing ST357 graph and five HUC6 leave-out tasks
  `{103001, 510020, 102701, 101302, 101900}`.
- External case: one independently selected basin, chosen by the data gate in
  `scripts/audit_transfer_data_gate.py` before model results are inspected.
- K ladder: `{0, 1, 3, 5}`. K=10 is exploratory only.
- Transfer definition: multi-task pretraining with one target analyte held out
  during evaluation, followed by target-analyte adaptation using only K support
  cells. Target-basin labels are also held out during adaptation.
- External data are a second complete case. They are never used for model
  selection or endpoint changes.

## Claims

1. The value of ecological information depends on the missingness regime.
2. A shared ecology--time representation can transfer across some analytes and
   river basins, with analyte- and basin-dependent gains.
3. A small target-analyte support set can improve reconstruction when the
   transfer task passes the preregistered K=5 gate.
4. Empirical uncertainty calibration must be reported together with interval
   width; coverage alone is not a success criterion.
5. Topology is a tested information source, not a guaranteed contribution.

## Non-claims

- No causal, mechanistic, or universal interpretation.
- No claim that every analyte, basin, or missingness regime transfers.
- No stable blind-spot, active-sampling, or management claim unless a later
  preregistered ranking gate is passed by at least two independent tools or
  indicators.
- The failed `support_encoder_v2` result remains a failure record and is not
  presented as positive transfer evidence.

## Stage gates

| stage | gate | failure action |
|---|---|---|
| 0 | charter, endpoints, roles, and external selection rule are frozen | no training; revise a new version |
| 1 | DOC/pH/conductance data and one external case pass the data gate | use ST357-only scope or return to DOC paper |
| 2 | baselines show measurable support signal for at least two analytes | stop transfer expansion and use DOC route |
| 3 | K=5 improves the best simple baseline by >=10% for >=2/3 analytes and >=3/5 HUC6, with no >5% primary harm | stop architecture expansion; retain diagnostic result |
| 4 | five-seed ST357 confirmation reproduces the pilot direction | downgrade to exploratory transfer if absent |
| 5 | DOC and one secondary analyte replicate direction in the external case | external claim becomes a boundary condition |
| 6 | calibration is reported with width and no unsupported ranking claim | uncertainty remains secondary |

Every stage has to pass before the next stage starts. The current DOC-only
results remain the fallback manuscript if the transfer gate fails.
