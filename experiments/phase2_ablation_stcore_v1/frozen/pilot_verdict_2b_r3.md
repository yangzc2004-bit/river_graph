# Phase-2B pilot verdict (corrected, 2B-R1/R2/R3) — 2026-09-24

Status: **directional pilot only** (spec §7). This verdict supersedes the
first "2B passed -> 2C" call, which is **withdrawn**: its gate preview
excluded the strongest no-message control by naming and relied on tabular
results with validation-label leakage. Numbers below are recomputed by
`scripts/analyze_phase2b.py` (equal-mask within seed, then mean over seeds;
station-clustered bootstrap of the same estimand; seed signs are k of 3
training seeds). CI covering 0 = insufficient evidence, never equivalence.

## 1. Two comparison sets, and what each supports

**tabular set (H2X vs eco_RF / eco_MLP)** — supports only "better than these
tabular implementations":

| family | vs eco_RF | vs eco_MLP |
|---|---|---|
| E2a | **−29.8% MAE, A better 3/3, CI excludes 0** | −44.1%, A better 3/3, CI excludes 0 |
| E2b | **−37.0% MAE, A better 3/3, CI excludes 0** | −39.5%, A better 3/3, CI excludes 0 |
| E1 | **+6.1% worse, B better 0/3, CI excludes 0** | −2.0%, A better 3/3, marginal |
| E3 | −2.3% (insufficient evidence, 0/3) | insufficient (MLP unstable) |

**message_ablation set (H2X vs H2X_nomsg, matched inputs)** — the only
evidence about edge messages: **E1 +4.7% better (A better 3/3, CI excludes
0)**; E2a / E2b / E3 all **insufficient evidence** (CI covers 0). Correct
reading: edge messages are shown useful only for random-missing completion;
outside E1 the pilot provides **no evidence** either way — topology is not
shown necessary, and "no-message wins" is equally unsupported.

**encoder set (H2X vs H2E)**: insufficient evidence in all four families
(deltas −0.005 to −0.032, all CIs cover 0). The encoder is not established
as a contribution.

## 2. Gate-relevant facts (endpoints v1 unchanged; ambiguity noted)

- Primary gate vs best **tabular**: 2 of 3 families clear ≥10% (E2a, E2b).
  eco_MLP is not a trustworthy representative after honest validation
  (E3 MAE 4.69 with mask-spread 3.41, R² −16; early-stop best_epoch=2
  typical) — the tabular comparator that matters is **eco_RF**.
- E1 no-harm margin (≤5% vs best baseline): **breached** — H2X is 6.1%
  worse than eco_RF (stable: 0/3 seeds, CI excludes 0). This fact is
  unchanged by the R3 re-run.
- vs no-message: no support for a topology claim outside E1. The endpoints-v1
  phrase "best no-graph model" is ambiguous about the no-message control;
  both readings are reported here. Any formal gate redefinition requires
  `primary_endpoints.json` v2 with a revision reason and pilot-seen
  acknowledgement.

## 3. 2B pass/fail against its screening purpose

- No identity/coverage/NaN/leakage issues **after R2/R3** (audit: 144/144
  identity ok, v1 GNN + v2 tabular; label-perturbation regression green;
  early-stop traces recorded). The as-run leaky tabular batch is preserved
  under `predictions_v1_tabular_leaky/` and is not evidence.
- Ordering is interpretable, not "completely counterintuitive".
- At least one extrapolation family shows ecological input worth pursuing
  (E3: H2E/H2X beat ecology-free H2 by ~5–6%; E2a/E2b GNN-vs-RF gaps large).
- No endpoint was changed in response to results.
- eco_MLP remains unstable under honest validation — recorded as an
  implementation limitation, not a bug verdict and not an endpoint change.

**Verdict: 2B now passes as a corrected directional pilot.** The earlier
"proceed to the 240-config 2C" recommendation is **withdrawn**; see §4.

## 4. Remaining questions and 2C scope recommendation

What the corrected pilot leaves genuinely open:

1. Does the E2a/E2b GNN-over-RF gap persist at 5 seeds? (already 3/3 seeds,
   CI excludes 0 — marginal confirmation value only)
2. Does H2X beat no-message anywhere outside E1? (currently no evidence; a
   tighter CI at 5 seeds would mostly re-measure a ~0 delta)
3. Does the E1 no-harm breach flip? (stable 0/3, CI excludes 0 — unlikely)

None of the three justifies the original 240-config matrix. Recommendation:
**do not launch the 240-config 2C**. Either (a) a small targeted confirmation
(only the decisive pairs H2X vs eco_RF vs H2X_nomsg on the 8 key masks at 2
extra seeds), or (b) close Phase 2 with this verdict and carry **eco_RF,
H2X and no-message all forward as eligible tools** into the ecological
blind-spot / budgeted monitoring-design main line (Phases 3–5), where the
remaining scientific questions live. The choice between (a) and (b) is a
research-route decision and is not made here.

High-DOC note (unchanged rule): E2a/E2b Q95 has **2 unique true-high cells**
(all arms mostly undetected) and Q90 has 12/7 — reported as "undetected /
estimate unstable" with denominators; seeds re-predict the same cells and do
not increase ecological sample size.
