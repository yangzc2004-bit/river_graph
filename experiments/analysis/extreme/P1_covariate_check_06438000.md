# Covariate check: 06438000

Follow-up to `P0_label_audit_06438000.md`: do routine co-measured water
quality parameters carry the 1976-82 high-DOC regime signal?

| parameter | 1976–82 (regime) | other years | verdict |
|---|---|---|---|
| pH | 8.11 (n=102) | 7.94 (n=379) | +0.17 — noise |
| Specific conductance (µS/cm) | 2254 (n=105) | 2161 (n=610) | +4% — negligible |
| Water temperature (°C) | 12.6 (n=116) | 11.3 (n=501) | slight, consistent with seasonal sampling bias |
| Discharge | see P0: Spearman(DOC,Q)≈0 | | no signal |

**Conclusion:** none of the routine co-measured covariates (Q, pH, SC,
temperature) detect the 1976-82 high-DOC episode. The regime is invisible
to every proxy available at monthly resolution in our dataset — the
unpredictability is an *information* problem, not a model-capacity
problem. Candidate signals that remain untested (and out of MVP scope):
historical land-use change (pre-2001 LULC), reservoir/wetland operations,
and analytical method-era artifacts (method metadata is absent for these
samples).

Implication for the paper: extreme regime episodes like this should be
addressed via uncertainty quantification / honest limitation, not by
chasing point-prediction accuracy.
