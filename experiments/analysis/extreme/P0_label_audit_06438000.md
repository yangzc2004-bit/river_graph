# Label audit: 06438000

## Summary

- Observed months: **28** only (sparse; gap 1981-04 → 1987-04).
- DOC: mean=58.2, **median=6.8**, max=460. Mean is dominated by 1978–1980.
- **DOC>50: 6 months, all in 1978–1980.** Peak 460 (1980-07), 450 (1980-10),
  212 (1978-08), 160 (1979-07). 1981+ max is 32 (1981-07); 1987–88 are single-digit.
- Two eras:
  - **1978–1980** (n=11): mean DOC **137.5**, median 53
  - **other years** (n=17): mean DOC **6.9**, median 4.6
- **Spearman(DOC, Q) = −0.00** (p=0.99). Peak DOC is **not** peak Q
  (1980-07 Q≈206 ≈ 50th pct; 1980-10 DOC=450 at low Q≈60).
- 1980-06 had high Q (≈388) but **no DOC sample**.
- Upstream `06437000`: DOC always low (max 6.7, mean 4.7). No co-observed
  high-DOC months with this site. Downstream graph neighbor: `06440000`.
- Calendar months of high DOC: spread (1,4,7,8,10) — not a single season lock.

## Interpretation

**Not a single grab-sample typo** (multiple high values over ~3 years), but also
**not evidence of same-month flood flushing** at monthly resolution.

More consistent with:

1. A **regime / source episode** in 1978–1980 (local wetland/soil/land-use
   or sampling-era change) that ended before the 1987–88 record,
2. Undersampled events (high-Q June 1980 unobserved for DOC),
3. Possibly method/era effects in late-1970s samples.

Upstream stays clean → high DOC is **local to this reach/catchment**, not
propagated from the monitored mainstem neighbor.

## Implication for modeling (H2 / H2X / seed43)

- Labels are real observations in E3 seed43 test and dominate test SS_tot.
- With **monthly-mean Q only** and **no event/temporal/source-history state**,
  a spatial GNN has little basis to emit 460. **Systematic underprediction is
  expected and scientifically interpretable**, not a pure engineering failure.
- Paper wording should be: *consistent with a historically localized high-DOC
  regime that monthly-mean discharge does not encode* — **not** “proves flushing”.

## Data files

- Full series: `case_06438000_series.csv`
- This note: `P0_label_audit_06438000.md`
