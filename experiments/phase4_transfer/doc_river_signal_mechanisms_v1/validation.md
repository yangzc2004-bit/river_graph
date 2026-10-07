# Validation

- Full suite: **1,125 passed, 2 skipped**, eight existing warnings (43.02 s).
- Five new tests cover exact variance decomposition, equal-amplitude scenarios,
  common-calendar mixing linearity, outlet-independent arrival inputs,
  zero-opportunity paths/parallel changes, missing/nonpositive flow and
  degenerate signals.
- `ruff check .`: all checks passed.
- Historical `audit_artifacts.py --verify`: exit 0. The known overwritten G0
  artifact is still excluded; the zero-coverage kriging case remains reported.
  Historical artifacts retain their existing missing-sidecar qualifications.
- Dedicated replay: **passed**. Nine tables, three observed diagnostic products,
  59 calendar fits, 5,000 system bootstrap draws and figure manifests reproduce.
- Current/previous source DOC and outlet DOC match original dataset cells,
  source-role permissions and calendar dates. The mean-delay predictions match
  saved predictions without fitting another model.
- Perturbing scored outlet DOC does not change source anomalies, mixture or
  arrival inputs. DOC perturbation does not change measured-flow availability;
  future hydro changes do not change earlier flow screens.
- Single-system elongated and sparse class intervals are unavailable for both
  system and HUC4 summaries. This correction prevents geographically split
  receivers inside one shared network being treated as independent class data.
- Native outlet variability and arithmetic variance summaries were examined
  together. No unusual concentration or high outlet/source ratio was deleted;
  log-scale and system-influence diagnostics were explicitly added after the
  initial native summaries and retain their supplementary status.
- Both English and Chinese figures were opened and inspected. Legends, units,
  scenarios, sample denominators and explanatory notes remain visible. Bubble
  plot limits include the full negative-correlation observations.

This verifies the diagnostic calculations and saved lineage. It does not turn
an area-weighted mixture into a complete field mass budget, an interpolation
fraction into a travel-time measurement, or these reused observations into
external validation. No previous model or experiment was overwritten.
