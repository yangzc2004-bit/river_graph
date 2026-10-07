# Internal structure, routing and observed DOC

This study keeps the existing three river outline classes and examines their
internal organization. All 322 classified real networks are assigned using
junction balance and relative path dispersion before DOC records are attached.
Read `research_decision.md` for findings and the next scientific step.

## Reproduce

Run from the repository root with the uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_internal_structure_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_internal_structure_v1.py
uv run python scripts/plot_doc_river_internal_structure_v1.py --chinese
uv run python scripts/verify_doc_river_internal_structure_v1.py
uv run pytest
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

## Inspect

- `analysis/network_profiles.csv`: one row per distinct classified receiving
  COMID; original outline, junction balance, path CV and internal profile.
- `analysis/outline_profile_counts.csv`: original outline by internal profile.
- `analysis/representatives.csv`: four geometry-only real map examples.
- `analysis/normalized_routing.csv`: 322 networks × three path scenarios,
  identical forcing and fixed mean relative delay.
- `analysis/representative_pulses.parquet`: replayable synthetic pulse curves.
- `analysis/observed_receivers.csv`: all 22 receivers from the prior signal study,
  retaining their 59 tributary connections and 11 monitoring systems.
- `analysis/observed_buffer_summary.csv`, `observed_profile_contrasts.csv`:
  receiver-equal summaries and exploratory contrasts with system bootstrap.
- `analysis/whole_pair_alignment.csv`: whole-network vs gauged-pair descriptors.
- `analysis/continuous_observed_associations.csv`, `association_omitted_systems.csv`:
  four exploratory associations after the initial profile results, with area,
  coverage and the other structural axis included.
- `analysis/fixed_reference_*`: additional cut points and unchanged observation
  population; this sensitivity does not replace the primary geometry medians.
- `analysis/mapped_flow_responses.csv`, `descriptive_flow_summary.csv`: context
  from the existing 205-station flow study, without refitting those responses.
- `figures/*[_cn].{png,pdf}`: outlines, actual river maps, controlled/observed
  buffering and scale alignment, in English and Chinese.

The four profiles are relative combinations of two continuous structural axes,
not four newly discovered natural clusters. Two junction-free networks remain
separate. Peak damping in the pulse experiment preserves the total DOC anomaly;
relative delay is a scenario coordinate rather than measured travel time. The
observed DOC comparison reuses the previous exploratory population and is not
an independent or external validation. No neural model is trained here.
