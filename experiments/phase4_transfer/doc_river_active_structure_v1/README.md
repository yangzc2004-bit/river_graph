# Observed water participation and DOC mixing

This study links fixed mapped river paths to changes in measured water
contributions and DOC mixing. Read `research_decision.md` for the full result
and `manuscript_section.md` for a paper-ready subsection. The frozen study
design is `study_plan.md`.

## Reproduction

Run from the repository root using its uv-managed environment. The ST357 dataset
and the existing source-cell, monthly-DOC and pathway products must be present.

```bash
uv run python scripts/analyze_doc_river_active_structure_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_active_structure_v1.py
uv run python scripts/diagnose_doc_river_active_structure_v1.py
uv run python scripts/plot_doc_river_active_structure_v1.py
uv run python scripts/plot_doc_river_active_structure_v1.py --chinese
```

Primary analyses use fixed observed frontiers and only permitted source-role DOC.
The coverage and leave-system-out diagnostics were added after primary results
were inspected and retain their original estimates. No new predictor training
is required.

## Outputs

- `analysis/`: monthly participation, source flow weights, eligibility ledger,
  flow references, network contrasts and 5,000-draw system summaries.
- `coverage_sensitivity.csv` and `leave_system_out.csv`: supplementary checks.
- `figures/`: three figure families, each in English and Chinese PNG/PDF.
- `analysis_sources.json`, `diagnostic_sources.json`, `figure_sources*.json`:
  recorded inputs and output identities.
- `verification.json` and `quality_checks.json`: observed validation results.

The eligible hydro population contains 17 networks and 916 network-months;
1,021 complete-flow rows in the full output also include ineligible networks.
The DOC-supported population contains 531 network-months. No sparse-form network
has sufficient records for the flow-state contrast. The positive participation
mean does not persist in the four high-area-coverage networks, and the direct
broad-versus-elongated DOC contrast remains unresolved.
