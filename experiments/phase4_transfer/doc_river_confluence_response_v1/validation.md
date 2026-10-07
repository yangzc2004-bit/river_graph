# Reproduction and visual checks

## Completed checks

- Full project tests: **1,234 passed, 2 skipped**. Eight existing warnings remain;
  no new failures. The twelve confluence-response tests cover waveform overlap,
  amplitude dominance, tied maxima, exact observation clocks, missing values,
  mixture normalization, lateral sampling and channel-survey grain.
- `ruff check .`: **passed**.
- Historical `audit_artifacts.py --verify`: **exit 0**. Historical parquet-only
  verification, the known excluded G0 conflict and zero-coverage kriging artifact
  retain their existing status.
- The new raw-file verifier: **passed** for seventeen downloaded public files,
  ten CSV tables, two parquet products, the summary and both figure receipts.
  It replays all twenty original water windows and all ten field campaigns.
- All six English/Chinese PNG figures were viewed. The final example panels and
  geometry bars have readable labels and clear title/legend spacing. The matching
  PDFs are exported from the same plots.

## Research scope

No training or model selection was performed. The original whole-network classes,
old predictions, paper endpoints and earlier river-form results are preserved.
Half-hour flow and sampled laboratory DOC retain their measurement resolutions;
neither is interpolated or filled. Eligibility concerns simultaneous observation
coverage, not DOC outcomes.

Twelve eligible water windows describe one mapped Krycklan junction. The other
mapped configurations remain in the coverage table. The five Tom's Creek
confluences are serial locations in one network; the four geometry correlations
are descriptive leads, not five-catchment inference. Salt-tracer flow estimates
were collected within three days after chemical sampling, rather than on a
synchronized DOC event clock.

## Reproduction

Run the analysis, plotting and verification scripts through the repository's
uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_confluence_response_v1.py
uv run python scripts/plot_doc_river_confluence_response_v1.py
uv run python scripts/verify_doc_river_confluence_response_v1.py
```

Source downloads can be reconstructed from the fetch entry point and retrieval
manifest. Large raw data remain under the gitignored `data/raw/` directory. The
committed analysis tables, observations and code snapshot support inspection and
comparison without overwriting previous experiments.
