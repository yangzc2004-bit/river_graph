# Controlled DOC routing on real river forms

This study isolates how path spread, tributary balance, arrival phase and shared
channel exposure reshape an identical input signal. Read
[`research_decision.md`](research_decision.md) for results and the next experiment.

## Reproduce

From the repository root, in the existing uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_routing_mechanisms_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_routing_mechanisms_v1.py
uv run python scripts/plot_doc_river_routing_mechanisms_v1.py --chinese
uv run python scripts/verify_doc_river_routing_mechanisms_v1.py
```

Inputs are the preceding morphology cohort, fixed environmental pairs,
location-screened tributary inventory, fixed geometry representatives, cached
directed path distances, VAA incremental areas/reach lengths and real flowline
geometry. Local raw caches are not committed. `analysis_sources.json` identifies
the execution inputs; the analysis source snapshot is retained.

## Read the figures

1. `real_forms_and_routed_signals`: real geometry, within-network path controls,
   all-network peak responses, fixed matched pairs and frequency response.
2. `junction_process_experiment`: fixed total paths, changing branch/shared
   partition under conservative speed and imposed processing scenarios.
3. `branch_balance_and_arrival_phase`: fixed-input branch weights and arrival
   timing; aligning arrivals restores the pulse without creating extra load.

Mapped river geometry is data. All routed pulses, imposed velocities and process
rates are scenarios; time units are not measured days/months. They are separate
from neural predictions, field DOC losses and external basin validation.
