# Availability-preserving DOC model: products and inference

This experiment reuses the nine saved monthly and daily encoder/GRU checkpoint
pairs from `doc_daily_hydro_residual_v1`. It introduces no new neural weights.
The reloadable `DailyHydroRouter` chooses the daily native prediction if any
numerical daily-discharge descriptor is valid and the monthly prediction if
all three validity flags are zero. Valid constant and zero flow use daily.

Each package saves:

- `router.json`: fixed information-availability rule, independent of DOC labels;
- `adapters.json`: direct, source-validation-selected support calibration;
- `mixers.json`: source-validation-selected ecological/support integration;
- `source_validation.csv`: native and integrated K0 errors by availability;
- `full_grid.parquet`: 233,478 cells with monthly, daily and routed native bases,
  context, ecological memory, route indicators and integrated K0 predictions;
- `predictions.parquet`: unchanged target query cells at K=0/1/3/5, with two
  support representations and twenty model products;
- sidecars and completion records linking the saved parent models, feature
  pack, datasets, masks and current execution sources.

The native route precedes downstream calibration. Final integrated or adapted
predictions can differ from monthly predictions on cells routed to monthly:
the hybrid model has its own validation-selected mixture/support parameters.
The route protects native information availability; it does not impose a final
output equality constraint after support adaptation.

Monthly/daily controls are reprocessed with identical validation fits and must
reproduce their parent products exactly. The six earlier reference products
are copied unchanged. Three bases plus context, two support representations,
three integrated bases and six references give twenty products at each K;
this is not twenty new neural fits. Positive K uses retrospective support.

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-fallback-v1
uv run python scripts/verify_doc_daily_hydro_fallback_v1.py
uv run python scripts/analyze_doc_daily_hydro_fallback_v1.py
```

At inference, load the linked monthly/daily checkpoint pair, compute its native
predictions using the frozen feature definitions, then pass aligned predictions
and the canonical eight-channel daily feature block to
`DailyHydroRouter.from_dict(router_state).predict(...)`. Apply the saved direct
support adapter or integrated mixer for the chosen K and support representation.
The independent verifier reproduces this path from the saved parent native
predictions and linked model lineage.
