# Candidate rejections (main protocol)

## Design decisions
- **Region unit = HUC6 + largest connected task component** (not raw HUC8).
- **K = (0, 1, 3, 5)** only; K=10 dropped (no candidate component supports it).
- **Base model = H2X** (`transport_enc`); H2 is ablation only.

## HUC8 not used as primary
- HUC8 `10300101`: 3 components; kept as **secondary case study** only (induced subgraph not a single tributary).
- HUC8 `10130201`: 3 components; kept as **secondary case study** only (induced subgraph not a single tributary).

## HUC6 backup / excluded
- HUC6 `101000`: backup (component n=10, months_ge5=11).
- HUC6 `101301`: backup (component n=8, months_ge5=9).

## K=10
- Removed from main protocol: only `10300101` ever offered K=10 (1 month, 2 query cells). HUC6 components are 7–10 stations, so K=10 + min_query is infeasible without coarsening to HUC4.
