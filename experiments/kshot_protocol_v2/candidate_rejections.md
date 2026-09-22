# Candidate rejections (main protocol)

## Design decisions
- **Region unit = HUC6 + largest connected task component** (not raw HUC8).
- **K = (0, 1, 3, 5)** only; K=10 dropped (no candidate component supports it).
- **Base model = H2X** (`transport_enc`); H2 is ablation only.

## Component-choice note (HUC6 103001)
- `103001` largest component = 10 stations / 17 K=5 months; a secondary 7-station component offers 35 K=5 months. Main protocol keeps the **largest** component (more query sites per month, fewer task-months). Documented, not a blocker.

## All HUC6 components ranked 0 — eligibility table

| HUC6 | n_comp | n_st | K=5 months | eligible | note |
|------|-------:|-----:|-----------:|----------|------|
| `100301` | 1 | 2 | 0 | no | too few stations/months |
| `100402` | 1 | 1 | 0 | no | too few stations/months |
| `100600` | 1 | 4 | 0 | no | too few stations/months |
| `100700` | 1 | 4 | 0 | no | too few stations/months |
| `100800` | 1 | 1 | 0 | no | too few stations/months |
| `100901` | 2 | 7 | 0 | no | too few stations/months |
| `100902` | 1 | 7 | 2 | no | too few stations/months |
| `101000` | 1 | 10 | 11 | yes | backup |
| `101101` | 7 | 1 | 0 | no | too few stations/months |
| `101102` | 3 | 4 | 0 | no | too few stations/months |
| `101201` | 3 | 1 | 0 | no | too few stations/months |
| `101202` | 1 | 6 | 0 | no | too few stations/months |
| `101301` | 1 | 8 | 9 | yes | backup |
| `101302` | 5 | 8 | 28 | yes | PRIMARY |
| `101303` | 1 | 1 | 0 | no | too few stations/months |
| `101401` | 1 | 1 | 0 | no | too few stations/months |
| `101500` | 1 | 1 | 0 | no | too few stations/months |
| `101701` | 1 | 3 | 0 | no | too few stations/months |
| `101702` | 1 | 1 | 0 | no | too few stations/months |
| `101800` | 2 | 3 | 0 | no | too few stations/months |
| `101900` | 7 | 10 | 22 | yes | PRIMARY |
| `102001` | 1 | 1 | 0 | no | too few stations/months |
| `102002` | 1 | 3 | 0 | no | too few stations/months |
| `102100` | 2 | 1 | 0 | no | too few stations/months |
| `102200` | 1 | 2 | 0 | no | too few stations/months |
| `102300` | 1 | 3 | 0 | no | too few stations/months |
| `102400` | 1 | 2 | 0 | no | too few stations/months |
| `102500` | 1 | 1 | 0 | no | too few stations/months |
| `102600` | 1 | 1 | 0 | no | too few stations/months |
| `102701` | 1 | 8 | 32 | yes | PRIMARY |
| `102702` | 1 | 5 | 0 | no | too few stations/months |
| `102801` | 1 | 1 | 0 | no | too few stations/months |
| `102802` | 1 | 1 | 0 | no | too few stations/months |
| `102901` | 1 | 3 | 0 | no | too few stations/months |
| `102902` | 1 | 2 | 0 | no | too few stations/months |
| `103001` | 3 | 10 | 17 | yes | PRIMARY, largest-comp preferred over 7st/35mo comp |
| `103002` | 1 | 1 | 0 | no | too few stations/months |
| `110100` | 10 | 4 | 0 | no | too few stations/months |
| `110200` | 2 | 3 | 0 | no | too few stations/months |
| `110300` | 2 | 2 | 0 | no | too few stations/months |
| `110500` | 1 | 1 | 0 | no | too few stations/months |
| `110702` | 2 | 1 | 0 | no | too few stations/months |
| `110800` | 2 | 1 | 0 | no | too few stations/months |
| `110902` | 2 | 1 | 0 | no | too few stations/months |
| `111101` | 1 | 1 | 0 | no | too few stations/months |
| `501000` | 2 | 12 | 0 | no | too few stations/months |
| `502000` | 1 | 7 | 0 | no | too few stations/months |
| `503010` | 3 | 2 | 0 | no | too few stations/months |
| `504000` | 3 | 1 | 0 | no | too few stations/months |
| `505000` | 1 | 4 | 0 | no | too few stations/months |
| `507020` | 1 | 5 | 0 | no | too few stations/months |
| `508000` | 1 | 1 | 0 | no | too few stations/months |
| `509010` | 1 | 6 | 0 | no | too few stations/months |
| `509020` | 1 | 1 | 0 | no | too few stations/months |
| `510020` | 1 | 7 | 34 | yes | PRIMARY |
| `512010` | 1 | 1 | 0 | no | too few stations/months |
| `512011` | 1 | 1 | 0 | no | too few stations/months |
| `512020` | 3 | 5 | 0 | no | too few stations/months |
| `513020` | 1 | 1 | 0 | no | too few stations/months |
| `514020` | 1 | 3 | 0 | no | too few stations/months |
| `601010` | 5 | 2 | 0 | no | too few stations/months |
| `601020` | 2 | 2 | 0 | no | too few stations/months |
| `602000` | 1 | 1 | 0 | no | too few stations/months |
| `603000` | 1 | 1 | 0 | no | too few stations/months |
| `604000` | 1 | 1 | 0 | no | too few stations/months |
| `701020` | 1 | 5 | 0 | no | too few stations/months |
| `702000` | 1 | 4 | 0 | no | too few stations/months |
| `702001` | 3 | 2 | 0 | no | too few stations/months |
| `703000` | 1 | 3 | 0 | no | too few stations/months |
| `704000` | 1 | 1 | 0 | no | too few stations/months |
| `705000` | 3 | 4 | 0 | no | too few stations/months |
| `706000` | 2 | 1 | 0 | no | too few stations/months |
| `708010` | 3 | 2 | 0 | no | too few stations/months |
| `708020` | 3 | 9 | 0 | no | too few stations/months |
| `709000` | 1 | 2 | 0 | no | too few stations/months |
| `710000` | 1 | 1 | 0 | no | too few stations/months |
| `711000` | 1 | 1 | 0 | no | too few stations/months |
| `712000` | 2 | 9 | 0 | no | too few stations/months |
| `713000` | 3 | 2 | 0 | no | too few stations/months |
| `713001` | 1 | 2 | 0 | no | too few stations/months |
| `714010` | 1 | 1 | 0 | no | too few stations/months |
| `802030` | 1 | 3 | 0 | no | too few stations/months |
| `803020` | 1 | 1 | 0 | no | too few stations/months |
| `807010` | 1 | 1 | 0 | no | too few stations/months |

## HUC8 not used as primary
- HUC8 `10300101`: 3 components; kept as **secondary case study** only (induced subgraph not a single tributary).
- HUC8 `10130201`: 3 components; kept as **secondary case study** only (induced subgraph not a single tributary).

## HUC6 backup / excluded
- HUC6 `101000`: backup (component n=10, months_ge5=11).
- HUC6 `101301`: backup (component n=8, months_ge5=9).

## K=10
- Removed from main protocol: only `10300101` ever offered K=10 (1 month, 2 query cells). HUC6 components are 7–10 stations, so K=10 + min_query is infeasible without coarsening to HUC4.
