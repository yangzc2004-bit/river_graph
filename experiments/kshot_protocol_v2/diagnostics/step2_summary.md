# Step 2a summary (zero-training)

## 2-hop coverage on frozen v2 tasks

| region | % queries >2 hop (K=1) | % >2 hop (K=5) | n query cells (K=5) |
|--------|----------------------:|---------------:|--------------------:|
| `101302` | 45.8% | 1.4% | 216 |
| `101900` | 23.3% | 2.0% | 150 |
| `102701` | 36.3% | 0.0% | 234 |
| `103001` | 47.2% | 0.0% | 108 |
| `510020` | 35.8% | 0.0% | 204 |

- Overall K=1 outside 2-hop: **37.6%**
- Overall K=5 outside 2-hop: **0.7%**

## Decision rule

- If outside-2hop > 20–30% at K=1/3 → support encoder is **necessary**.
- Current K=1 outside-2hop = **37.6%** → support encoder required under the pre-registered rule.

## Proofs

- `step2_hop_decay.md` — 2-hop RF bitwise proof
- `step2_cross_month.md` — cross-month influence ≡ 0
- `step2_tributary_mainstem.csv` — tributary→mainstem pairs
- `step2_reachability.csv` — per query-cell hops

## H2 proxy caveat

Old `kshot_st357` H2 predictions may be used only as **proxy narrative**.
H2X empirical hop-decay is Step 2b (after base training) and is part of
the success gate — not replaced by this file.
