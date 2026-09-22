# Step 2a — 2-hop receptive-field proof (random weights, no training)

Two-layer `TransportGCNImputer` message passing has an effective
receptive field of **2 hops**. Perturbing an input at node 0:

| node | hop from 0 | |Δ output| |
|-----:|-----------:|------------:|
| 0 | 0 | 0.049947 |
| 1 | 1 | 0.0229324 |
| 2 | 2 | 0.000805736 |
| 3 | 3 | **0.0** |

Node 3 is bitwise unchanged (`Δ = 0`) — support beyond 2 hops cannot
affect a query under this architecture, trained or not.

## Decision rule

- If a large fraction of v2 queries has `min_hop > 2`, a support encoder
  is **necessary**, not optional.
- If most queries are within 2 hops, failure cannot be blamed on RF alone.
