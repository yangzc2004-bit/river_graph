# Step 2a summary (zero-training)

## 2-hop coverage on frozen v2 tasks

Denominator of every percentage below: **query cells** at that K.
The query set is identical for K=1/3/5 (nested support, fixed query),
so the K columns share one denominator per region. Pooled overall is a
micro-average over all query cells; the region-macro line weights each
basin equally.

| region | n query cells | % >2 hop (K=1) | % >2 hop (K=3) | % >2 hop (K=5) |
|--------|--------------:|---------------:|---------------:|---------------:|
| `101302` | 216 | 45.8% | 11.1% | 1.4% |
| `101900` | 150 | 23.3% | 4.0% | 2.0% |
| `102701` | 234 | 36.3% | 5.1% | 0.0% |
| `103001` | 108 | 47.2% | 10.2% | 0.0% |
| `510020` | 204 | 35.8% | 2.9% | 0.0% |

- Pooled K=1 outside 2-hop: **37.6%** (343 of 912 query cells)
- Pooled K=3 outside 2-hop: **6.5%**
- Pooled K=5 outside 2-hop: **0.7%**
- Region-macro K=1 outside 2-hop: **37.7%** (mean of the 5 per-region rates above)

## Decision rule

- If outside-2hop > 20–30% at K=1/3 → support encoder is **necessary**.
- Current K=1 outside-2hop = **37.6%** (pooled) → support encoder required under the pre-registered rule.

## Tributary→mainstem (stream order)

Stream-order source: `reach_attributes.csv` (node-level `streamorde` from reach_attributes.csv).

Two different denominators — do not mix them:

- **pairs** = (support, query) rows in `step2_tributary_mainstem.csv`; each query contributes K pairs, so pair counts scale with K.
- **query cells** = fixed query set (same as the 2-hop table).

### Pair-level order relation (denominator: pairs at that K)

| region | K | pairs | trib_to_main | main_to_trib | same_order | unknown_order |
|--------|--:|------:|-------------:|-------------:|-----------:|--------------:|
| `101302` | 1 | 216 | 79 (36.6%) | 91 (42.1%) | 46 (21.3%) | 0 (0.0%) |
| `101302` | 3 | 648 | 241 (37.2%) | 256 (39.5%) | 151 (23.3%) | 0 (0.0%) |
| `101302` | 5 | 1080 | 409 (37.9%) | 421 (39.0%) | 250 (23.1%) | 0 (0.0%) |
| `101900` | 1 | 150 | 71 (47.3%) | 59 (39.3%) | 20 (13.3%) | 0 (0.0%) |
| `101900` | 3 | 450 | 194 (43.1%) | 195 (43.3%) | 61 (13.6%) | 0 (0.0%) |
| `101900` | 5 | 750 | 335 (44.7%) | 311 (41.5%) | 104 (13.9%) | 0 (0.0%) |
| `102701` | 1 | 234 | 78 (33.3%) | 70 (29.9%) | 86 (36.8%) | 0 (0.0%) |
| `102701` | 3 | 702 | 242 (34.5%) | 195 (27.8%) | 265 (37.7%) | 0 (0.0%) |
| `102701` | 5 | 1170 | 421 (36.0%) | 331 (28.3%) | 418 (35.7%) | 0 (0.0%) |
| `103001` | 1 | 108 | 45 (41.7%) | 47 (43.5%) | 16 (14.8%) | 0 (0.0%) |
| `103001` | 3 | 324 | 126 (38.9%) | 139 (42.9%) | 59 (18.2%) | 0 (0.0%) |
| `103001` | 5 | 540 | 218 (40.4%) | 236 (43.7%) | 86 (15.9%) | 0 (0.0%) |
| `510020` | 1 | 204 | 67 (32.8%) | 67 (32.8%) | 70 (34.3%) | 0 (0.0%) |
| `510020` | 3 | 612 | 221 (36.1%) | 193 (31.5%) | 198 (32.4%) | 0 (0.0%) |
| `510020` | 5 | 1020 | 364 (35.7%) | 319 (31.3%) | 337 (33.0%) | 0 (0.0%) |

- Unknown stream-order pairs: **0 of 8208** (0.0%)
- Every pair has a stream-order relation; no unknown residual.

### Query-level (denominator: query cells at that K)

A query counts once if **any** of its K supports is a ≤2-hop trib→main pair.

| region | K | n query cells | % with ≥1 trib→main ≤2hop |
|--------|--:|--------------:|--------------------------:|
| `101302` | 1 | 216 | 17.6% |
| `101302` | 3 | 216 | 32.9% |
| `101302` | 5 | 216 | 42.1% |
| `101900` | 1 | 150 | 35.3% |
| `101900` | 3 | 150 | 54.0% |
| `101900` | 5 | 150 | 66.0% |
| `102701` | 1 | 234 | 22.2% |
| `102701` | 3 | 234 | 42.7% |
| `102701` | 5 | 234 | 50.4% |
| `103001` | 1 | 108 | 20.4% |
| `103001` | 3 | 108 | 35.2% |
| `103001` | 5 | 108 | 40.7% |
| `510020` | 1 | 204 | 19.6% |
| `510020` | 3 | 204 | 46.1% |
| `510020` | 5 | 204 | 58.8% |

## Proofs

- `step2_hop_decay.md` — 2-hop RF bitwise proof
- `step2_cross_month.md` — cross-month influence ≡ 0
- `step2_tributary_mainstem.csv` — tributary→mainstem pairs (source: `reach_attributes.csv`)
- `step2_reachability.csv` — per query-cell hops

## H2 proxy caveat

Old `kshot_st357` H2 predictions may be used only as **proxy narrative**.
H2X empirical hop-decay is Step 2b (after base training) and is part of
the success gate — not replaced by this file.
