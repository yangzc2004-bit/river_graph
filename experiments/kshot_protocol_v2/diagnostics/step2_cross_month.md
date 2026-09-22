# Step 2a — Cross-month support isolation (architecture)

`GCNDocModel.predict` / `fit` call `fwd(xt[j])` **independently for each month
index `j`**. There is no temporal edge, RNN, or cross-month feature.

Therefore a support cell at month `t' ≠ t`:

1. is never placed in `xt[t]`'s DOC channel;
2. cannot enter the 2-layer spatial message passing of month `t`;
3. has **exact zero influence** on a query at month `t`.

## Decision rule

- Same-month support is the only channel available to the current GNN.
- If a region lacks same-month K, **Case C (explicit cross-time support) must be
  designed** — it cannot be recovered by "letting" the model see old samples.
