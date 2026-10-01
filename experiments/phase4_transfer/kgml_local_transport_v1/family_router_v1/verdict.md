# Fixed DOC family router

The route is fixed before aggregation: RF-context for E1 and E3, and the local residual expert for E2a and E2b. This summarizes the existing five-seed products and does not fit a new model.

 n_families  n_cells  rf_context_mae  local_residual_mae  learned_conditional_fusion_mae  fixed_router_mae  fixed_router_gain_vs_rf_pct  fixed_router_gain_vs_local_pct
          4    11046        1.592266             1.60789                        1.507467          1.533299                     3.703344                        4.639119

The router is a performance candidate for a unified system; its family-specific rule should be confirmed on a new held-out batch before being presented as a general routing law.
