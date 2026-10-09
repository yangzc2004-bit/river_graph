# Structure aware river residual experiment

## Model question

Can observed upstream DOC and the intervening river path improve the current
environment, observation-memory and source-retrieval model at unmonitored
receiving stations? The first experiment targets DOC K0 on source roles.

## Extension of the current model

Keep the complete current predictor as the reference. Add a zero-initialized
river correction in native DOC space:

\[
\hat y_{i,t}=\max(0,\hat y^{current}_{i,t}+\Delta^{river}_{i,t}).
\]

The correction represents additional information after the complete model,
including its existing similarity retrieval. It must not add the same tree
residual twice or compare a new branch only with a weaker environmental base.

Encode each real upstream station path with its length, number of substantial
confluences, tributary drainage-area balance, upstream/receiving drainage area,
and fraction intersecting lake/reservoir paths. Include source observation
age, validity and concentration departure; preserve missing values as explicit
availability rather than imputed observed DOC.

Three small message components share an encoder:

1. **Conveyance** passes upstream departures with path distance and hydrologic
   conditioning.
2. **Mixing** combines substantial tributary contributions. Drainage-area
   weights are an initial proxy, with discharge used where genuinely available.
3. **Storage memory** uses causal upstream history conditioned on mapped lake
   and reservoir exposure.

A gate depending on continuous path attributes blends these components. This
is a structure-aware GNN branch using actual river connectivity. Initial lag
candidates are 0, 1 and 3 months, motivated by the source atlas; retain the
current longer local GRU memory. Monthly lag weights describe predictive memory,
not inferred water velocity. No common concentration-conservation loss is
imposed without reach flows, volumes and source/sink information.

## Matched comparisons

| Arm | Role |
|---|---|
| Complete current model | Fixed performance reference |
| Current plus simple directed correction | Value of available real upstream information |
| Current plus structure-conditioned correction | Value of path and mixing representation |
| Structure-conditioned correction with rewired source links | Value of physical connectivity beyond the added capacity |

Also evaluate a zero-message prediction of the fitted branch and keep all local
features unchanged. Rewiring should preserve receiving candidate counts and
source eligibility; describe its distance/structure matching explicitly.
Use the same 142/143/144 source station roles and seeds 42/43/44. The initial
branch budget remains 30 epochs with patience 5. Keep receiving DOC, pH and
conductance hidden for K0, including historical and availability-derived inputs.

Train residuals against station-blocked predictions of the complete reference,
with the receiving fold excluded from the source library. Audit reusable saved
OOF fits before computing missing folds. The existing tree OOF residual is not
automatically an OOF residual of the complete neural/retrieval model.

## Scientific readout

Report overall and station-weighted MAE, Q90 error and concentration bias;
compare each arm against the same complete reference. Examine gain by continuous
path distance, major tributary balance, storage fraction and available upstream
support. Keep the five atlas profiles as overlapping diagnostic groups.

Real-link improvement over rewiring would show that physical connectivity
adds predictive information. A further structure-conditioned improvement would
show that different paths require different information transformations. If
gain appears only in supported confluences or particular hydrologic conditions,
build the next iteration around that pattern rather than an all-site average.
Geographical and external confirmation follow source development of the new
version; completed test products remain results of their original versions.
