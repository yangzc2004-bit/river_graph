# Nonlinear chemistry-conditioned DOC decoder

## Motivation

The preceding auxiliary-chemistry study found lower held-station error in all
nine chemistry-tree packages: K5 MAE1.537905 versus1.577596 for the matched
no-auxiliary tree. The linear neural branch did not capture a comparable gain.
Those development results have already been seen. This iteration tests one
specific hypothesis: a learned nonlinear chemical response, conditioned on the
existing ecological/recurrent state, uses the additional measurements better.

DOC remains the only target. The current backbone, source forest OOF base,
source-trained neural correction, ecological memory and support basis stay fixed.
Same-month pH and conductance are measured inputs. They describe retrospective
reconstruction with known auxiliary chemistry. Current measurements cover19.74%
of genuinely DOC-missing cells; the original model is preserved elsewhere.

## Decoder

Source-standardize the actual original550 features plus the four auxiliary
columns. The normalization uses source loss rows only. From the final four
columns learn phi=SiLU(Linear(4,8)). Concatenate original550, phi8 and the
64-dimensional standardized hidden state multiplied by phi8. A zero-initialized
1070-to-1 projection predicts a signed native-unit correction. Total trainable
parameters:40 chemical-encoder parameters plus1071 output parameters=1111.
The chemical encoder is initialized deterministically by training seed.
There is no dropout, backbone update or additional loss.

Three matched modes use identical architecture and the same real-availability
gate: no_aux zeros all auxiliary columns; masks preserves visibility only;
chemistry preserves values and visibility. Even the no_aux mode retains the
same gate. If neither measurement is present, copy the retained parent model
after direct/integrated support adaptation, including its component columns.

Keep the preceding source objective and budget: native MAE with source-Q90
weight2, Adam.001, batch512, gradient clip1, at most120 epochs and patience10.
Source-validation K0 native MAE selects checkpoint and scale{0,.25,.5,1}.
Scale0 and epoch0 are available. Ties favor smaller scale then earlier epoch.
Only the source forest baseline is OOF; the frozen neural expert is not OOF.

## Experiment and comparisons

Partitions142/143/144 and seeds42/43/44 give27 new fits. All use the unchanged
query identities, nested K0/1/3/5 supports and legacy GRU support basis.
Retain the preceding thirteen model names, with neural_* now identifying the
new nonlinear decoder within this version. Reuse the three preceding chemistry
tree controls exactly, without fitting additional trees. Copy old linear
chemistry direct/integrated curves under linear_chemistry names:15 curves total.

Report the preceding twelve contrasts plus nonlinear versus linear chemistry,
direct and integrated at K0/K5:16 specified contrasts. Use the same5000 paired
whole-station bootstrap, seed averaging within partition, equal partition
weights and shared multiplicity for stations appearing in multiple partitions.
Include complete K curves, native/log error, Q90 error and bias, false-high
rate, partition/seed directions and chemistry-availability strata.

Source-validation choices and inactive fallback follow the preceding study.
New support/ecological choices score active validation queries, with fixed
parent errors at inactive queries. Retained references preserve their original
selection. Target outcomes do not choose a mode or K-specific route.

## Products

Archive fitted chemical embeddings, projection weights, source normalizers,
traces, source/validation feature identities and native bases. Save complete
233478-row grids and sixty query panels per package. Bind copied chemistry
products and replay all thirty-six reference panels per package exactly.
Independent replay evaluates the nonlinear operator, selection and adaptations.
This is another development iteration; independent confirmation remains a
separate future experiment after a model has been selected.
