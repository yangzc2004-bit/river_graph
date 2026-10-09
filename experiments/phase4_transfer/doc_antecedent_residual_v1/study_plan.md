# Antecedent hydro-climate states in the retained native DOC residual

This follows the observed antecedent tree result: state values beat a matched
availability control, but the changed tree does not improve the retained
strong tree. Keep the retained environmental tree and station-fold OOF reference
fixed, so the neural branch is not compensating for a degraded new baseline.

Use source partitions142/143/144 × seeds42/43/44,30 epochs, patience5, tail
weight2 and the same initial ecology/GRU/decay states. Append the four bounded
antecedent descriptor values and four validity flags to the existing38 current-
month features. Keep original interactions,12-month memory,64 hidden units,
linear native-unit residual and ecological-memory fusion unchanged. The eight
new scalar head inputs share its existing1e-3 learning rate. All outputs start
at the fixed tree reference with a zero head.

Compare availability-only and physical-state residuals, each native-only and
complete; preserve all retained parent predictions. Train only source labels,
with station-hidden input views. Temperature/discharge descriptors read present
and past input months only; no receiving-site water quality or target results
are inputs. Old geographical/external queries do not select this mechanism.

Save each fitted stage before export, replay saved predictions, and analyze
all9 packages with5,000 paired station draws. Report actual complete-model gain,
matched availability-control gain, Q90, bias and source partition consistency.
The tree comparison is not positive evidence for the full procedure; use the
new experiment to answer that remaining question. Keep the deployed release
unless a complete, fixed candidate's evidence supports replacement.
