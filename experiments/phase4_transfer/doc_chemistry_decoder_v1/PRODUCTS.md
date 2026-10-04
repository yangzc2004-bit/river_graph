# Nonlinear chemistry decoder products

This model adds a small learned pH/conductance representation to the current
DOC model. The original recurrent/ecological trunk, native prediction,
ecological residual memory and target-support basis are fixed. No new output
analyte is fitted.

Each of nine runs stores three1,111-parameter heads, their seeded initial
chemical embeddings, learned embeddings/projections, source normalizers and
training/selection traces. Inputs are554 columns; the learned output basis has
1070 columns. no_aux, masks and chemistry share the same real-availability gate.
Absent-chemistry cells copy the retained model after support integration.

The three chemistry trees are fixed copied controls from
doc_auxiliary_chemistry_v1. Their trained states and provenance remain there,
bound by the preceding completion/full-grid/prediction identities. This study
fits27 neural heads and zero trees. The earlier linear chemistry direct and
integrated predictions are copied under linear_chemistry names.

Full grids contain233478 station-months each, including original/native
components, nonlinear predictions, fixed chemistry-tree predictions, the
preceding linear chemical prediction and auxiliary observation flags. Query
products contain15 model curves at K0/1/3/5:60 panels per package. Truth is
attached after source-validation selection and allowed target-support use.
source_validation.csv contains60 final full/active validation summaries,
including eight copied linear-chemistry rows. reference_checks.json verifies
thirty-six copied control panels per package.

Native source bases and original550 features reproduce the previous study.
Only the source forest baseline is OOF; the frozen source neural correction
was trained on source labels. Positive-K supports and calendar anchor describe
retrospective reconstruction. Same-month chemistry is observed covariate
information, not chemistry predicted from held-out DOC.

```bash
OMP_NUM_THREADS=2 uv run python scripts/run_ladder.py --experiment doc-chemistry-decoder-v1
uv run python scripts/verify_doc_chemistry_decoder_v1.py
uv run python scripts/analyze_doc_chemistry_decoder_v1.py
```

The runtime source archive preserves the executed version. Historical
experiments remain unchanged; _smoke is a separate three-epoch implementation
check and does not enter scientific analysis.
