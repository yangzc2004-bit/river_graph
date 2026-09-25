# Stage 2C execution resume addendum (2026-09-25)

The first Stage 2C execution stopped after 82 unit prediction files. No
aggregate metrics were produced and no query labels were opened. The initial
runner did not checkpoint source-validation rows per unit, so those partial
files are not accepted as a completed result.

The execution implementation was corrected to:

- checkpoint one prediction and one source-validation sidecar per unit;
- resume one unit per process to avoid long-process memory growth;
- bind the runner source hash into every unit configuration;
- reject an existing prediction whose configuration hash does not match the
  current reviewed plan;
- finalize only after all 180 unit checkpoints are present.

This is an implementation and provenance correction only. The arms, task
manifest, visibility rules, K values, endpoints, and training budget are
unchanged. The 82 partial files are treated as superseded and will be
recomputed under the revised execution identity before any result is
evaluated.
