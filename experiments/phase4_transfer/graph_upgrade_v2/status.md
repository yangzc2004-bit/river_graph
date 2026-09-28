# Graph upgrade status

The first M1 and M2 pilots are complete. Their results and interpretation are
in `m1_verdict.md` and `m2_verdict.md`; each mechanism directory contains its
run plan, metrics, full-grid products and per-run metadata. The first M3
attempt was stopped after its initial smoke exposed a slow rolling-window
implementation. A refactored continuation is in `continuation_r1/` and is
being evaluated separately.

The `development_v0/` directory retains the first code snapshot and the
interrupted M3 development inventory. It is not mixed into the current pilot
tables. The M1/M2 raw prediction grids remain in their mechanism directories;
the committed tables are the compact metrics and summaries.
