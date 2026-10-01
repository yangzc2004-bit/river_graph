# Few-shot spatial transfer verdict

The strict E3 benchmark hides every DOC label at a target station. That is a
useful zero-shot stress test, but a monitoring network can often obtain a few
initial samples after a new station is installed. This experiment adds a
separate few-shot deployment mode on top of the source-similarity regional
expert.

## Protocol

* The same 43 E3 target stations are held out.
* For each target station, K = 1, 3, or 5 observed target-month labels are
  opened as support. Five evenly spaced observed months provide a fixed,
  label-independent support schedule.
* The forest is still fitted only on source stations. Target support labels
  enter the prediction context and a station-level log-space level correction;
  query labels remain hidden until evaluation.
* The correction strength is selected on a separate station-held-out split.
  Five seeds selected `alpha = 0.5` for K=5.

## Result for K=5

| model | query MAE | query RMSE | relative MAE |
| --- | ---: | ---: | ---: |
| Source regional expert, no support correction | 2.459 (mean across seeds) | -- | -- |
| Source regional expert + five target support months | **2.137** (median ensemble) | **4.715** | **13.1% lower** |

The five individual calibrated MAEs are 2.141, 2.146, 2.135, 2.136 and
2.152 (mean 2.142, SD 0.007). The gain is consistent across seeds. K=1 and
K=3 also improve, but K=5 gives the clearest reduction.

The query set contains 2,316 cells after removing the 215 support cells. The
zero-support comparison must therefore be made on the same reduced query set;
on that paired set, the uncorrected source expert has MAE about 2.459 and the
K=5 corrected model has MAE 2.137.

## Interpretation

The result identifies the missing ingredient in the worst strict-E3 cases:
the target station's local DOC level. A few target observations let the model
correct the level while retaining the source-station regional model for the
seasonal pattern. This is a spatial adaptation result, not evidence that a
deeper GNN alone solves zero-shot extrapolation.

The recommended system now has two explicit deployment modes:

1. **Zero-shot new station:** source-similarity regional expert, E3 MAE 2.459.
2. **Five-sample warm start:** the same expert plus target-level calibration,
   query MAE 2.137.

The support experiment is kept separate from the strict zero-shot benchmark;
the two numbers answer different operational questions.
