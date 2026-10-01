# Spatial transfer upgrade: validation-first verdict

## Scientific question

Can the current DOC reconstruction system adapt its training distribution to a
station that is completely held out in space, without using the target station's
DOC labels? The upgrade treats spatial transfer as a regionalization problem:
the target station selects a compact set of source stations with similar
hydro-ecological and graph profiles, and a context ExtraTrees model is fitted
on those source stations.

This is an adaptation layer around the existing model family. It does not
replace the temporal H2X-T/KGML branch used for temporal forecasting.

## Design

* Target task: frozen `e3_spatial_seed42` DOC holdout (43 stations, 2,531
  station-month cells).
* Source descriptor: label-free temperature/discharge climatology and
  availability, static coordinates, ecological regime, graph degree and edge
  attribute summaries.
* Candidate source pools: the 20, 40, 80 and 160 nearest source stations in
  descriptor space.
* Nested selection: a separate station-held-out split of the non-test source
  stations selected the source-pool size and the forest leaf size. The frozen
  E3 test stations were scored only after selection.
* Final model: 40-source-station context ExtraTrees, 120 trees, leaf size 4,
  five seeds (42--46), with a median prediction ensemble.

## Result

| E3 model | MAE | RMSE | relative to existing E3 ensemble |
| --- | ---: | ---: | ---: |
| Existing selected context ensemble | 2.537 | 5.326 | -- |
| Source-similarity regional expert, seed median | **2.459** | **5.306** | **3.1% lower MAE** |

The five individual source-selection MAEs are 2.475, 2.468, 2.463, 2.457 and
2.466 (mean 2.466, SD 0.006). The improvement is consistent across seeds and
does not require target DOC labels.

Replacing only the E3 branch in the existing four-scenario DOC product changes
the cell-weighted median-ensemble MAE from 1.471 to 1.453 (1.2% overall). The
smaller pooled gain reflects that the spatial holdout is one of four scenarios;
the spatial branch itself is the relevant comparison for this upgrade.

## What did not improve

The following alternatives were evaluated with the same nested station-heldout
logic and were not promoted: density-ratio weighting, multi-hop graph context,
geographic nearest-station summaries, station-profile bias correction,
headwater experts, HUC-level context, tail-weighted forests, and global/source
prediction blends. They either lost on matched validation or produced only
split-specific gains.

The dominant remaining error is a small number of high-DOC target headwaters
with no monitored upstream source. The worst case (`06355310`) has test MAE
about 18.7, true DOC mean about 32.4, and predicted mean about 13.8. A deeper
GNN cannot infer this unseen local state from the current covariates; further
gain requires target-station DOC support or new local process covariates.

## Route for the next model version

Use source-similarity regionalization as the spatial branch of the current
system:

* temporal extrapolation: retain the existing H2X-T/KGML hybrid;
* spatial extrapolation: use the nested 40-source regional expert;
* future spatial experiments: add a small target-station DOC support set and
  test adaptation of the regional expert, keeping the zero-support E3 result as
  the strict transfer benchmark.

This direction follows recent water-quality transfer work that combines shared
patterns with station-specific adaptation and studies progressive local data
integration, while preserving the current project's observed-label split.
