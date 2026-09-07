# river_graph

**Topology-aware Graph Neural Networks for Reconstructing Sparse Dissolved Organic Carbon Observations in the Mississippi River Basin** (working title)

## Idea

DOC observations from USGS monitoring stations are sparse and irregular in time.
We model stations as nodes in a graph whose edges follow the river network
(upstream → downstream), and test whether a topology-aware GNN can reconstruct
withheld DOC observations better than topology-agnostic baselines.

MVP questions:

1. Can USGS DOC stations form a meaningful graph? → yes, via NHDPlus/NLDI linkage
2. Can a GNN predict masked DOC observations?
3. Does real river topology contribute? → ablation: MLP (no graph) vs random graph vs river graph

## Data profile (Mississippi River Basin, HUC2: 05 06 07 08 10 11)

From the NWIS water-quality series catalog (2026-09 snapshot):

- 11,948 stations with ≥1 DOC sample (USGS pcode 00681), 75,202 samples total (1958–2024)
- 660 stations with ≥20 samples — median record span 6.2 yr, median sampling gap ~48 days
- Covariate availability among those 660: water temperature 93%, pH 94%, specific conductance 96%, discharge 74%

Sampling is not synchronized across stations, so the MVP target is
**seasonal-mean DOC per station** rather than instantaneous values.

## Layout

```
configs/            experiment configs (mvp.yaml)
data/               raw + processed data (gitignored, reproducible via scripts)
scripts/            data download / analysis entry points
src/river_graph/
  config.py         config loading
  data/             NWIS/WQP data access
  graph/            station -> river network graph construction
  models/           GNN / MLP models
  baselines/        kriging / random forest baselines
  experiments/      mask reconstruction + topology ablation
tests/
```

## Setup

```bash
uv sync                 # core deps
uv sync --extra gnn     # + torch / torch-geometric
```

## Data pipeline

```bash
python scripts/fetch_doc_inventory.py    # station list + per-parameter sample counts (NWIS)
python scripts/analyze_doc_inventory.py  # availability report -> data/processed/
bash    scripts/download_wqp_by_huc.sh data/raw/mrb_huc4.txt   # DOC values (WQP, resumable)
```

Data sources: USGS NWIS (waterservices.usgs.gov), Water Quality Portal
(waterqualitydata.us), NHDPlus HR / NLDI (river network, upcoming).
