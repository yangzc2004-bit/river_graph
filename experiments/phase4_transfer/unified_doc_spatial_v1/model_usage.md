# Using the integrated DOC model

Each completed run saves one fitted prediction package: environmental forests,
the recurrent residual model, selected expert combination, and station-calibration
strengths. The training batch contains nine such packages; they are replicate
fits, not nine stages of one model or an automatically selected ensemble.

## Reproduce a saved station-adaptation prediction

Run Python through the repository's uv environment. The following example uses
the first completed partition and training seed:

```python
from pathlib import Path
import numpy as np
import torch

from river_graph.experiments.transfer import DATASETS
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.unified_doc import UnifiedDOCReconstructor

root = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation")
run = root / "runs/split142_seed42"
dataset = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
with np.load(root / "masks/split142.npz") as saved:
    split = {role: saved[role] for role in saved.files}

model = UnifiedDOCReconstructor.load(run, dataset, split)
support, query = support_query_cells(
    split, target_role="test", k=5, n_months=dataset["y"].shape[1]
)
support_values = np.asarray(dataset["y"]).ravel()[support]
prediction = model.predict(
    query, support_cells=support, support_values=support_values, k=5,
    arm="hybrid", calibrated=True,
)
```

`prediction` is DOC in mg/L, aligned exactly with `query`. Cell IDs follow
`station_index * n_months + month_index`. Query labels are not an inference
argument. The same interface supports `k=0`, `1`, and `3`; support must contain
the specified number of observations per queried station and remain disjoint
from query cells. For the matched environmental baseline use `arm="context"`.

## Inspect the components

```python
components = model.predict_components()
```

The returned station-by-month arrays are `context_pred`, `temporal_pred`,
`hybrid_pred`, `local_pred`, and `temporal_delta`. Concentration predictions are
in mg/L; `temporal_delta` is a correction in transformed target space. These
arrays precede target-station support calibration. `predict(...)` applies that
calibration when requested.

The package retains its fixed cohort, feature preprocessing, and training DOC
visibility. It can reconstruct query months within this grid. Applying it to a
different river dataset requires a separately specified data-adaptation path.

## Reproduce the experiment

```bash
uv run python scripts/run_ladder.py --experiment unified-doc-spatial
uv run python scripts/run_ladder.py --experiment unified-doc-spatial --verify-only
uv run python scripts/analyze_unified_doc_spatial.py
uv run python scripts/analyze_unified_doc_affine_control.py
uv run python scripts/analyze_unified_doc_flow.py
uv run python scripts/plot_unified_doc_manuscript.py
```

The analysis commands require all nine completed runs by default. During a live
batch, `--allow-partial` creates interim analysis separately. The paper uses the
completed batch and the supplementary component controls.

`analyze_unified_doc_flow.py` adds descriptive discharge-variability and DOC
record-availability strata, using training-station tertiles and saved
seed-averaged station responses. Record availability counts observed months;
it does not change the equal K-support budget or open target history inputs.
It writes `confirmation/flow_analysis/`; insufficient discharge records remain
explicit rather than disappearing from the comparison. No model is refitted.
The plotting command writes publication PDF/PNG figures and captions under
`confirmation/analysis/manuscript_figures/`.

## Completed study

All nine fits are complete: three station partitions by three training seeds.
They cover 172 distinct test stations and 10,520 distinct fixed query cells
(12,865 cell occurrences across partitions). The principal MAEs are:

| Predictor | No target observations | Five target observations |
|---|---:|---:|
| Environmental ExtraTrees | 1.9028 | 1.6094 |
| Environmental–temporal hybrid | 2.0015 | 1.6261 |

Five-observation calibration improves both models in every partition: 15.42%
for ExtraTrees and 18.76% for the hybrid. The calibrated hybrid does not establish
an additional MAE advantage over equally calibrated ExtraTrees. Without support,
its error is 5.19% higher. The full paired intervals are in
`confirmation/analysis/primary_comparisons.csv`.

The supplementary affine-context and two-forest controls show that transferring
the fitted output combination is a priority for further improvement. The trained
recurrent residual adds only a small spatial increment relative to combining the
two forests, and that increment changes direction with the support budget.
These results do not replace the earlier positive temporal reconstruction results.

Both prediction arms remain available through the saved interface. Their fitted
forests, recurrent weights and full-grid component products are retained locally
in each run directory. Git records the compact query products, configurations,
analysis and manuscript figures; the larger forest packages and full grids are
local training artifacts. The manuscript is
`docs/paper/latex/spatial_transfer_draft_v1.tex` with a compiled PDF beside it.
