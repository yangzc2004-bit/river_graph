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
```

The analysis commands require all nine completed runs by default. During a live
batch, `--allow-partial` creates interim analysis separately. The paper uses the
completed batch and the supplementary component controls.
