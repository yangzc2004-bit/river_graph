"""Region-blocked source observation views for unmonitored DOC learning."""
from __future__ import annotations

import numpy as np


def huc4_source_folds(train_cells, n_months, huc_cd, seed, n_folds=5):
    """Keep each source HUC4 intact and greedily balance observed training rows.

    HUC identity comes from aligned graph-node huc_cd, never station identifiers.
    Counts affect balancing; target concentrations do not. Empty/non-source
    stations are excluded. These are observation-hidden source training views,
    not separate out-of-fold neural models.
    """
    cells = np.asarray(train_cells)
    codes = [str(value).split(".")[0].zfill(8) for value in huc_cd]
    if (not isinstance(n_months, (int, np.integer)) or n_months < 1 or cells.ndim != 1
            or cells.dtype.kind not in "iu" or not len(cells) or (cells < 0).any()
            or (cells >= len(codes)*n_months).any() or len(np.unique(cells)) != len(cells)
            or any(len(code) != 8 or not code.isdigit() for code in codes)):
        raise ValueError("valid source cells and aligned eight-digit HUC identities are required")
    station, station_count = np.unique(cells//n_months, return_counts=True)
    regions = np.array([code[:4] for code in codes])
    names = np.unique(regions[station])
    if len(names) < 2 or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("regional OOF requires at least two source HUC4 groups")
    count = {name: int(station_count[regions[station] == name].sum()) for name in names}
    order = np.random.default_rng(seed).permutation(names).tolist()
    order.sort(key=lambda name: -count[name])
    groups, loads = [[] for _ in range(min(n_folds, len(names)))], np.zeros(min(n_folds, len(names)), int)
    for region in order:
        fold = int(np.argmin(loads))
        groups[fold].extend(station[regions[station] == region].tolist())
        loads[fold] += count[region]
    return [np.array(sorted(group), dtype=np.int64) for group in groups]
