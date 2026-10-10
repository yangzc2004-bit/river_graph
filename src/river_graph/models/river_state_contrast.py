"""Express environmental river messages relative to the receiving site."""
from __future__ import annotations

import numpy as np


def receiver_contrast(arrays, cells, full_states, *, copy=True):
    """Upstream state at its allowed lag minus receiver state at query time.

    The upstream gather, candidate eligibility, attention features and query
    remain unchanged. Both states use the same frozen environmental encoder;
    neither contains receiving-site water quality. This is an information
    contrast, not a concentration gradient or physical transport equation.
    """
    values, valid = np.asarray(arrays['states']), np.asarray(arrays['valid'])
    cells, full_states = np.asarray(cells), np.asarray(full_states)
    if (full_states.ndim != 3 or values.ndim != 4 or valid.dtype != bool
            or valid.shape != values.shape[:-1] or values.shape[-1] != full_states.shape[-1]
            or cells.shape != (len(values),) or cells.dtype.kind not in 'iu'
            or (cells < 0).any() or (cells >= np.prod(full_states.shape[:2])).any()
            or values.dtype != full_states.dtype):
        raise ValueError('aligned environmental values and global receiving cells required')
    result = values.copy() if copy else values
    receiver = full_states.reshape(-1, full_states.shape[-1])[cells]
    if not np.isfinite(receiver).all():
        raise ValueError('nonfinite receiving environmental state')
    for start in range(0, len(values), 512):
        stop = min(start+512, len(values))
        result[start:stop] = np.where(valid[start:stop, ..., None],
            result[start:stop]-receiver[start:stop, None, None], 0.)
    return {**arrays, 'states': result}
