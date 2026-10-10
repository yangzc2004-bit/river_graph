"""Controlled DOC signal routing on measured channel paths."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

PULSE_SD = .15
PERIODS = (1., 4., 12.)


def normalized_inputs(area, distance, basin_area):
    area, distance = np.asarray(area, float), np.asarray(distance, float)
    if area.ndim != 1 or area.shape != distance.shape or not len(area):
        raise ValueError("aligned nonempty area/path vectors required")
    if not np.isfinite(area).all() or (area < 0).any() or area.sum() <= 0:
        raise ValueError("finite nonnegative catchment areas with positive total required")
    if not np.isfinite(basin_area) or basin_area <= 0:
        raise ValueError("positive measured basin area required")
    good = (area > 0) & np.isfinite(distance) & (distance >= 0)
    if not good.any():
        raise ValueError("no positive-area reachable paths")
    return area[good]/area[good].sum(), distance[good]/np.sqrt(basin_area), area[good].sum()/area.sum()


def contract_paths(delay, weights, factor):
    if not 0 <= factor <= 1:
        raise ValueError("path contraction must be between zero and one")
    mean = np.dot(weights, delay)
    return mean+factor*(np.asarray(delay)-mean)


def distribute_reach_inputs(delay, weights, reach_lengths, points=5):
    """Uniform input along each reach, approximated by midpoint quadrature.

    Lengths use the same normalized units as delay. Preserve each incremental
    catchment's total flow and its mean delay; only within-reach spread changes.
    """
    delay, weights, lengths = (np.asarray(a, float) for a in (delay, weights, reach_lengths))
    if delay.ndim != 1 or not len(delay) or delay.shape != weights.shape or delay.shape != lengths.shape:
        raise ValueError("aligned nonempty reach arrays required")
    if not isinstance(points, int) or points < 1:
        raise ValueError("positive integer quadrature count required")
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("finite nonnegative reach lengths required")
    positions = (np.arange(points)+.5)/points-.5
    distributed = delay[:, None]+lengths[:, None]*positions
    if distributed.min() < -1e-10:
        raise ValueError("within-reach source extends past the receiving outlet")
    return np.maximum(distributed.ravel(), 0.), np.repeat(weights/points, points)


def route_pulse(delay, weights, *, offsets=None, survival=None, dt=.005, sigma=PULSE_SD):
    """Concentration-anomaly convolution; pre-existing steady water flow is one.

    Linear interpolation of the arrival kernel preserves its first moment.
    Negative forcing offsets are legal; negative travel delays are not.
    """
    delay, weights = np.asarray(delay, float), np.asarray(weights, float)
    offsets = np.zeros_like(delay) if offsets is None else np.asarray(offsets, float)
    survival = np.ones_like(delay) if survival is None else np.asarray(survival, float)
    if delay.ndim != 1 or not len(delay) or not (delay.shape == weights.shape == offsets.shape == survival.shape):
        raise ValueError("aligned nonempty source arrays required")
    if not np.isfinite(np.concatenate([delay, weights, offsets, survival])).all():
        raise ValueError("finite source arrays required")
    if (delay < -1e-12).any() or (weights <= 0).any() or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("causal delays and positive unit-sum flow weights required")
    if (survival < 0).any() or (survival > 1).any() or dt <= 0 or sigma <= 0:
        raise ValueError("survival in [0,1] and positive time resolution required")
    arrival = delay+offsets
    origin = np.floor(min(0., float(arrival.min()))/dt)*dt
    positions = (arrival-origin)/dt
    low = np.floor(positions).astype(int)
    frac = positions-low
    kernel = np.bincount(low, weights=weights*survival*(1-frac), minlength=int(low.max())+2)
    kernel += np.bincount(low+1, weights=weights*survival*frac, minlength=len(kernel))
    half = int(np.ceil(8*sigma/dt))
    forcing_time = np.arange(-half, half+1)*dt
    forcing = np.exp(-.5*(forcing_time/sigma)**2)
    pulse = fftconvolve(forcing, kernel)
    pulse = np.maximum(pulse, 0.)
    time = origin+forcing_time[0]+np.arange(len(pulse))*dt
    retained = np.dot(weights, survival)
    if retained > 0:
        centroid = np.sum(weights*survival*arrival)/retained
        width = np.sqrt(sigma**2+np.sum(weights*survival*(arrival-centroid)**2)/retained)
    else:
        centroid, width = np.nan, np.nan
    metrics = {
        "pulse_peak": pulse.max(), "pulse_peak_time": time[np.argmax(pulse)],
        "pulse_centroid": centroid, "pulse_sd": width,
        "anomaly_mass_fraction": pulse.sum()/forcing.sum(), "retained_fraction": retained,
        "steady_doc": 5.*retained, "input_flow": weights.sum(), "outlet_flow": 1.,
        "mean_travel_delay": np.dot(weights, delay),
        "travel_delay_sd": np.sqrt(np.sum(weights*(delay-np.dot(weights, delay))**2)),
        "dt": dt,
    }
    for period in PERIODS:
        metrics[f"period_{int(period)}_gain"] = abs(np.sum(weights*survival*np.exp(-2j*np.pi*arrival/period)))
    return metrics, time, pulse


def two_branch_process(total_paths, common_path, weight_a, *, flow_exponent=0.,
                       branch_rate=0., common_rate=0., offset_b=0., dt=.005):
    """Real endpoint distances with a controlled branch/common-segment partition.

    Speeds and rates are imposed scenario parameters. Both sources have the same
    baseline and integrated pulse; only their timing can differ.
    """
    paths = np.asarray(total_paths, float)
    if paths.shape != (2,) or not np.isfinite(paths).all() or (paths < 0).any():
        raise ValueError("two finite causal total paths required")
    if not 0 <= common_path <= paths.min()+1e-10:
        raise ValueError("common segment must fit inside both total paths")
    if not 0 < weight_a < 1 or min(flow_exponent, branch_rate, common_rate) < 0:
        raise ValueError("positive branch flow shares and nonnegative parameters required")
    weights = np.array([weight_a, 1-weight_a])
    branch = np.maximum(0., paths-common_path)
    branch_time = branch/weights**flow_exponent
    common_time = common_path  # Common flow/speed is one.
    delay = branch_time+common_time
    survival = np.exp(-branch_rate*branch_time-common_rate*common_time)
    result, time, pulse = route_pulse(delay, weights, offsets=[0., offset_b], survival=survival, dt=dt)
    result.update(common_path=common_path, branch_a_path=branch[0], branch_b_path=branch[1],
                  branch_a_delay=branch_time[0], branch_b_delay=branch_time[1],
                  weight_a=weight_a, flow_exponent=flow_exponent, branch_rate=branch_rate,
                  common_rate=common_rate, offset_b=offset_b)
    return result, time, pulse
