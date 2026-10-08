"""Serial waterbody responses on the previously selected directed river paths."""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len


class StoragePaths:
    """Midpoint routes with contiguous physical waterbodies represented once.

    Persistent lists share downstream storage elements. A list node contains
    the remaining length of its FIRST waterbody and a pointer to the next
    distinct downstream waterbody. Partial source reaches have their own first
    element; non-storage reaches reuse the complete downstream list.
    """

    def __init__(self, frame, receiving_comid, distances, *, scale=1.):
        required = {"comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm",
                    "areasqkm", "wbareatype", "wbareacomi"}
        if required-set(frame) or frame.comid.duplicated().any() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("unique reaches, routing/storage columns and positive scale required")
        self.comids = frame.comid.to_numpy(np.int64)
        self.length = frame.lengthkm.to_numpy(float)/scale
        self.area = frame.areasqkm.to_numpy(float)
        distance = np.asarray(distances, float)/scale
        if distance.shape != self.length.shape or not np.isfinite(distance).all():
            raise ValueError("finite distances aligned with reachable reaches required")
        if not np.isfinite(self.length).all() or (self.length < 0).any():
            raise ValueError("finite nonnegative reach lengths required")
        if not np.isfinite(self.area).all() or (self.area < 0).any() or self.area.sum() <= 0:
            raise ValueError("finite nonnegative incremental areas with positive total required")
        root_pos = np.flatnonzero(self.comids == receiving_comid)
        if len(root_pos) != 1:
            raise ValueError("one receiving reach required")
        root = int(root_pos[0])
        if not np.isclose(distance[root], self.length[root]/2, atol=1e-8):
            raise ValueError("saved path must terminate at the receiving reach outlet")
        hydro = frame.hydroseq.to_numpy(np.int64)
        if len(np.unique(hydro)) != len(hydro):
            raise ValueError("unique hydroseq required")
        lookup = {int(h): i for i, h in enumerate(hydro)}
        successor = np.full(len(frame), -1, dtype=int)
        secondary = np.zeros(len(frame), bool)
        errors = []
        for i, (primary, minor) in enumerate(zip(frame.dnhydroseq, frame.dnminorhyd)):
            if i == root:
                continue
            selected = False
            for alternative, seq in enumerate((primary, minor)):
                j = lookup.get(int(seq)) if np.isfinite(seq) else None
                if j is None or j == i:
                    continue
                error = abs(distance[i]-distance[j]-(self.length[i]+self.length[j])/2)
                if error <= 1e-7/scale and hydro[j] < hydro[i]:
                    successor[i], secondary[i] = j, bool(alternative)
                    errors.append(error*scale)
                    selected = True
                    break
            if not selected:
                raise ValueError(f"no successor reproduces saved path for COMID {self.comids[i]}")
        is_storage = frame.wbareatype.isin(["LakePond", "Reservoir"]).to_numpy()
        waterbody = frame.wbareacomi.fillna(0).to_numpy(np.int64)
        if np.any(is_storage & (waterbody <= 0)):
            raise ValueError("mapped storage reaches need physical waterbody IDs")
        self.is_storage, self.waterbody = is_storage, waterbody
        self.successor, self.secondary = successor, secondary
        self.distance = distance
        self.max_route_error_km = max(errors, default=0.)
        # Downstream hydroseq precedes upstream hydroseq, even at zero length.
        full = np.zeros(len(frame), int)
        midpoint = np.zeros(len(frame), int)
        tau, tail, depth, total, variance, ids = [0.], [0], [0], [0.], [0.], [0]

        def append(value, next_state, wb):
            if value <= 0:
                return next_state
            node = len(tau)
            tau.append(value)
            tail.append(next_state)
            depth.append(depth[next_state]+1)
            total.append(value+total[next_state])
            variance.append(value*value+variance[next_state])
            ids.append(wb)
            return node

        for i in np.argsort(hydro):
            j = successor[i]
            downstream = full[j] if j >= 0 else 0
            if not is_storage[i]:
                full[i] = midpoint[i] = downstream
                continue
            same = j >= 0 and is_storage[j] and ids[downstream] == waterbody[i]
            previous_length = tau[downstream] if same else 0.
            next_state = tail[downstream] if same else downstream
            full[i] = append(self.length[i]+previous_length, next_state, waterbody[i])
            midpoint[i] = append(self.length[i]/2+previous_length, next_state, waterbody[i])
        self.full_state, self.midpoint_state = full, midpoint
        self.tau = np.asarray(tau)
        self.tail = np.asarray(tail, int)
        self.depth = np.asarray(depth, int)
        self.total = np.asarray(total)
        self.variance = np.asarray(variance)
        self.layers = [np.flatnonzero(self.depth == k) for k in range(1, self.depth.max()+1)]
        if np.any(self.total[midpoint] > distance+1e-8):
            raise ValueError("storage mean exceeds available path mean")

    def inputs(self, points=1):
        if not isinstance(points, int) or points < 1:
            raise ValueError("positive integer source quadrature count required")
        good = self.area > 0
        weights = self.area[good]/self.area[good].sum()
        # Position measured upstream from the downstream end of the source reach.
        position = (np.arange(points)+.5)/points
        delay = self.distance[good, None]+self.length[good, None]*(position-.5)
        state = np.repeat(self.midpoint_state[good, None], points, axis=1)
        first = np.zeros_like(delay)
        continuation = state.copy()
        stored = self.is_storage[good]
        first[stored] = self.tau[state[stored]] + self.length[good][stored, None]*(position-.5)
        continuation[stored] = self.tail[state[stored]]
        total = first+self.total[continuation]
        variance = first**2+self.variance[continuation]
        if np.any(delay < -1e-10) or np.any(total > delay+1e-9):
            raise ValueError("source/storage quadrature extends beyond its path")
        return {"delay": np.maximum(delay.ravel(), 0), "weights": np.repeat(weights/points, points),
                "first": first.ravel(), "continuation": continuation.ravel(),
                "storage_total": total.ravel(), "storage_variance": variance.ravel()}

    def descriptors(self):
        x = self.inputs()
        w, delay = x["weights"], x["delay"]
        mean = w@delay
        return {"n_sources": len(w), "n_reaches": len(self.comids),
                "n_storage_reaches": int(self.is_storage.sum()),
                "n_waterbody_ids": len(np.unique(self.waterbody[self.is_storage])),
                "n_secondary_successors": int(self.secondary.sum()),
                "max_route_error_km": self.max_route_error_km,
                "mean_delay": mean, "path_variance": w@((delay-mean)**2),
                "mean_storage_length": w@x["storage_total"],
                "storage_mean_share": (w@x["storage_total"])/mean,
                "storage_exposed_flow_share": w@(x["storage_total"] > 0),
                "serial_storage_variance": w@x["storage_variance"],
                "lumped_storage_variance": w@(x["storage_total"]**2),
                "mean_serial_elements": w@self.depth[self.midpoint_state[self.area > 0]]}


def storage_transfer(paths, omega, fraction):
    """Centred serial responses; deterministic translation is handled separately."""
    if not 0 <= fraction <= 1:
        raise ValueError("storage allocation must be in [0,1]")
    omega = np.asarray(omega, float)
    value = np.ones((len(paths.tau), len(omega)), complex)
    for nodes in paths.layers:
        tau = fraction*paths.tau[nodes, None]
        factor = np.exp(1j*tau*omega)/(1+1j*tau*omega)
        value[nodes] = value[paths.tail[nodes]]*factor
    return value


def network_responses(paths, *, fractions=(0., .25, .5, 1.), sigmas=(.075, .15, .30),
                      points=1, dt=.0025, keep_curves=False):
    """Exact pathwise serial kernels, analytic forcing, band-limited inversion.

    Fourier summation uses no reach-delay quantization. All fractions use the
    same long domain. The Gaussian frequency cutoff is below 1e-12 in amplitude.
    The source forcing is two-sided; every routing kernel is strictly causal.
    """
    if dt <= 0 or not sigmas or min(sigmas) <= 0 or not fractions or min(fractions) < 0 or max(fractions) > 1:
        raise ValueError("positive resolution/input SDs and storage fractions in [0,1] required")
    x = paths.inputs(points)
    d, w = x["delay"], x["weights"]
    origin = -8*max(sigmas)
    end = d.max()+32*max(fractions)*x["storage_total"].max()+8*max(sigmas)
    size = next_fast_len(int(np.ceil((end-origin)/dt)))
    time = origin+np.arange(size)*dt
    frequencies = np.fft.rfftfreq(size, dt)*2*np.pi
    active = np.flatnonzero(frequencies <= np.sqrt(-2*np.log(1e-12))/min(sigmas))
    spectra = np.zeros((len(fractions), len(frequencies)), complex)
    for start in range(0, len(active), 48):
        idx = active[start:start+48]
        omega = frequencies[idx]
        transfers = [storage_transfer(paths, omega, f) if f else None for f in fractions]
        for first_source in range(0, len(d), 4096):
            s = slice(first_source, first_source+4096)
            phase = w[s, None]*np.exp(-1j*d[s, None]*omega)
            for j, f in enumerate(fractions):
                if not f:
                    spectra[j, idx] += phase.sum(axis=0)
                else:
                    tau = f*x["first"][s, None]
                    factor = np.exp(1j*tau*omega)/(1+1j*tau*omega)
                    spectra[j, idx] += (phase*factor*transfers[j][x["continuation"][s]]).sum(axis=0)
    rows, curves = [], []
    mean, path_variance = w@d, w@((d-w@d)**2)
    for j, f in enumerate(fractions):
        added_variance = f*f*(w@x["storage_variance"])
        for sigma in sigmas:
            forcing = sigma*np.sqrt(2*np.pi)*np.exp(-.5*(sigma*frequencies)**2)
            pulse = np.fft.irfft(spectra[j]*forcing*np.exp(1j*frequencies*origin), n=size)/dt
            minimum = pulse.min()
            if minimum < -1e-8:
                raise ValueError("Fourier response has material negative values")
            pulse = np.maximum(pulse, 0.)
            mass = pulse.sum()*dt
            numerical_mean = pulse@time*dt/mass
            numerical_variance = pulse@((time-numerical_mean)**2)*dt/mass
            cdf = np.cumsum(pulse)*dt/mass
            q10, q50, q90 = np.interp([.1, .5, .9], cdf, time)
            peak = pulse.max()
            rows.append({"storage_fraction": f, "input_sd": sigma, "source_points": points,
                "pulse_peak": peak, "peak_time": time[pulse.argmax()], "pulse_centroid": mean,
                "pulse_sd": np.sqrt(sigma*sigma+path_variance+added_variance),
                "path_variance": path_variance, "storage_variance": added_variance,
                "input_variance": sigma*sigma, "t10": q10, "t50": q50, "t90": q90,
                "duration_80": q90-q10, "anomaly_area_fraction": mass/(sigma*np.sqrt(2*np.pi)),
                "steady_gain": float(spectra[j, 0].real), "numerical_centroid": numerical_mean,
                "numerical_sd": np.sqrt(numerical_variance), "minimum_before_clip": minimum,
                "dt": dt, "domain_end": time[-1], "n_active_frequencies": len(active)})
            if keep_curves:
                # Saving every 4th sample keeps the figure product modest.
                for t, p in zip(time[::4], pulse[::4]):
                    curves.append({"storage_fraction": f, "input_sd": sigma, "source_points": points,
                                   "time": t, "outlet_anomaly": p})
    return rows, curves
