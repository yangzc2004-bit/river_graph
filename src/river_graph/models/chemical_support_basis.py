"""Source-fitted chemical support coordinates from a frozen nonlinear head.

Within-source-station centering identifies directions of chemical variation.
Inference instead uses one source-global mean: no target-station timeline or
future auxiliary measurement determines the current coordinate. This distinct
centering gauge is intentional; the support adapter centers support pairs.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead

PHI_DIM = 8
N_COMPONENTS = 2
PHI_BATCH_SIZE = 512


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _auxiliary(value):
    array = _array(value)
    if array.ndim != 3 or min(array.shape[:2]) < 1 or array.shape[2] != 4:
        raise ValueError("auxiliary inputs must be [station,month,4]")
    return array


class ChemicalSupportBasis:
    """Two chemical PCs with source-only whitening and row-local application.

    The head must be a fitted ``NonlinearChemistryHead``. Its selected phi and
    last four source standardization parameters are copied, never retrained.
    ``mode`` is fixed by ``fit``. Masks mode zeroes chemical values before the
    same frozen normalization, while retaining actual availability flags.
    """

    def __init__(self, head, eigenvalue_floor=1e-8):
        if not isinstance(head, NonlinearChemistryHead) or not head.fitted_:
            raise ValueError("a fitted NonlinearChemistryHead is required")
        if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be finite and positive")
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.phi_weight_ = head.head.phi.weight.detach().cpu().numpy().copy()
        self.phi_bias_ = head.head.phi.bias.detach().cpu().numpy().copy()
        self.auxiliary_mean_ = head.feature_mean_[-4:].detach().cpu().numpy().copy()
        self.auxiliary_scale_ = head.feature_scale_[-4:].detach().cpu().numpy().copy()
        self.fitted_ = False

    def _phi(self, auxiliary, mode):
        if mode not in ("chemistry", "masks"):
            raise ValueError("chemical basis mode must be chemistry or masks")
        array = _auxiliary(auxiliary).astype(np.float64, copy=True)
        if not np.isfinite(array).all() or not np.isin(array[..., 2:], (0, 1)).all():
            raise ValueError("auxiliary inputs must be finite with binary visibility flags")
        active = np.any(array[..., 2:] != 0, axis=-1)
        # Missing numeric placeholders never become chemical information.
        array[..., :2] = np.where(array[..., 2:] != 0, array[..., :2], 0)
        if mode == "masks":
            array[..., :2] = 0
        flat = array.reshape(-1, 4)
        result = np.empty((len(flat), PHI_DIM), dtype=np.float64)
        weight = torch.as_tensor(self.phi_weight_, dtype=torch.float64)
        bias = torch.as_tensor(self.phi_bias_, dtype=torch.float64)
        with torch.inference_mode():
            for start in range(0, len(flat), PHI_BATCH_SIZE):
                standardized = torch.as_tensor(
                    (flat[start:start + PHI_BATCH_SIZE] - self.auxiliary_mean_) / self.auxiliary_scale_,
                    dtype=torch.float64)
                result[start:start + PHI_BATCH_SIZE] = F.silu(F.linear(standardized, weight, bias)).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite frozen chemistry coordinates")
        return result.reshape(*array.shape[:2], PHI_DIM), active

    def fit(self, auxiliary, source_station_indices, *, source_role, mode="chemistry"):
        if source_role != "source_training":
            raise ValueError("chemical basis fitting requires source_training stations")
        array = _auxiliary(auxiliary)
        stations = _array(source_station_indices)
        if (stations.ndim != 1 or not len(stations) or stations.dtype.kind not in "iu"
                or len(np.unique(stations)) != len(stations)
                or (stations < 0).any() or (stations >= array.shape[0]).any()):
            raise ValueError("source_station_indices must be unique valid integer station indices")
        # Sort for deterministic accumulation independent of caller order.
        stations = np.sort(stations.astype(np.int64))
        phi, active = self._phi(array[stations], mode)
        station_means = np.zeros((len(stations), PHI_DIM), dtype=np.float64)
        station_counts = active.sum(1).astype(np.int64)
        covariance_sum = np.zeros((PHI_DIM, PHI_DIM), dtype=np.float64)
        global_sum = np.zeros(PHI_DIM, dtype=np.float64)
        for i, mask in enumerate(active):
            selected = phi[i, mask]
            if not len(selected):
                continue
            station_means[i] = selected.mean(0)
            centered = selected - station_means[i]
            covariance_sum += centered.T @ centered
            global_sum += selected.sum(0)
        count = int(station_counts.sum())
        covariance = covariance_sum / max(count, 1)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(-values, kind="stable")[:N_COMPONENTS]
        components = vectors[:, order].T.copy()
        for vector in components:
            if vector[np.argmax(np.abs(vector))] < 0:
                vector *= -1
        self.mode_, self.source_role_ = mode, source_role
        self.source_station_indices_ = stations
        self.source_active_counts_ = station_counts
        self.source_station_active_means_ = station_means
        self.n_source_active_rows_ = count
        self.source_global_mean_ = global_sum / max(count, 1)
        self.covariance_ = covariance
        self.components_ = components
        self.eigenvalues_ = np.maximum(values[order], 0)
        self.identified_ = self.eigenvalues_ > self.eigenvalue_floor
        self.whitening_scale_ = np.sqrt(np.maximum(self.eigenvalues_, self.eigenvalue_floor))
        self.fitted_ = True
        return self

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("fit or restore the source chemical basis first")

    def transform(self, auxiliary, *, mode=None):
        self._require_fitted()
        if mode is not None and mode != self.mode_:
            raise ValueError("projection mode must match the fitted mode")
        phi, active = self._phi(auxiliary, self.mode_)
        basis = ((phi - self.source_global_mean_) @ self.components_.T) / self.whitening_scale_
        basis[..., ~self.identified_] = 0
        basis[~active] = 0
        if not np.isfinite(basis).all():
            raise FloatingPointError("nonfinite chemical support basis")
        return basis

    def augment(self, auxiliary, legacy_basis, *, mode=None):
        """Return grid-shaped chemical2 and legacy2+chemical2 coordinates.

        The legacy input may have [N,T,2] or flat [N*T,2] shape. Its values are
        copied unchanged. Only chemical coordinates are zero at absent-auxiliary
        cells; any final-prediction fallback remains the runner's responsibility.
        """
        chemical = self.transform(auxiliary, mode=mode)
        legacy = _array(legacy_basis).astype(np.float64, copy=False)
        shape = chemical.shape
        if legacy.shape == (shape[0] * shape[1], 2):
            legacy = legacy.reshape(shape)
        if legacy.shape != shape or not np.isfinite(legacy).all():
            raise ValueError("legacy basis must be finite aligned [N,T,2] or [N*T,2]")
        active = np.any(_array(auxiliary)[..., 2:] != 0, axis=-1)
        return {"chemical_basis": chemical, "augmented_basis": np.concatenate((legacy, chemical), axis=-1),
                "active": active, "source_fit": self.to_dict()}

    def to_dict(self):
        self._require_fitted()
        return {"model_class": "ChemicalSupportBasis", "schema_version": 1,
                "n_components": N_COMPONENTS, "phi_dim": PHI_DIM, "mode": self.mode_,
                "source_role": self.source_role_, "eigenvalue_floor": self.eigenvalue_floor,
                "phi_batch_size": PHI_BATCH_SIZE,
                "definition": {
                    "phi": "SiLU(frozen Linear(4,8)) after frozen last4 source standardization",
                    "fit_rows": "all auxiliary-active months at explicit source stations; no DOC label/mask filter",
                    "fit_center": "each source station's active-month phi mean",
                    "covariance_denominator": "total active source station-month rows",
                    "projection_center": "one pooled source-global active phi mean; no target centering",
                    "whitening": "sqrt(max(eigenvalue,eigenvalue_floor)); components at/below floor are zero",
                    "eigen_sign": "largest absolute loading positive; first index breaks a loading tie",
                    "absent_auxiliary": "chemical coordinates zero; legacy coordinates unchanged",
                    "target_label_dependency": "none",
                    "centering_gauges": "within-source covariance and source-global projection are intentionally distinct"},
                "frozen_phi_weight": self.phi_weight_.tolist(), "frozen_phi_bias": self.phi_bias_.tolist(),
                "auxiliary_mean": self.auxiliary_mean_.tolist(), "auxiliary_scale": self.auxiliary_scale_.tolist(),
                "source_station_indices": self.source_station_indices_.tolist(),
                "source_active_counts": self.source_active_counts_.tolist(),
                "source_station_active_means": self.source_station_active_means_.tolist(),
                "n_source_stations": len(self.source_station_indices_),
                "n_source_active_stations": int((self.source_active_counts_ > 0).sum()),
                "n_source_active_rows": self.n_source_active_rows_,
                "source_global_mean": self.source_global_mean_.tolist(), "covariance": self.covariance_.tolist(),
                "components": self.components_.tolist(), "eigenvalues": self.eigenvalues_.tolist(),
                "whitening_scale": self.whitening_scale_.tolist(), "identified": self.identified_.tolist(),
                "identified_components": int(self.identified_.sum())}

    @classmethod
    def from_dict(cls, state):
        state = copy.deepcopy(state)
        if (state.get("model_class") != "ChemicalSupportBasis" or state.get("schema_version") != 1
                or state.get("n_components") != N_COMPONENTS or state.get("phi_dim") != PHI_DIM
                or state.get("phi_batch_size") != PHI_BATCH_SIZE
                or state.get("source_role") != "source_training" or state.get("mode") not in ("chemistry", "masks")):
            raise ValueError("unsupported chemical support basis state")
        obj = cls.__new__(cls)
        obj.eigenvalue_floor = float(state["eigenvalue_floor"])
        if not np.isfinite(obj.eigenvalue_floor) or obj.eigenvalue_floor <= 0:
            raise ValueError("invalid eigenvalue floor")
        stations = np.asarray(state["source_station_indices"])
        if (stations.ndim != 1 or not len(stations) or stations.dtype.kind not in "iu"
                or (stations < 0).any() or np.any(np.diff(stations) <= 0)):
            raise ValueError("invalid saved source station indices")
        obj.source_station_indices_ = stations.astype(np.int64)
        obj.mode_, obj.source_role_ = state["mode"], state["source_role"]
        specifications = (
            ("phi_weight_", "frozen_phi_weight", (PHI_DIM, 4)), ("phi_bias_", "frozen_phi_bias", (PHI_DIM,)),
            ("auxiliary_mean_", "auxiliary_mean", (4,)), ("auxiliary_scale_", "auxiliary_scale", (4,)),
            ("source_station_active_means_", "source_station_active_means", (len(stations), PHI_DIM)),
            ("source_global_mean_", "source_global_mean", (PHI_DIM,)), ("covariance_", "covariance", (PHI_DIM, PHI_DIM)),
            ("components_", "components", (N_COMPONENTS, PHI_DIM)), ("eigenvalues_", "eigenvalues", (N_COMPONENTS,)),
            ("whitening_scale_", "whitening_scale", (N_COMPONENTS,)))
        for attribute, key, shape in specifications:
            array = np.asarray(state[key], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid saved {key}")
            setattr(obj, attribute, array)
        counts = np.asarray(state["source_active_counts"])
        if counts.shape != (len(stations),) or counts.dtype.kind not in "iu" or (counts < 0).any():
            raise ValueError("invalid active source counts")
        obj.source_active_counts_ = counts.astype(np.int64)
        obj.n_source_active_rows_ = int(counts.sum())
        obj.identified_ = obj.eigenvalues_ > obj.eigenvalue_floor
        if ((obj.auxiliary_scale_ <= 0).any() or (obj.eigenvalues_ < 0).any()
                or not np.array_equal(obj.whitening_scale_, np.sqrt(np.maximum(obj.eigenvalues_, obj.eigenvalue_floor)))
                or state["n_source_active_rows"] != obj.n_source_active_rows_
                or not np.array_equal(np.asarray(state["identified"]), obj.identified_)
                or not np.allclose(obj.components_ @ obj.components_.T, np.eye(N_COMPONENTS), atol=1e-10)):
            raise ValueError("invalid chemical whitening/identification state")
        obj.fitted_ = True
        return obj
