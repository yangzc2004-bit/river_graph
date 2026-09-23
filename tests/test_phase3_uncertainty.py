"""Phase-3 contracts (docs/paper/phase3_uncertainty_spec.md §1, §2, §3).

Hard label-perturbation contract: changing any test DOC label must leave the
full-grid prediction, uncertainty, support_count, network_distance and
ecological_novelty bit-identical. Plus provenance binding (dataset / mask /
config / runtime / phase3-spec hashes) and the frozen interval math.
"""

import json

import numpy as np
import torch

from river_graph.baselines.baselines import EcoRandomForest
from river_graph.experiments.phase3_uncertainty import (
    SPEC_PATH,
    calibrate_interval,
    calibration_cells,
    covariates,
    family_of,
    product_hashes,
    scenario_visibility,
    visibility_role,
    write_product,
)
from river_graph.models.gcn import GCNDocModel

N, T = 6, 4


def _toy_dataset(seed: int = 5) -> dict:
    g = torch.Generator().manual_seed(seed)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    return {
        "y": torch.rand(N, T, generator=g) * 5,
        "y_mask": torch.ones(N, T, dtype=torch.bool),
        "x": torch.rand(N, T, 2, generator=g),
        "x_mask": (torch.rand(N, T, 2, generator=g) > 0.2).float(),
        "months": [f"2020-{m:02d}-01" for m in range(1, T + 1)],
        "static": torch.rand(N, 2, generator=g),
        "regime": torch.randn(N, 13, generator=g),
        "edge_index": ei,
        "edge_attr": torch.randn(ei.shape[1], 6, generator=g),
        "site_no": [f"s{i}" for i in range(N)],
    }


def _toy_split() -> dict:
    return {
        "train": np.array([0, 1, 2, 4, 5, 8], dtype=np.int64),
        "val": np.array([3, 6], dtype=np.int64),
        "test": np.array([7, 10, 11], dtype=np.int64),
        "context": np.array([9], dtype=np.int64),
    }


def _perturb_test_labels(ds: dict, split: dict) -> dict:
    ds2 = {**ds, "y": ds["y"].clone()}
    for cell in np.asarray(split["test"], dtype=np.int64):
        i, j = divmod(int(cell), T)
        ds2["y"][i, j] += 10.0 + float(i)
    return ds2


def test_scenario_visibility_and_calibration_cells():
    split = _toy_split()
    for fam, want in (("E1", {0, 1, 2, 4, 5, 8}),
                      ("E2a", {0, 1, 2, 4, 5, 8}),
                      ("E3", {0, 1, 2, 4, 5, 8})):
        assert set(scenario_visibility(split, fam).tolist()) == want, fam
        cal = set(calibration_cells(split, fam).tolist())
        assert cal == want | {3, 6}, fam  # + val, never test/context
    e2b = set(scenario_visibility(split, "E2b").tolist())
    assert e2b == {0, 1, 2, 4, 5, 8, 9}
    assert set(calibration_cells(split, "E2b").tolist()) == e2b | {3, 6}
    assert 7 not in calibration_cells(split, "E2b")
    assert family_of("e1_r20_seed42") == "E1"
    assert family_of("e3_spatial_seed44") == "E3"


def test_visibility_role_labels():
    role = visibility_role(_toy_split(), N, T)
    assert role[0] == "train" and role[3] == "val"
    assert role[7] == "test" and role[9] == "context"
    assert role[23] == "unobserved"


def test_label_perturbation_gnn_fullgrid_invariant():
    """Spec §1 hard contract for the GNN tool (fit + full-grid predict)."""
    ds, split = _toy_dataset(), _toy_split()
    vis = scenario_visibility(split, "E2a")

    def run(d):
        model = GCNDocModel(architecture="transport_enc", env_groups=None,
                            env_encoder=True, edge_set="river",
                            max_epochs=1, patience=1, seed=3)
        model.fit(d, split)
        return model.predict(only_visible=vis)

    p1 = run(ds)
    p2 = run(_perturb_test_labels(ds, split))
    assert np.allclose(p1, p2, rtol=0, atol=0), (
        "test DOC labels must not change full-grid predictions"
    )


def test_label_perturbation_ecorf_and_covariates_invariant():
    ds, split = _toy_dataset(), _toy_split()
    vis = scenario_visibility(split, "E2b")
    keys = {"train", "context"}

    def run(d):
        pred = EcoRandomForest(n_estimators=8, seed=0).fit_predict_full(
            d, split, predict_visibility=keys)
        cov = covariates(d, vis)
        return pred, cov

    pred1, cov1 = run(ds)
    pred2, cov2 = run(_perturb_test_labels(ds, split))
    assert np.allclose(pred1, pred2, rtol=0, atol=0)
    for k in ("support_count", "network_distance", "ecological_novelty"):
        assert np.allclose(cov1[k], cov2[k], rtol=0, atol=0, equal_nan=True), k


def test_label_perturbation_interval_invariant():
    ds, split = _toy_dataset(), _toy_split()
    cal = calibration_cells(split, "E1")
    rng = np.random.default_rng(0)
    seeds = rng.normal(size=(5, N, T))
    y1 = np.log1p(ds["y"].numpy())
    y2 = np.log1p(_perturb_test_labels(ds, split)["y"].numpy())
    a = calibrate_interval(seeds, y1, cal)
    b = calibrate_interval(seeds, y2, cal)
    assert np.allclose(a["uncertainty"], b["uncertainty"], rtol=0, atol=0)
    assert a["qhat"] == b["qhat"]
    assert np.allclose(a["pi_lower"], b["pi_lower"], rtol=0, atol=0)


def test_interval_math_median_spread_and_finite_sample_quantile():
    seeds = np.zeros((5, 2, 2))
    seeds[:, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]  # log1p preds
    y = np.zeros((2, 2))
    y[0, 0] = 3.0  # residual 0 at the median
    y[1, 1] = 2.0
    cal = np.array([0, 3])  # both cells
    out = calibrate_interval(seeds, y, cal)
    assert np.isclose(out["prediction_median"][0, 0], np.expm1(3.0))
    assert out["uncertainty"].shape == (2, 2)
    assert out["n_cal"] == 2
    # finite-sample level: ceil((n+1)*0.9)/n = ceil(2.7)/2 = 1.5 -> clamped 1.0
    assert 0 < out["qhat"] < 1e6
    half = out["uncertainty"]
    assert np.allclose(out["pi_upper"], np.expm1(
        np.log1p(out["prediction_median"]) + half), rtol=1e-6)


def test_covariates_on_a_line_graph():
    ds = _toy_dataset()
    ds["edge_index"] = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
                                    dtype=torch.long)
    # only stations 0 and 5 have visible labels
    vis = np.array([0, 1, T * 5, T * 5 + 1], dtype=np.int64)
    cov = covariates(ds, vis)
    assert cov["support_count"][0] == 2  # its own two cells within 2 hops
    assert cov["network_distance"][0] == 0.0
    assert cov["network_distance"][1] == 1.0  # one hop from station 0
    assert cov["network_distance"][2] == 2.0
    assert cov["network_distance"][5] == 0.0  # visible itself
    assert cov["ecological_novelty"][0] > 0  # distance to the other visible
    assert np.isnan(cov["ecological_novelty"]).sum() == 0


def test_product_binds_all_provenance_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr("river_graph.experiments.phase3_uncertainty.SPEC_PATH",
                        SPEC_PATH)
    d1 = tmp_path / "ds.pt"
    d2 = tmp_path / "m.npz"
    d1.write_bytes(b"dataset-bytes")
    d2.write_bytes(b"mask-bytes")
    prov = product_hashes(str(d1), str(d2))
    for key in ("dataset_sha256", "mask_sha256", "phase3_spec_sha256",
                "runtime_code_snapshot_sha256"):
        assert prov.get(key), key
    out = tmp_path / "full_grid" / "P3_H2X__e2b_partial.parquet"
    write_product(out, {"station_id": ["s0"], "month": ["2020-01"],
                        "prediction_median": [1.0]}, prov)
    import pandas as pd

    df = pd.read_parquet(out)
    bound = json.loads(df["provenance_hashes"].iloc[0])
    assert bound == prov
    meta = json.loads(out.with_suffix(".json").read_text())
    assert "not conformal" in meta["calibration"]
    assert prov["phase3_spec_sha256"] == __import__(
        "hashlib").sha256(SPEC_PATH.read_bytes()).hexdigest()
