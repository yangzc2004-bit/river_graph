"""Describe support transfer on reused source-validation DOC stations only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_auxiliary_chemistry_v1 import BASIS
from run_doc_chemistry_support_v1 import (
    frozen_reference,
    make_panels,
    validation_summary,
)

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_chemistry_support_v1")
COMPONENTS = ["context_pred", "ecological_memory", "neural_chemistry_pred", "tree_chemistry_pred",
              "point_pred", "tree_prior_pred"]
DESCRIPTORS = ["cell", "station", "month", "analyte", "visibility_role", "ecological_novelty",
               "upstream_support", "ph_available", "ec_available", "aux_available", "doc_observed"]


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def selected_base(name, k, full, adapters, mixers, old_adapters, old_mixers):
    if "integrated" in name:
        state = (old_mixers[f"point_integrated_{BASIS}"] if name == "point_integrated_legacy"
                 else mixers[name].to_dict())
        model = SupportAwareResidualTransfer.from_dict(state)
        column = "point_pred" if name == "point_integrated_legacy" else "neural_chemistry_pred"
        base = model.selected_base(full.context_pred.to_numpy(), full[column].to_numpy(),
                                   full.ecological_memory.to_numpy(), k=k)
        choice = model.to_dict()["selection_by_k"][str(k)]
    else:
        pipe = "point" if name == "point_legacy" else "tree_prior" if name == "tree_prior_legacy" else name.rsplit("_", 1)[0]
        if pipe.endswith("_chemistry") and name.endswith("chemistry_aug"):
            pipe = pipe.removesuffix("_chemistry")
        if name.startswith("neural_chemistry_"):
            pipe = "neural_chemistry"
        elif name.startswith("tree_chemistry_"):
            pipe = "tree_chemistry"
        base = full[f"{pipe}_pred"].to_numpy()
        state = (old_adapters[f"{pipe}_{BASIS}"] if pipe in ("point", "tree_prior") else adapters[name].to_dict())
        choice = {**state["selection_by_k"][str(k)], "gamma": 0.}
    return base, choice


def diagnose_run(run):
    config = json.loads((run / "config.json").read_text())
    prior = Path(config["prior_run"])
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role] for role in ("train", "val", "test", "context")}
    dataset = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    months = dataset["y"].shape[1]
    # Only source-validation values enter this diagnostic. All other label
    # positions are replaced with NaN before any predictor/metric function.
    truth = np.full(np.prod(dataset["y"].shape), np.nan)
    truth[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    del dataset
    full = pd.read_parquet(run / "full_grid.parquet", columns=DESCRIPTORS + COMPONENTS)
    assert np.array_equal(full.cell.to_numpy(), np.arange(len(full)))
    with np.load(run / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name] for name in ("legacy", "masks_aug", "chemistry_aug")}
        chemistry, active = saved["chemistry_chemical"], saved["active"]
    state_adapters = json.loads((run / "adapters.json").read_text())
    state_mixers = json.loads((run / "mixers.json").read_text())
    adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in state_adapters.items()}
    mixers = {name: SupportAwareResidualTransfer.from_dict(state) for name, state in state_mixers.items()}
    old_adapters = json.loads((prior / "adapters.json").read_text())
    old_mixers = json.loads((prior / "mixers.json").read_text())
    bases = {name: full[f"{name}_pred"].to_numpy() for name in ("neural_chemistry", "tree_chemistry")}
    panels = make_panels(full, bases, shapes, adapters, mixers, old_adapters, old_mixers,
                        truth, split, months, active, role="val")
    reproduced = validation_summary(panels, truth)
    saved = pd.read_csv(run / "source_validation.csv")
    checked = reproduced.merge(saved, on=["model_name", "k"], suffixes=("", "_saved"), validate="one_to_one")
    np.testing.assert_allclose(checked.mae, checked.mae_saved, rtol=0, atol=1e-12)
    extras = []
    for k in (0, 1, 3, 5):
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        for pipe in ("point", "tree_prior"):
            row = full.iloc[query][DESCRIPTORS].copy()
            row["model_name"], row["k"] = f"{pipe}_legacy", k
            for column, values in frozen_reference(pipe, full, shapes["legacy"], old_adapters, old_mixers,
                                                  truth, support, query, k).items():
                row[column] = values
            extras.append(row)
    panels = pd.concat([panels, *extras], ignore_index=True)
    summaries, stations, choices = [], [], []
    for (name, k), group in panels.groupby(["model_name", "k"], sort=True):
        cells = group.cell.to_numpy()
        errors = truth[cells] - group.y_pred.to_numpy()
        log_error = np.log1p(truth[cells]) - np.log1p(group.y_pred.to_numpy())
        summaries.append({"model_name": name, "k": k, "n": len(cells), "mae": np.abs(errors).mean(),
                          "mean_residual": errors.mean(), "log_mae": np.abs(log_error).mean(),
                          "mean_log_residual": log_error.mean()})
    support, query = support_query_cells(split, target_role="val", k=5, n_months=months)
    names = ["point_legacy", "point_integrated_legacy", "tree_prior_legacy"]
    names += [f"{pipe}_{variant}" for pipe in ("neural_chemistry", "neural_chemistry_integrated", "tree_chemistry")
              for variant in ("legacy", "chemistry_aug")]
    for name in names:
        base, choice = selected_base(name, 5, full, adapters, mixers, old_adapters, old_mixers)
        _, choice0 = selected_base(name, 0, full, adapters, mixers, old_adapters, old_mixers)
        choices.append({"model_name": name, "k": 5, "gamma_k0": choice0["gamma"], **choice})
        predictions = panels[panels.model_name.eq(name) & panels.k.eq(5)].set_index("cell")
        basis = shapes["chemistry_aug" if name.endswith("chemistry_aug") else "legacy"]
        for station in np.unique(query // months):
            s, q = support[support // months == station], query[query // months == station]
            q = q[active[q]]
            if not len(q):
                continue
            sr = np.log1p(truth[s]) - np.log1p(base[s])
            qr = np.log1p(truth[q]) - np.log1p(base[q])
            centered = basis[s] - basis[s].mean(0)
            ridge = choice["ridge_strength"]
            coefficient = (np.zeros(basis.shape[1]) if ridge == "infinity" else
                           np.linalg.solve(centered.T @ centered / 5 + float(ridge)*np.eye(basis.shape[1]),
                                           centered.T @ (sr-sr.mean()) / 5))
            shape_correction = (basis[q]-basis[s].mean(0)) @ coefficient
            available_s, available_q = s[active[s]], q[active[q]]
            if len(available_s) and len(available_q):
                distances = np.linalg.norm(chemistry[available_q, None]-chemistry[None, available_s], axis=-1)
                nearest = distances.min(1)
                similarity = correlation(nearest, np.abs(qr[active[q]]-sr.mean()))
            else:
                nearest, similarity = np.asarray([np.nan]), np.nan
            p = predictions.loc[q, "y_pred"].to_numpy()
            stations.append({"model_name": name, "station_index": station, "station": str(full.iloc[q[0]].station),
                "n_query": len(q), "support_active": int(active[s].sum()), "query_active": int(active[q].sum()),
                "support_log_residual_mean": sr.mean(), "query_log_residual_mean": qr.mean(),
                "support_query_mean_gap": qr.mean()-sr.mean(), "support_log_residual_sd": sr.std(),
                "query_log_residual_sd": qr.std(), "base_mae": np.abs(truth[q]-base[q]).mean(),
                "adapted_mae": np.abs(truth[q]-p).mean(), "adapted_mean_residual": (truth[q]-p).mean(),
                "adapted_mean_log_residual": (np.log1p(truth[q])-np.log1p(p)).mean(),
                "level_correction": choice["alpha"]*sr.mean(), "shape_coefficient_norm": np.linalg.norm(coefficient),
                "shape_correction_mean": shape_correction.mean(),
                "shape_correction_abs_mean": np.abs(shape_correction).mean(),
                "shape_query_residual_correlation": correlation(shape_correction, qr-qr.mean()),
                "nearest_support_chemical_distance_mean": nearest.mean(),
                "nearest_support_chemical_distance_q90": np.quantile(nearest, .9),
                "distance_residual_mismatch_correlation": similarity})
    for rows in (summaries, stations, choices):
        for row in rows:
            row.update(split_seed=config["split_seed"], seed=config["seed"])
    sources = [run / name for name in ("config.json", "complete.json", "full_grid.parquet", "representations.npz",
               "adapters.json", "mixers.json", "source_validation.csv", "basis_definition.json")]
    sources += [prior / name for name in ("adapters.json", "mixers.json")]
    sources += [Path(config[key]) for key in ("dataset_path", "mask_path")]
    return summaries, stations, choices, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    runs = sorted(path.parent for path in (args.root / "runs").glob("*/complete.json"))
    if not runs or (len(runs) != 9 and not args.allow_partial):
        raise ValueError("Requires nine completed packages, or explicit --allow-partial")
    summaries, stations, choices, sources = [], [], [], []
    for run in runs:
        a, b, c, d = diagnose_run(run)
        summaries.extend(a); stations.extend(b); choices.extend(c); sources.extend(d)
    out = args.root / "diagnostics"
    out.mkdir(exist_ok=True)
    frames = {"source_validation_metrics": pd.DataFrame(summaries), "support_query_stations": pd.DataFrame(stations),
              "calibration_choices": pd.DataFrame(choices)}
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    averages = frames["source_validation_metrics"].groupby(["model_name", "k", "split_seed"])[
        ["mae", "mean_residual", "log_mae", "mean_log_residual"]].mean().groupby(["model_name", "k"]).mean().reset_index()
    averages.to_csv(out / "source_validation_mean.csv", index=False)
    station = frames["support_query_stations"].groupby(["model_name", "split_seed", "station_index"]).mean(numeric_only=True).reset_index()
    diag = []
    for name, group in station.groupby("model_name"):
        diag.append({"model_name": name, "station_partition_pairs": len(group),
            "support_query_log_mean_correlation": correlation(group.support_log_residual_mean, group.query_log_residual_mean),
            "opposite_residual_mean_sign_fraction": float((group.support_log_residual_mean*group.query_log_residual_mean < 0).mean()),
            "absolute_mean_mismatch": group.support_query_mean_gap.abs().mean(),
            "shape_query_correlation_median": group.shape_query_residual_correlation.median(),
            "nearest_chemical_distance_mean": group.nearest_support_chemical_distance_mean.mean(),
            "distance_mismatch_correlation_median": group.distance_residual_mismatch_correlation.median(),
            "station_fraction_worsened_by_adaptation": float((group.adapted_mae > group.base_mae).mean()),
            "support_fully_aux_active_fraction": float(group.support_active.eq(5).mean())})
    diagnostics = pd.DataFrame(diag)
    diagnostics.to_csv(out / "support_query_summary.csv", index=False)
    report = ["# Source-validation support-transfer diagnosis", "",
        (f"Reconstructed {len(runs)} completed packages; all saved source-validation MAEs reproduce within 1e-12. "
        "Only validation station DOC labels were used. Target prediction files were not read."), "",
        ("This is a descriptive diagnosis on reused selection data, not new confirmation. "
        "Means average seeds within partition and then partitions equally. Station summaries average seeds first; "
        "stations repeated across partitions remain descriptive station-partition pairs. "
        "MAE tables use all fixed validation queries; support-transfer station diagnostics use auxiliary-active queries."), "",
        "## Source-validation MAE", "", "| Model | K0 | K5 |", "|---|---:|---:|"]
    for name in diagnostics.model_name:
        values = averages[averages.model_name.eq(name)].set_index("k")
        report.append(f"| {name} | {values.loc[0, 'mae']:.6f} | {values.loc[5, 'mae']:.6f} |")
    lookup = averages.set_index(["model_name", "k"])
    parent = "point_integrated_legacy"
    legacy = "neural_chemistry_integrated_legacy"
    augmented = "neural_chemistry_integrated_chemistry_aug"
    gain0 = lookup.loc[(parent, 0), "mae"]-lookup.loc[(legacy, 0), "mae"]
    gain5 = lookup.loc[(parent, 5), "mae"]-lookup.loc[(legacy, 5), "mae"]
    aug5 = lookup.loc[(legacy, 5), "mae"]-lookup.loc[(augmented, 5), "mae"]
    c = frames["calibration_choices"]
    c = c[c.model_name.eq(legacy)]
    report += ["", "## What changes with five support measurements", "",
        (f"The integrated chemical decoder's validation advantage over the general point model is {gain0:.6f} mg/L at K0 "
         f"and {gain5:.6f} mg/L at K5. Chemical coordinates add {aug5:.6f} mg/L of K5 improvement over the legacy coordinates. "
         "These contrasts use identical fixed validation queries; they are not held-out confirmation."), "",
        (f"The integrated decoder uses nonzero ecological mixing in {int(c.gamma_k0.ne(0).sum())}/{len(c)} packages at K0 "
         f"and {int(c.gamma.ne(0).sum())}/{len(c)} at K5. Increasing gamma reduces the share of the temporal/chemical expert "
         "before station calibration. Accordingly, a smaller K5 decoder contrast can reflect both support recalibration "
         "and changes in the ecological mixture, rather than failure of the chemistry inputs alone.")]
    report += ["", "## Support-query residual transfer", "",
        ("Positive residual means underprediction. The adapter transfers a shrunk support mean plus a linear ridge shape. "
        "A chemistry correction common to support and query can be partly cancelled by the mean-residual recalibration: "
        "its log-scale contrast includes query correction minus alpha times mean support correction. "
        "That is an accounting property, not proof of why a performance contrast changes."), "",
        "| Model | Support/query mean correlation | Opposite signs | Mean absolute mean mismatch | Median shape/query correlation | Stations worsened |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in diagnostics.itertuples():
        report.append(f"| {row.model_name} | {row.support_query_log_mean_correlation:.3f} | "
            f"{row.opposite_residual_mean_sign_fraction:.1%} | {row.absolute_mean_mismatch:.3f} | "
            f"{row.shape_query_correlation_median:.3f} | {row.station_fraction_worsened_by_adaptation:.1%} |")
    report += ["", ("Chemical distance uses the frozen two-dimensional whitened chemical representation, "
        "only auxiliary-active support/query dates, and nearest same-station support. "
        "It is not a physical chemical distance or a causal estimate. The station table records gaps, "
        "support coverage, shape magnitude and residual signs. No kernel, bandwidth or new predictive operator was fitted."), ""]
    specific = diagnostics.set_index("model_name")
    old, new = specific.loc[legacy], specific.loc[augmented]
    report += ["## Interpretation and one next operator hypothesis", "",
        (f"Support and query station-mean log residuals correlate at {old.support_query_log_mean_correlation:.3f}; "
         f"their signs differ for {old.opposite_residual_mean_sign_fraction:.1%} of station-partition pairs. "
         "The useful mean correction should therefore be retained. The weaker component is within-station shape: "
         f"median shape/query residual correlation rises from {old.shape_query_correlation_median:.3f} to "
         f"{new.shape_query_correlation_median:.3f} with chemical coordinates."), "",
        (f"Chemical distance is only weakly associated with support-mean residual mismatch "
         f"(median within-station correlation {new.distance_mismatch_correlation_median:.3f}). "
         "These diagnostics do not justify a distance-based reliability gate or a claim that distant chemical states cause failure."), "",
        ("If the augmented linear basis does not generalize, a focused operator test would retain the current base and shrunk "
         "station mean but replace the chemical linear extrapolation with a bounded, similarity-weighted interpolation of "
         "centered support residuals. Use the same frozen two chemical coordinates, a bandwidth derived from source-station "
         "chemical distances, and the matched availability-only control. This tests nonlinear support-to-query transfer "
         "without another neural architecture or loss sweep. It is a hypothesis motivated by weak shape transfer, not an "
         "established distance mechanism or an authorization to fit a new model."), ""]
    (out / "source_validation.md").write_text("\n".join(report))
    sources += [Path(__file__), Path("scripts/run_doc_chemistry_support_v1.py"),
                Path("src/river_graph/models/support_shape_adapter.py"),
                Path("src/river_graph/models/support_aware_residual_transfer.py")]
    (out / "source_validation_sources.json").write_text(json.dumps({"scope": "source_validation_only",
        "complete": len(runs) == 9, "sources": {str(path): sha256_file(path) for path in sorted(set(sources))}}, indent=2)+"\n")
    print(averages[averages.k.isin([0, 5])].to_string(index=False))


if __name__ == "__main__":
    main()
