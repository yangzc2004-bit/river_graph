"""Check source-only monthly responses, visibility and matched geographic tasks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_hydrologic_activation import (
    monthly_source_panel,
    station_flow_responses,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_hydrologic_activation_v1")


def main():
    receipt = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in receipt["source_hashes"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"changed source: {path}")
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", map_location="cpu", weights_only=False)
    cells = np.load("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
    allowed = []
    for split in (142, 143, 144):
        with np.load(f"experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks/split{split}.npz") as z:
            allowed.extend(z["train"])
    np.testing.assert_array_equal(cells, np.unique(allowed))
    metadata = pd.read_csv("experiments/phase4_transfer/doc_river_source_placement_v1/analysis/station_doc_source_panel.csv",
                           dtype={"station": str, "huc2": str, "huc4": str, "huc_cd": str})
    monthly = monthly_source_panel(dataset, cells, metadata)
    saved = pd.read_parquet(ROOT/"analysis/monthly_source_panel.parquet")
    pd.testing.assert_frame_equal(monthly, saved)
    changed = dataset.copy()
    target = np.asarray(dataset["y"]).copy()
    hidden = np.ones(target.size, bool)
    hidden[cells] = False
    target.ravel()[hidden] = 654321
    changed["y"] = target
    changed_x = np.asarray(dataset["x"]).copy()
    changed_x[~np.asarray(dataset["x_mask"], bool)] = 654321
    changed["x"] = changed_x
    pd.testing.assert_frame_equal(monthly, monthly_source_panel(changed, cells, metadata))
    residuals = pd.read_parquet(ROOT/"analysis/within_station_residuals.parquet")
    fits = pd.read_csv(ROOT/"analysis/station_response_fits.csv", dtype={"station": str})
    counts = {}
    for population, panel, temperature in (
        ("all_source_months", monthly, False), ("observed_temperature", monthly, True),
        ("source_months_since_2009", monthly[monthly.month.dt.year.ge(2009)], False)):
        rebuilt, summary = station_flow_responses(panel, temperature=temperature)
        rebuilt["population"] = population
        pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True),
                                      residuals[residuals.population.eq(population)].reset_index(drop=True))
        expected = fits[fits.population.eq(population)].set_index("station")
        actual = summary.set_index("station")
        for name in ("cq_linear", "cq_quadratic", "interquartile_response"):
            np.testing.assert_allclose(actual[name], expected[name], equal_nan=True)
        # Independent full-model C-Q check, not the residualization implementation.
        max_difference = 0.
        for station, s in rebuilt.groupby("station"):
            months, years = s.month.dt.month.to_numpy(), s.month.dt.year.to_numpy()
            fractional = years+(months-1)/12
            q = np.log1p(s.discharge_cfs.to_numpy())
            center = q-np.median(q)
            columns = [np.ones(len(s)), np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12),
                       (fractional-fractional.mean())/10]
            if temperature:
                columns.append(s.temperature_c.to_numpy()-s.temperature_c.mean())
            design = np.column_stack([*columns, center, center**2])
            coefficient = np.linalg.lstsq(design, np.log1p(s.doc), rcond=None)[0]
            difference = abs(coefficient[-2]-actual.loc[station, "cq_linear"])
            max_difference = max(max_difference, difference)
        if max_difference > 1e-7:
            raise ValueError(f"station regression mismatch: {population}")
        counts[population] = {"stations": int(summary.response_status.eq("included").sum()),
                              "monthly_pairs": len(rebuilt), "independent_OLS_max_difference": max_difference}
    if not residuals.flow_usable.all() or not residuals.landscape_included.all():
        raise ValueError("invalid or excluded cells entered within-station responses")
    predictions = pd.read_csv(ROOT/"analysis/huc4_response_predictions.csv", dtype={"station": str, "huc4": str})
    for (population, fold), p in predictions.groupby(["population", "fold"]):
        arms = p.groupby("model").station.apply(lambda x: tuple(sorted(x)))
        if arms.nunique() != 1:
            raise ValueError(f"different stations across arms: {population}/{fold}")
    for population, p in predictions.groupby("population"):
        if p.groupby("station").fold.nunique().max() != 1 or p.groupby("huc4").fold.nunique().max() != 1:
            raise ValueError(f"geography split failure: {population}")
        if not np.isfinite(p[["cq_linear_true", "cq_linear_pred"]]).all().all():
            raise ValueError("nonfinite geographic diagnostic")
    result = {"status": "passed", "source_cells": len(cells), "populations": counts,
              "hidden_DOC_and_masked_hydro_perturbation": "unchanged monthly source panel",
              "equal_station_weight": "unit test confirms replication invariance",
              "matched_HUC4_tasks": True, "neural_model_modified": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
