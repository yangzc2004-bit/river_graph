"""Independent scalar replay of the archived mixing and geometry calculations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4_transfer/doc_river_measured_confluences_v1"
RAW = ROOT / "data/raw/river_measured_confluences_v1"


def main():
    sources = json.loads((OUT / "analysis_sources.json").read_text())
    for section in ["inputs", "code"]:
        for name, digest in sources[section].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Changed {section}: {name}")
    for name, digest in sources["outputs"].items():
        if sha256_file(OUT / name) != digest:
            raise ValueError(f"Changed result: {name}")
    raw = list(csv.DictReader((RAW / "Plontetal_WRR_database.csv").open()))
    actual = pd.read_csv(OUT / "analysis/point_ledger.csv").set_index(
        ["season", "location", "transect.location"])
    errors, groups = [], defaultdict(list)
    for row in raw:
        qm, qt, qd = (float(row[name]) for name in ["Q.Ls.main", "Q.Ls.trib", "Q.Ls"])
        cm, ct, cd = (float(row[name]) for name in
                      ["DOC.mgL.mean.main", "DOC.mgL.mean.trib", "DOC.mgL.mean"])
        incoming = qm * cm + qt * ct
        prediction = incoming / (qm + qt)
        expected = {"incoming_doc_mgs": incoming, "downstream_doc_mgs": qd * cd,
                    "doc_departure_pct": (cd - prediction) * 100 / prediction,
                    "doc_flux_discrepancy_pct": (qd * cd - incoming) * 100 / incoming,
                    "water_closure_pct": (qd - qm - qt) * 100 / (qm + qt)}
        key = (row["season"], row["location"], row["transect.location"])
        for name, value in expected.items():
            np.testing.assert_allclose(actual.loc[key, name], value, atol=1e-11, rtol=1e-11)
            errors.append(abs(actual.loc[key, name] - value))
        groups[key[:2]].append(expected)
    campaigns = pd.read_csv(OUT / "analysis/campaigns.csv").set_index(["season", "location"])
    for key, positions in groups.items():
        for name in ["incoming_doc_mgs", "doc_departure_pct", "doc_flux_discrepancy_pct"]:
            np.testing.assert_allclose(campaigns.loc[key, name], mean([p[name] for p in positions]),
                                       atol=1e-11, rtol=1e-11)
    width = defaultdict(dict)
    for row in csv.DictReader((RAW / "Plontetal_WRR_Fall2021_widthdepth_notrib.csv").open()):
        key = (f"Con-{row['location'].split('.')[0]}",
               "upstream" if row["reach"] == "1.upstream" else "downstream")
        value = float(row["wettedwidth.m"])
        if row["transect"] in width[key] and width[key][row["transect"]] != value:
            raise ValueError("Width conflict in raw transect")
        width[key][row["transect"]] = value
    geometry = pd.read_csv(OUT / "analysis/geometry_reaches.csv").set_index(["location", "reach"])
    for key, transects in width.items():
        values = list(transects.values())
        np.testing.assert_allclose(geometry.loc[key, "width_mean_m"], mean(values), atol=1e-12)
        np.testing.assert_allclose(geometry.loc[key, "width_cv_pct"],
                                   100 * stdev(values) / mean(values), atol=1e-10)
        assert geometry.loc[key, "n_width_transects"] == len(values)
    report = {"point_accounts_replayed": len(raw), "campaign_accounts_replayed": len(groups),
              "geometry_reaches_replayed": len(width), "max_scalar_account_difference": max(errors),
              "method": "csv scalar arithmetic and statistics; independent of analysis helpers",
              "source_chain": "Current inputs, analysis code and result hashes match"}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
