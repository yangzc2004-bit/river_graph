"""Run the fixed, compact independent real-DOC morphology comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_neon_form import (
    ARMS,
    CONTEXT,
    OUTCOMES,
    calendar_endpoints,
    gain_summary,
    heldout_predictions,
    matched_shape_pairs,
    transfer_form_classes,
    upstream_overlap_groups,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_neon_form_validation_v1")
RAW = Path("data/raw/river_neon_form_validation_v1")
OLD = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis")
EVIDENCE = Path("experiments/phase4_transfer/doc_river_wholeform_evidence_v1/analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT / "analysis"
    records = json.loads((out / "geometry_records.json").read_text())
    inventory = pd.read_csv(EVIDENCE / "neon_site_inventory.csv")
    inventory = inventory[inventory.waterbody_scope.eq("stream_or_river")]
    if set(inventory.site) != set(records):
        raise ValueError("acquisition must account for all 27 sites before effect fitting")
    forms = pd.read_csv(out / "network_forms.csv")
    eligible = forms[forms.form_eligibility.eq("eligible")].reset_index(drop=True)
    old_features = pd.read_csv(OLD / "features.csv")
    eligible = transfer_form_classes(eligible, old_features, pd.read_csv(OLD / "centroids.csv"))
    memberships = {}
    for site in eligible.site:
        with np.load(RAW / "members" / f"{site}.npz") as z:
            memberships[site] = set(z["comids"].tolist())
    groups, overlaps = upstream_overlap_groups(memberships)
    overlaps.to_csv(out / "network_overlaps.csv", index=False)
    eligible["network_group"] = eligible.site.map(groups)
    eligible.to_csv(out / "transferred_geometry_classes.csv", index=False)
    cat = pd.DataFrame(json.loads((RAW / "streamcat_ws_coverage.json").read_text())["items"])
    if cat.comid.duplicated().any():
        raise ValueError("duplicate watershed ecology records")
    cat["wetland"] = cat.pctwdwet2019ws + cat.pcthbwet2019ws
    cat["forest"] = cat.pctconif2019ws + cat.pctdecid2019ws + cat.pctmxfst2019ws
    cat["agriculture"] = cat.pctcrop2019ws + cat.pcthay2019ws
    cat["urban"] = cat.pcturbhi2019ws + cat.pcturbmd2019ws + cat.pcturblo2019ws + cat.pcturbop2019ws
    cat["log_precipitation"] = np.log(cat.precip9120ws)
    cat["climate_temperature"] = cat.tmean9120ws
    cat["ecology_coverage_eligible"] = cat.nlcd2019_wspctfull.ge(95) & cat.prism_1991_2020_wspctfull.ge(95)
    ecology_columns = ["comid", "wetland", "forest", "agriculture", "urban", "log_precipitation",
                       "climate_temperature", "ecology_coverage_eligible", "nlcd2019_wspctfull", "prism_1991_2020_wspctfull"]
    eligible = eligible.merge(cat[ecology_columns], on="comid", how="left", validate="many_to_one")
    eligible = eligible.merge(inventory[["site", "domain"]], on="site", validate="one_to_one")
    occasions = pd.read_csv(EVIDENCE / "neon_doc_occasions.csv")
    primary, monthly = calendar_endpoints(occasions)
    primary.to_csv(out / "doc_endpoints_primary.csv", index=False)
    monthly.to_csv(out / "doc_monthly_primary.csv", index=False)
    common, _ = calendar_endpoints(occasions, start="2020-01-01", end="2024-12-31",
                                   minimum_months=24, minimum_years=4)
    common.to_csv(out / "doc_endpoints_common_window.csv", index=False)
    ledgers = []
    for site in inventory.site:
        r = records[site]
        ledgers.append({"site": site, "geometry_status": r["status"],
                        "geometry_eligibility": r.get("form_eligibility", r.get("error")),
                        "calendar_eligible": bool(primary.set_index("site").loc[site, "calendar_eligible"]),
                        "n_primary_months": int(primary.set_index("site").loc[site, "n_months"])})
    ledger = pd.DataFrame(ledgers).merge(eligible[["site", "ecology_coverage_eligible", "network_group", "cluster"]],
                                        on="site", how="left", validate="one_to_one")
    ledger["analysis_eligible"] = ledger.geometry_eligibility.eq("eligible") & ledger.calendar_eligible & ledger.ecology_coverage_eligible.eq(True)
    ledger.to_csv(out / "eligibility_ledger.csv", index=False)
    base = eligible[eligible.ecology_coverage_eligible.eq(True)].copy()
    base["log_basin_area"] = np.log(base.basin_area_km2)
    gains, all_predictions, influences, statuses = [], [], [], []
    for population in ("primary", "common_window", "no_st357_upstream_overlap", "domain_holdout"):
        responses = common if population == "common_window" else primary
        panel = base.merge(responses[responses.calendar_eligible], on="site", validate="one_to_one")
        if population == "no_st357_upstream_overlap":
            panel = panel[panel.st357_upstream_overlap_reaches.eq(0)]
        panel = panel.reset_index(drop=True)
        panel["log_n_months"] = np.log(panel.n_months)
        panel.to_csv(out / f"site_panel_{population}.csv", index=False)
        group_column = "domain" if population == "domain_holdout" else "network_group"
        status = {"population": population, "n_sites": len(panel), "n_groups": panel[group_column].nunique()}
        try:
            pred = heldout_predictions(panel, group_column=group_column)
            gain, influence = gain_summary(pred, draws=args.bootstrap_draws)
            for f in (pred, gain, influence):
                f["population"] = population
            gains.append(gain)
            all_predictions.append(pred)
            influences.append(influence)
            status["status"] = "evaluated"
        except ValueError as error:
            status.update(status="not_identifiable", reason=str(error))
        statuses.append(status)
        if population == "primary":
            pairs = matched_shape_pairs(panel)
            lookup = panel.set_index("site")
            for outcome in OUTCOMES:
                pairs[outcome + "_broad_minus_elongated"] = (lookup.reindex(pairs.site_b)[outcome].to_numpy() -
                                                             lookup.reindex(pairs.site_a)[outcome].to_numpy())
            pairs.to_csv(out / "context_matched_shape_pairs.csv", index=False)
    if gains:
        result = pd.concat(gains, ignore_index=True)
        result.to_csv(out / "heldout_form_gains.csv", index=False)
        pd.concat(influences, ignore_index=True).to_csv(out / "leave_group_influence.csv", index=False)
        pred = pd.concat(all_predictions, ignore_index=True)
        p = out / "heldout_site_predictions.parquet"
        pred.to_parquet(p, index=False)
        config = {"method": "Ridge", "alpha": 10, "arms": ARMS, "context": CONTEXT,
                  "outcomes": OUTCOMES, "populations": statuses, "bootstrap_draws": args.bootstrap_draws}
        meta = {"config": config, "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "dataset_hash": sha256_file(out / "site_panel_primary.csv"),
                "mask_hash": sha256_file(out / "network_overlaps.csv"), "prediction_hash": sha256_file(p),
                "spec_hash": sha256_file(ROOT / "study_plan.md"),
                "runtime_snapshot_hash": sha256_file(Path("src/river_graph/analysis/river_neon_form.py")),
                "rows": len(pred), "scope": "site-level statistical replication; not monthly reconstruction predictions"}
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(result[result.population.eq("primary")].to_string(index=False), flush=True)
    sources = [ROOT / "study_plan.md", Path(__file__), Path("src/river_graph/analysis/river_neon_form.py"),
               out / "network_forms.csv", out / "registered_locations.csv", out / "geometry_records.json",
               RAW / "streamcat_ws_coverage.json", EVIDENCE / "neon_doc_occasions.csv", OLD / "features.csv", OLD / "centroids.csv"]
    summary = {"candidate_sites": len(inventory), "measured_networks": len(forms),
               "geometry_eligible": len(eligible), "primary_analysis_eligible": int(ledger.analysis_eligible.sum()),
               "class_counts": {str(k): int(v) for k, v in eligible.cluster.value_counts().items()},
               "populations": statuses, "bootstrap_draws": args.bootstrap_draws,
               "sources": {str(p): sha256_file(p) for p in sources},
               "new_neural_training": False, "historical_geometry_cache_modified": False,
               "independent_replication_scope": "new laboratory sample archive; ST357 upstream overlap separately analyzed",
               "original_neon_quality_flags_available": False}
    (ROOT / "analysis_sources.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
