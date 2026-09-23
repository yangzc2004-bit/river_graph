"""Build a training dataset version (Milestone 2).

Stages: select graph nodes -> fetch WQP observations (resumable) ->
fetch daily discharge -> covariate quality rules -> monthly aggregation ->
publish a versioned dataset + provenance.

    python scripts/build_dataset.py --version v07
    python scripts/build_dataset.py --version v07 --smoke 50

A version is published once. The registry in
data/processed/dataset_registry.json records the content hash of every version
that has been published, and a version that is already registered is NEVER
re-bound to different bytes: pick a new name instead. That matters because a
frozen experiment names the exact dataset hash it ran on, so silently replacing
a released version would retroactively invalidate it, and re-using a name whose
file has been deleted would attach a frozen name to new content.

The build also writes a per-version raw-input manifest: every cached provider
file with its size, sha256 and actual DOC sample count, next to the count the
NWIS catalog reports. A cache file can exist and still be missing most of its
history, so the completeness of the inputs is recorded rather than assumed.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from river_graph.data.aggregate import to_monthly
from river_graph.data.nwis import fetch_daily_discharge, load_daily_discharge
from river_graph.data.quality import (
    REPO_ROOT,
    RULE_VERSION,
    RULES,
    apply_rules,
    rejection_sample,
    rules_digest,
    summarize,
    write_report,
)
from river_graph.data.wqp import (
    cached_doc_sample_count,
    extract_covariate_obs,
    extract_doc_obs,
    fetch_station_results,
    load_station_results,
)
from river_graph.dataset import build_dataset, save_dataset

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
COVARIATES = ("temperature", "ph", "spec_conductance")
REGISTRY = PROCESSED / "dataset_registry.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"versions": {}}


def git_revision() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, encoding="utf-8", check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a build outside a checkout is fine
        return "unknown"


def catalog_counts() -> dict:
    """DOC sample count the NWIS series catalog reports per station."""
    path = PROCESSED / "doc_site_inventory.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype={"site_no": str})
    return table.groupby("site_no")["count_nu"].max().to_dict()


def acceptable_doc_samples(catalog_count: int) -> int:
    """How few cached samples still count as a complete response.

    The catalog counts raw provider samples; the extractor applies the frozen
    fraction, unit and detection filters, so a shortfall of a few samples is
    normal (669 of 683 at one station). A shortfall of hundreds is not: that is
    the truncation signature that lost 1,895 samples. The threshold sits between
    the two so a normal build does not re-hammer a station whose provider simply
    returns less.
    """
    if not catalog_count or catalog_count <= 0:
        return 0
    return max(1, min(int(catalog_count * 0.9), int(catalog_count) - 5))


def prepare_discharge(dv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly discharge from NWIS daily values, through the quality rules.

    The rules are applied on the real build path, not only in the audit script:
    before this, discharge was aggregated straight from the daily values while
    the documented rule was never executed, so "0 rejected" proved nothing.
    NWIS values are already in the configured unit, and the rule bounds the
    magnitude in both directions while keeping the sign, because reversing flow
    in tidal and backwater reaches is a real measurement.
    """
    candidates = pd.DataFrame(
        {
            "site_no": dv["site_no"].values,
            "date": dv["date"].values,
            "value": dv["discharge_cfs"].values,
            "unit": None,
        }
    )
    audited = apply_rules(candidates, "discharge")
    kept = audited[audited.qc_status == "accepted"]
    monthly = to_monthly(
        kept.rename(columns={"value": "discharge_cfs"})[
            ["site_no", "date", "discharge_cfs"]
        ],
        "discharge_cfs",
    ).rename(columns={"discharge_cfs": "discharge"})
    return monthly, audited


def select_sites(nodes: pd.DataFrame, edges: pd.DataFrame, smoke: int | None) -> list[str]:
    """Smoke mode: best-sampled sites inside the largest connected component."""
    if smoke is None:
        return sorted(nodes["site_no"])
    g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
    largest = max(nx.weakly_connected_components(g), key=len)
    inv = pd.read_csv(PROCESSED / "doc_site_inventory.csv", dtype={"site_no": str})
    top = (
        inv[inv["site_no"].isin(largest)]
        .nlargest(smoke, "count_nu")["site_no"]
        .tolist()
    )
    print(f"smoke mode: top {len(top)} DOC-sampled sites in largest component "
          f"({len(largest)} nodes)")
    return top


def raw_input_manifest(sites: list[str], expected: dict) -> dict:
    """Every cached provider file used, with its size, hash and DOC count."""
    cache = RAW / "wqp_results"
    entries = []
    short = []
    for site in sites:
        path = cache / (site + ".csv")
        if not path.is_file():
            entries.append({"site_no": site, "present": False})
            short.append(site)
            continue
        samples = cached_doc_sample_count(path)
        want = expected.get(site)
        entry = {
            "site_no": site,
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "doc_samples": samples,
            "catalog_doc_samples": want,
        }
        if want is not None and samples < want:
            entry["shortfall"] = int(want) - samples
            short.append(site)
        entries.append(entry)
    digest = hashlib.sha256(
        json.dumps(
            sorted(((e["site_no"], e.get("bytes"), e.get("sha256")) for e in entries),
                   key=lambda item: item[0]),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "digest": digest,
        "files": len(entries),
        "stations_short_of_catalog": len(short),
        "short_stations": sorted(short)[:50],
        "entries": entries,
    }


def publish(dataset: dict, version: str, tag: str, provenance: dict,
            allow_republish: bool) -> Path:
    """Write a dataset version, refusing to rebind an existing one."""
    registry = load_registry()
    registered = registry["versions"].get(version)
    out = PROCESSED / f"mississippi_graph_{version}{tag}.pt"
    if registered is not None:
        raise SystemExit(
            "refusing to publish " + version + ": it is already registered with "
            "sha256 " + str(registered.get("sha256"))[:16] + " and a frozen "
            "experiment may name it. Publish a new version name instead."
        )
    if out.exists() and not allow_republish:
        raise SystemExit(
            "refusing to overwrite " + str(out) + "; that file is not registered "
            "but it exists. Pass --allow-republish if you are sure it is a "
            "disposable artifact, or choose a new version name."
        )
    temp = out.with_name(out.name + ".tmp")
    save_dataset(dataset, temp)
    digest = sha256_file(temp)
    os.replace(temp, out)
    entry = {
        "version": version,
        "file": out.name,
        "sha256": digest,
        "bytes": out.stat().st_size,
        "published_at": time.time(),
        "git_revision": git_revision(),
        **provenance,
    }
    registry["versions"][version] = entry
    write_report(REGISTRY, registry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None, help="limit to N sites")
    ap.add_argument(
        "--cached-only",
        action="store_true",
        help="use only stations with a complete cached WQP result file",
    )
    ap.add_argument(
        "--version",
        required=True,
        help="output version tag; must be a name that has never been published",
    )
    ap.add_argument(
        "--allow-republish",
        action="store_true",
        help=(
            "overwrite an unregistered file of the same name. A registered "
            "version is never overwritten, whatever this flag says."
        ),
    )
    ap.add_argument(
        "--skip-completeness-check",
        action="store_true",
        help=(
            "do not refetch cache files that hold fewer DOC samples than the "
            "NWIS catalog reports. Only for deliberate diagnostics."
        ),
    )
    args = ap.parse_args()

    with open(PROCESSED / "mississippi_graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    nodes, edges = graph["nodes"], graph["edges"]
    sites = select_sites(nodes, edges, args.smoke)
    if args.cached_only:
        cached_sites = {p.stem for p in (RAW / "wqp_results").glob("*.csv")}
        sites = [site for site in sites if site in cached_sites]
        print(f"cached-only mode: {len(sites)} stations have WQP files")
    nodes = nodes[nodes["site_no"].isin(sites)].reset_index(drop=True)
    if args.cached_only:
        edges = edges[
            edges["source"].isin(sites) & edges["target"].isin(sites)
        ].reset_index(drop=True)
    print(f"sites: {len(sites)}")

    # --- fetch observations (resumable, per-station cache) ---
    catalog = {} if args.skip_completeness_check else catalog_counts()
    expected = {
        site: acceptable_doc_samples(int(count))
        for site, count in catalog.items()
    }
    cache = RAW / "wqp_results"
    results_by_site = {}
    failed = []

    def fetch_one(site_no: str):
        try:
            path = fetch_station_results(
                site_no, cache, min_doc_samples=expected.get(site_no)
            )
            if path is None:
                return site_no, None
            return site_no, load_station_results(path)
        except Exception as exc:  # noqa: BLE001 - one bad provider file must not stop all sites
            print(f"  {site_no}: load failed: {exc}")
            return site_no, None

    # Requests are independent and each station has its own cache file. A
    # small pool shortens the rebuild without changing the station-level retry
    # or completeness rules.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(fetch_one, site_no) for site_no in sites]
        for i, future in enumerate(as_completed(futures), start=1):
            site_no, frame = future.result()
            if frame is None:
                failed.append(site_no)
            else:
                results_by_site[site_no] = frame
            if i % 25 == 0:
                print(f"  fetched {i}/{len(sites)}")
    results = [results_by_site[s] for s in sites if s in results_by_site]
    if failed:
        print(f"WQP failed for {len(failed)} sites (rerun to retry): {failed[:10]}")

    # --- validation 1: DOC field consistency ---
    allres = pd.concat(results, ignore_index=True)
    oc = allres[allres["variable"] == "organic_carbon"]
    print("\n[validation 1] organic carbon rows by fraction/unit:")
    print(oc.groupby(["fraction", "unit"], dropna=False).size().nlargest(8))
    doc_obs = extract_doc_obs(allres)
    print(f"DOC obs (dissolved, mg/L, uncensored): {len(doc_obs):,} "
          f"across {doc_obs['site_no'].nunique()} sites")

    # --- covariate quality rules ---
    # A value is rejected only when its unit or its physical range makes it
    # impossible for that variable. DOC keeps its already-frozen unit and
    # detection filters and gains no value-range rule, so real extremes stay.
    audit_frames = []
    covariate_obs = {
        name: extract_covariate_obs(allres, name, audit=audit_frames)
        for name in COVARIATES
    }

    # --- monthly aggregation ---
    monthly_doc = to_monthly(doc_obs, "doc")
    temp_obs = covariate_obs["temperature"]
    monthly_temp = to_monthly(temp_obs, "temperature")
    print("\n[validation 2] monthly DOC labels per site:")
    print(monthly_doc.groupby("site_no").size().describe().round(1))

    # --- discharge: NWIS daily values through the SAME quality rules ---
    # Keep response payloads small enough for the public NWIS service and
    # make interrupted runs resumable at the cache-batch level.
    fetch_daily_discharge(sites, RAW / "nwis_dv", batch_size=5)
    dv = load_daily_discharge(RAW / "nwis_dv")
    dv = dv[dv["site_no"].isin(sites)]
    monthly_flow, discharge_audited = prepare_discharge(dv)
    audit_frames.append(discharge_audited)
    print(f"\n[quality] discharge candidates {len(discharge_audited):,}, "
          f"accepted {int((discharge_audited.qc_status == 'accepted').sum()):,}, "
          f"rejected {int((discharge_audited.qc_status == 'rejected').sum()):,}")

    # --- the quality report is per version, not a single shared file ---
    audited = pd.concat(audit_frames, ignore_index=True)
    quality = summarize(audited)
    print(f"[quality] rule version {RULE_VERSION}")
    for (variable, reason), count in (
        audited[audited.qc_status == "rejected"]
        .groupby(["variable", "qc_reason"]).size().items()
    ):
        print(f"  rejected {variable}: {reason} x {count}")
    rejections_path = PROCESSED / f"covariate_rejections_{args.version}.csv"
    audited[audited.qc_status == "rejected"].to_csv(rejections_path, index=False)
    quality_report = {
        "rule_version": RULE_VERSION,
        "rules_sha256": rules_digest(),
        "rules": RULES,
        "version": args.version,
        "summary": quality,
        "sample": rejection_sample(audited),
        "rejections_file": rejections_path.name,
        "discharge_note": (
            "NWIS daily values go through the same rules as the WQP covariates; "
            "the magnitude bound is applied in both directions and the sign is "
            "kept, so reversing flow survives while unit errors do not."
        ),
        "doc_note": (
            "DOC is the target, not a covariate: the extractor keeps its "
            "frozen unit/detection filters and no value range is imposed."
        ),
    }
    quality_path = PROCESSED / f"covariate_quality_report_{args.version}.json"
    write_report(quality_path, quality_report)

    # --- validation 3: missingness funnel ---
    n_feat = monthly_flow["site_no"].nunique()
    print("\n[validation 3] funnel:")
    print(f"  selected sites:            {len(sites)}")
    print(f"  with >=1 DOC obs:          {doc_obs['site_no'].nunique()}")
    print(f"  with >=12 monthly labels:  "
          f"{(monthly_doc.groupby('site_no').size() >= 12).sum()}")
    print(f"  with discharge data:       {n_feat}")
    print(f"  with temperature data:     {monthly_temp['site_no'].nunique()}")

    # --- assemble dataset ---
    if monthly_doc.empty:
        raise RuntimeError("no DOC labels extracted; check validation 1 above")
    months = pd.date_range(
        monthly_doc["month"].min(), monthly_doc["month"].max(), freq="MS"
    )

    # H2 additions: node regime + edge transport attributes (v03)
    reach = pd.read_csv(PROCESSED / "reach_attributes.csv", dtype={"site_no": str})
    reach = reach.set_index("site_no")
    in_deg = edges.groupby("target").size()
    regime = []
    for s in nodes["site_no"]:
        r = reach.loc[s]
        regime.append([
            r["streamorde"], np.log1p(r["totdasqkm"]), r["slope"],
            float(in_deg.get(s, 0) == 0),  # is_headwater
        ])

    # M5 additions (v04): StreamCat ecological context, keyed by comid
    sc_path = PROCESSED / "streamcat_attributes.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path, dtype={"comid": str}).set_index("comid")
        comid_of = reach["comid"].astype("Int64").astype(str)
        for s_i, s in enumerate(nodes["site_no"]):
            c = comid_of.get(s)
            if c in sc.index:
                row = sc.loc[c]
                regime[s_i] += [
                    row["pctconif2019ws"] + row["pctdecid2019ws"] + row["pctmxfst2019ws"],
                    row["pctcrop2019ws"] + row["pcthay2019ws"],
                    row["pcturbhi2019ws"] + row["pcturbmd2019ws"]
                    + row["pcturblo2019ws"] + row["pcturbop2019ws"],
                    row["pctwdwet2019ws"] + row["pcthbwet2019ws"],
                    row["precip9120ws"], np.log1p(row["tmean9120ws"] + 20),
                    row["omws"], row["elevws"], row["bfiws"],
                ]
            else:
                regime[s_i] += [np.nan] * 9
        regime = np.nan_to_num(np.asarray(regime, dtype=float), nan=-1.0)
    regime = np.asarray(regime, dtype=np.float32)
    ef = pd.read_csv(PROCESSED / "edge_features.csv", dtype={"source": str, "target": str})
    ef = edges.merge(ef, on=["source", "target"], how="left")
    edge_attr = np.nan_to_num(ef[[
        "hop_dist", "geo_dist_deg", "target_lengthkm",
        "target_totdasqkm", "target_slope", "target_streamorde",
    ]].values.astype(float))
    edge_attr[:, 2] = np.log1p(edge_attr[:, 2])  # lengthkm
    edge_attr[:, 3] = np.log1p(edge_attr[:, 3])  # drainage area

    dataset = build_dataset(
        nodes, edges, monthly_doc,
        {"temperature": monthly_temp, "discharge": monthly_flow},
        months,
        edge_attr=edge_attr,
        regime=np.asarray(regime, dtype=np.float32),
    )

    manifest = raw_input_manifest(sites, catalog_counts())
    print(f"\n[raw inputs] {manifest['files']} cache files, digest "
          f"{manifest['digest'][:16]}")
    if manifest["stations_short_of_catalog"]:
        print(f"[raw inputs] WARNING: {manifest['stations_short_of_catalog']} "
              f"stations still hold fewer DOC samples than the catalog reports: "
              f"{manifest['short_stations'][:10]}")
    manifest_path = PROCESSED / f"raw_input_manifest_{args.version}.json"
    write_report(manifest_path, manifest)

    tag = f"_smoke{args.smoke}" if args.smoke else ""
    temperature = dataset["x"][:, :, 0]
    provenance = {
        "sites": len(dataset["site_no"]),
        "months": len(dataset["months"]),
        "edges": int(dataset["edge_index"].shape[1]),
        "quality_rule_version": RULE_VERSION,
        "quality_rules_sha256": rules_digest(),
        "quality_summary": quality,
        "quality_report": quality_path.name,
        "quality_report_sha256": sha256_file(quality_path),
        "rejections_file": rejections_path.name,
        "rejections_sha256": sha256_file(rejections_path),
        "raw_input_manifest": manifest_path.name,
        "raw_input_manifest_sha256": sha256_file(manifest_path),
        "raw_input_digest": manifest["digest"],
        "raw_input_short_stations": manifest["stations_short_of_catalog"],
        "temperature_cells_outside_rule": int(
            ((temperature > RULES["temperature"]["max"])
             | (temperature < RULES["temperature"]["min"])).sum()
        ),
        "smoke_sites": args.smoke,
    }
    out = publish(dataset, args.version, tag, provenance, args.allow_republish)
    write_report(out.with_suffix(".provenance.json"), provenance)
    print(f"\npublished {out}: N={provenance['sites']}, T={provenance['months']}, "
          f"E={provenance['edges']}, "
          f"label coverage={dataset['y_mask'].float().mean():.1%}")
    print(f"temperature cells still outside the rule: "
          f"{provenance['temperature_cells_outside_rule']}")
    print(f"registered in {REGISTRY}")


if __name__ == "__main__":
    main()
