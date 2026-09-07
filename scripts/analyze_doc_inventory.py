"""Analyze DOC (00681) data availability from the NWIS series catalog.

Reads data/raw/nwis_catalog/batch_*.rdb (one row per site x parameter with
begin_date/end_date/count_nu) and reports the DOC station data profile for
the study region, plus covariate availability.

Usage: python scripts/analyze_doc_inventory.py
"""


from river_graph.config import load_config
from river_graph.data.nwis import load_qw_catalog

CATALOG_DIR = "data/raw/nwis_catalog"


def main() -> None:
    cfg = load_config()
    pcode = cfg["target"]["pcode"]

    cat = load_qw_catalog(CATALOG_DIR)
    print(f"catalog rows: {len(cat):,}; sites: {cat['site_no'].nunique():,}")

    doc = cat[(cat["data_type_cd"] == "qw") & (cat["parm_cd"] == pcode)].copy()
    doc["span_years"] = (doc["end_date"] - doc["begin_date"]).dt.days / 365.25
    doc["avg_gap_days"] = (doc["end_date"] - doc["begin_date"]).dt.days / (doc["count_nu"] - 1).clip(lower=1)
    doc = doc[doc["count_nu"] > 0]

    print(f"\n=== DOC (pcode {pcode}) availability ===")
    print(f"sites with >=1 DOC sample: {len(doc):,}")
    print(f"total DOC samples: {int(doc['count_nu'].sum()):,}")
    print(f"earliest: {doc['begin_date'].min().date()}, latest: {doc['end_date'].max().date()}")
    print("\nSample-count thresholds:")
    for th in [1, 5, 10, 20, 50, 100, 200]:
        n = (doc["count_nu"] >= th).sum()
        print(f"  sites with >= {th:>3} samples: {n:>5}")
    s20 = doc[doc["count_nu"] >= 20]
    print("\nRecord span (years), sites with >=20 samples:")
    print(s20["span_years"].describe().round(1))
    print("\nAverage sampling gap (days), sites with >=20 samples:")
    print(s20["avg_gap_days"].describe().round(1))

    print("\n=== covariate availability among DOC sites ===")
    base = set(doc["site_no"])
    n20 = set(s20["site_no"])
    for name, code in cfg["covariates"].items():
        sub = cat[(cat["parm_cd"] == code) & (cat["site_no"].isin(base))]
        sub20 = sub[sub["site_no"].isin(n20)]
        print(f"  {code} {name:>18}: {sub['site_no'].nunique():>5} of all DOC sites, "
              f"{sub20['site_no'].nunique():>5} of sites with >=20 DOC samples")

    doc_out = doc[["site_no", "begin_date", "end_date", "count_nu", "span_years", "avg_gap_days"]]
    doc_out = doc_out.sort_values("count_nu", ascending=False)
    doc_out.to_csv("data/processed/doc_site_inventory.csv", index=False)
    print("\nSaved: data/processed/doc_site_inventory.csv")
    print("\nTop 15 best-sampled DOC sites:")
    print(doc_out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
