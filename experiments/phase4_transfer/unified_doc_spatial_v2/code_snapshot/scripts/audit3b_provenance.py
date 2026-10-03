"""3B-R0: provenance audit of the Phase-3 product tree (no training).

Checks what can be checked and records — never repairs — what cannot:

* content pins (sha256) for the 24 seed-ensemble files and 24 products,
  each marked ``retrospective_content_pin`` where no contemporaneous
  sidecar exists;
* retrospective sidecars for seed files, explicitly dated and labelled;
* gap list: missing config hash, missing seed-file binding, incomplete
  runtime snapshot in already-written products, cache-hit = existence-only.

Usage: python scripts/audit3b_provenance.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path("experiments/phase3_uncertainty_stcore_v1")
SEEDS = ("42", "43", "44", "45", "46")
RUN_LOG_CLAIM = ("produced by scripts/run3b_fullgrid.py in the run completed "
                 "2026-09-23T17:03:43Z (log /tmp/phase3b_fullgrid.log, "
                 "committed snapshot: 'Phase 3B complete' commit)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "binding_status": "retrospective_content_pin",
        "note": ("content hash pinned AFTER the fact (3B-R0); no "
                 "contemporaneous sidecar existed at write time"),
    }


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    seeds = sorted((OUT / "seed_preds").glob("*.parquet"))
    products = sorted((OUT / "full_grid").glob("P3_*.parquet"))

    seed_inv = [pin(p) for p in seeds]
    for rec, p in zip(seed_inv, seeds):
        meta_path = p.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({
            "created_at": now,
            "status": "retrospective_sidecar",
            "run_claim": RUN_LOG_CLAIM,
            "seeds": list(SEEDS),
            **rec,
        }, indent=2), encoding="utf-8")

    gaps: list[dict] = []
    prod_rows = []
    for p in products:
        df = pd.read_parquet(p, columns=["provenance_hashes"])
        prov = json.loads(df["provenance_hashes"].iloc[0])
        row = {"product": p.name,
               "has_dataset_hash": "dataset_sha256" in prov,
               "has_mask_hash": "mask_sha256" in prov,
               "has_spec_hash": "phase3_spec_sha256" in prov,
               "has_runtime_hash": "runtime_code_snapshot_sha256" in prov,
               "has_config_hash": "config_hash" in prov,
               "has_seed_file_hashes": "seed_file_sha256s" in prov,
               "sha256": sha256_file(p)}
        prod_rows.append(row)
        if not row["has_config_hash"]:
            gaps.append({"product": p.name,
                         "gap": "no config_hash (config dict stored instead)"})
        if not row["has_seed_file_hashes"]:
            gaps.append({"product": p.name,
                         "gap": "seed-ensemble file hashes not bound at write"})
        gaps.append({
            "product": p.name,
            "gap": ("runtime snapshot written before run3b_fullgrid.py was "
                    "part of the snapshot (hash covers an incomplete "
                    "execution surface)"),
        })
    gaps.append({
        "product": "*",
        "gap": ("24 seed files had no sidecars at write time; cache reuse was "
                "file-existence only — see run3b_fullgrid.py history"),
    })
    gaps.append({
        "product": "*",
        "gap": ("verdict labels revised in 3B-R0: land-cover strata are "
                "POST-HOC defined (not optimized), not a priori; network_"
                "distance tertile binning is structurally degenerate "
                "(visible stations' distance is 0 by definition)"),
    })

    inv_path = OUT / "eval" / "seed_identity_inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps({
        "audit_at": now,
        "mode": "read-only audit + retrospective sidecars (never re-stamping "
                "as contemporaneous)",
        "seed_files": seed_inv,
    }, indent=2), encoding="utf-8")
    pd.DataFrame(prod_rows).to_csv(OUT / "eval" / "product_identity.csv",
                                   index=False)
    (OUT / "eval" / "provenance_audit.json").write_text(json.dumps({
        "audit_at": now,
        "summary": {
            "seed_files": len(seeds),
            "seed_sidecars_contemporaneous": 0,
            "seed_sidecars_retrospective": len(seeds),
            "products": len(products),
            "products_with_full_binding": sum(
                1 for r in prod_rows if r["has_config_hash"]
                and r["has_seed_file_hashes"]),
        },
        "gaps": gaps,
    }, indent=2), encoding="utf-8")
    print(f"seed inventory -> {inv_path}")
    print(f"gaps: {len(gaps)} (see provenance_audit.json)")
    for g in gaps[:6]:
        print("  -", g["product"], ":", g["gap"][:70])


if __name__ == "__main__":
    main()
