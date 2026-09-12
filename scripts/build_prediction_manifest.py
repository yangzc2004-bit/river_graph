"""A0: freeze the Phase-A analysis input list (prediction manifest).

Every analysis in Phase A must be able to name the exact prediction files it
was allowed to read. This script writes that allow-list, together with the
conflicts and gaps it deliberately excludes, so a later reader can tell which
numbers came from which batch.

Content pinning (ratchet)
-------------------------
A manifest is only trustworthy if a file cannot be swapped out from under it.
Rebuilding by re-hashing whatever is on disk would do exactly that: overwrite a
prediction and the next build re-admits the new, unverified content as trusted.
So each admitted file keeps a ``pinned_sha256`` that is carried forward from the
previous manifest. A file whose bytes no longer match its pin is never admitted;
it is recorded as ``content_changed_since_pin`` and has to be reviewed and
re-pinned explicitly with ``--repin`` (which prints why).

Usage:
    python scripts/build_prediction_manifest.py
    python scripts/build_prediction_manifest.py --repin   # accept changed bytes
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from river_graph.experiments.prediction_sources import sha256_file

FROZEN = Path("experiments/frozen_results")
CURRENT = Path("experiments/predictions")
HISTORICAL = Path("experiments/predictions_historical")
ANALYSIS = Path("experiments/analysis")

MANIFEST_VERSION = 2


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=False)
    return res.stdout.strip() if res.returncode == 0 else ""


def today() -> str:
    return datetime.now(timezone.utc).date().strftime("%Y%m%d")


def load_previous_pins(manifest_path: Path) -> dict[str, str]:
    """``{"batch/file": pinned_sha256}`` carried forward from the last manifest.

    Both manifest generations are supported: v2 entries carry ``sha256`` and no
    explicit pin, v3 entries carry ``pinned_sha256``.
    """
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    pins: dict[str, str] = {}
    for group in ("predictions", "historical_recovered"):
        for entry in payload.get(group, []):
            key = f"{entry.get('batch', 'predictions')}/{entry['file']}"
            pin = entry.get("pinned_sha256") or entry.get("sha256")
            if pin:
                pins[key] = pin
    return pins


def apply_pin(entry: dict, pins: dict[str, str], seen: set[str],
              repin: bool) -> bool:
    """Attach a content pin to ``entry``; return True when its bytes match.

    A changed file is never admitted: the previous pin wins, the mismatch is
    recorded on the entry so a reviewer can see it, and ``--repin`` is required
    to accept the new bytes deliberately.
    """
    key = f"{entry['batch']}/{entry['file']}"
    seen.add(key)
    current = entry["sha256"]
    pin = pins.get(key)
    entry["sha256_at_build"] = current
    if pin is None or repin or pin == current:
        entry["pinned_sha256"] = current
        entry["pin_status"] = "pinned" if pin is None else "repinned"
        return True
    entry["pinned_sha256"] = pin
    entry["pin_status"] = "content_changed_since_pin"
    entry["used_for_phase_a"] = False
    entry["repin_required"] = True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=today())
    ap.add_argument("--manifest", default=None,
                    help="manifest to rebuild (default: the dated file)")
    ap.add_argument("--repin", action="store_true",
                    help="accept content that changed since the previous pin "
                         "(prints every change; refuse to use this to silence a "
                         "mismatch you have not reviewed)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else \
        FROZEN / f"prediction_manifest_{args.date}.json"
    pins = load_previous_pins(manifest_path)
    seen_pins: set[str] = set()

    ANALYSIS.mkdir(parents=True, exist_ok=True)

    frozen = pd.read_csv(FROZEN / "benchmark.csv")
    multiseed = pd.read_csv(FROZEN / "benchmark_multiseed.csv")
    verification = ANALYSIS / f"artifact_verification_{args.date}.csv"
    vdf = pd.read_csv(verification) if verification.exists() else None

    status: dict[str, str] = {}
    if vdf is not None:
        status = {r["file"]: r["status"] for _, r in vdf.iterrows()}

    # Batch keys are directory names, and every entry carries the content hash,
    # so a name that exists in both batches cannot be confused or double-counted.
    entries = []
    for p in sorted(CURRENT.glob("*.parquet")):
        model = str(pd.read_parquet(p, columns=["model"])["model"].iloc[0])
        mask = p.stem.split("__", 1)[1]
        st = status.get(p.name, "unverified")
        in_frozen = bool(((frozen["model"] == model) &
                          (frozen["mask"] == mask)).any())
        in_multi = bool(((multiseed["model"] == model) &
                         (multiseed["mask"] == mask)).any())
        entry = {
            "file": p.name, "batch": "predictions", "status": st,
            "sha256": sha256_file(p),
            "in_frozen_table": in_frozen, "in_multiseed_table": in_multi,
            "used_for_phase_a": st.startswith("verified"),
        }
        apply_pin(entry, pins, seen_pins, args.repin)
        entries.append(entry)

    historical = []
    for p in sorted(HISTORICAL.glob("*.parquet")):
        entry = {
            "file": p.name, "batch": "predictions_historical",
            "sha256": sha256_file(p),
            "used_for_phase_a": True,
            "reason": "overwritten or deleted after the frozen batch; "
                      "recomputed metrics match the frozen table",
        }
        apply_pin(entry, pins, seen_pins, args.repin)
        historical.append(entry)

    # Exactly one source per (model, mask): where a name exists in both batches
    # and both are otherwise acceptable, the recovered copy wins and the
    # overwritten one is recorded as a conflict. A name that is merely
    # unverified is admitted from the batch that owns it (current preferred).
    claimed: dict[str, str] = {}
    conflicts: list[dict] = []
    for entry in historical:
        model, mask = entry["file"].split("__", 1)
        entry["mask"] = mask[: -len(".parquet")]
        entry["model"] = model
        claimed.setdefault(f"{model}__{entry['mask']}", "predictions_historical")
    for entry in entries:
        model, mask = entry["file"].split("__", 1)
        key = f"{model}__{mask[: -len('.parquet')]}"
        if key in claimed:
            entry["used_for_phase_a"] = False
            entry["superseded_by"] = f"predictions_historical/{entry['file']}"
            conflicts.append({
                "file": entry["file"], "batch": "predictions",
                "sha256": entry["sha256"],
                "problem": ("a same-named recovered copy is the frozen-table "
                            "source; this overwritten copy is excluded so the "
                            "(model, mask) is counted once"),
                "used_for_phase_a": False,
                "authoritative_value": f"predictions_historical/{entry['file']}",
            })
        else:
            claimed[key] = "predictions"
        if entry["status"] == "conflict" and not any(
                c["file"] == entry["file"] for c in conflicts):
            conflicts.append({
                "file": entry["file"], "batch": "predictions",
                "sha256": entry["sha256"],
                "problem": ("recomputed test metrics disagree with the frozen "
                            "table; the file was overwritten after the freeze by "
                            "a different model run"),
                "used_for_phase_a": False,
                "authoritative_value": "experiments/frozen_results/benchmark.csv",
            })

    gaps = [
        {
            "file": e["file"], "batch": e["batch"], "sha256": e["sha256"],
            "problem": "no finite test-set prediction; metrics are NaN by design",
            "used_for_phase_a": True,
            "note": "report n and coverage; do not compare error magnitudes "
                    "against full-coverage models",
        }
        for e in entries if e["status"] == "zero_coverage"
    ]

    dupes = {name: count for name, count in
             Counter(e["file"] for e in entries + historical).items()
             if count > 1}
    if dupes and len({(e["batch"], e["file"]) for e in entries + historical}) \
            != len(entries) + len(historical):
        raise RuntimeError(f"duplicate (batch, file) entries: {dupes}")

    changed = [e for e in entries + historical
               if e.get("pin_status") == "content_changed_since_pin"]
    drop = sorted({key.split("/", 1)[0] + "/" + key.split("/", 1)[1]
                   for key in pins if key not in seen_pins})

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "duplicate_names_across_batches": dupes,
        "repinned": args.repin,
        "content_changed_since_pin": [
            {"file": e["file"], "batch": e["batch"],
             "pinned_sha256": e["pinned_sha256"],
             "sha256_at_build": e["sha256_at_build"]}
            for e in changed
        ],
        "pins_removed": drop,
        "manifest_date": args.date,
        "commit": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--format=%s", "HEAD"),
        "materials": {
            "frozen_benchmark": str(FROZEN / "benchmark.csv"),
            "frozen_multiseed": str(FROZEN / "benchmark_multiseed.csv"),
            "prediction_dir": str(CURRENT),
            "historical_dir": str(HISTORICAL),
            "mask_dir": "experiments/masks",
            "masks": sorted(p.stem for p in Path("experiments/masks").glob("*.npz")),
        },
        "counts": {
            "current_predictions": len(entries),
            "used_for_phase_a": sum(e["used_for_phase_a"] for e in entries),
            "historical_recovered": sum(e["used_for_phase_a"] for e in historical),
            "conflicts": len(conflicts),
            "zero_coverage": len(gaps),
            "content_changed_since_pin": len(changed),
            "frozen_table_rows": len(frozen),
            "multiseed_table_rows": len(multiseed),
        },
        "predictions": entries,
        "historical_recovered": historical,
        "conflicts": conflicts,
        "zero_coverage": gaps,
        "pins": {f"{e['batch']}/{e['file']}": e["pinned_sha256"]
                 for e in entries + historical if e.get("pinned_sha256")},
        "new_five_seed_predictions": {
            "available": False,
            "note": ("No per-cell predictions exist for H1/H2/H2X training "
                     "seeds 0-4; five-seed prediction ensembling, residual maps "
                     "and per-station intervals are therefore not possible and "
                     "are listed as not-run."),
        },
        "excluded_from_analysis": {
            "env_group_runs": [
                "H2X_hydro_river", "H2X_landcover_river",
                "H2X_climate_river", "H2X_soil_topo_river",
            ],
            "reason": ("env_groups column selection is known to be wrong "
                       "(src/river_graph/models/gcn.py); the default full-feature "
                       "H2X path is unaffected. Group rankings cannot support "
                       "mechanism conclusions."),
        },
    }

    out = manifest_path
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"manifest -> {out}")
    for key, value in manifest["counts"].items():
        print(f"  {key:28} {value}")
    if dupes:
        print(f"  duplicate names across batches: {dupes}")
    selected = manifest["counts"]["used_for_phase_a"] + \
        manifest["counts"]["historical_recovered"]
    print(f"  SELECTED (unique sources)   : {selected}")
    print()
    for name in conflicts:
        print(f"  excluded: {name['batch']}/{name['file']}")
    if changed:
        print()
        print(f"REFUSED: {len(changed)} file(s) whose bytes changed since they "
              f"were pinned; they are NOT in the allow-list:")
        for entry in changed:
            print(f"  - {entry['batch']}/{entry['file']}")
            print(f"      pinned   : {entry['pinned_sha256'][:16]}")
            print(f"      on disk  : {entry['sha256_at_build'][:16]}")
        print("  review what replaced them, then re-run with --repin to accept "
              "(or restore the pinned bytes from git).")
    if drop:
        print()
        print(f"pins dropped (files no longer present): {len(drop)}")
        for name in drop:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
