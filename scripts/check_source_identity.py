"""Check whether a run's recorded sources are reproducible from the repository.

    python scripts/check_source_identity.py --root experiments/h3a_v1r3

A run record names the sha256 of every source file it was produced from. That
is only useful if someone can tell WHERE those bytes are. For each source this
reports three hashes:

    recorded   what the run recorded
    working    the file as it is in this checkout right now
    committed  the blob at HEAD, normalised to LF

Three outcomes are possible:

    reproducible   working == committed == recorded: any fresh clone reproduces it
    uncommitted    working == recorded != committed: the run depends on work that
                   has not been committed, so a fresh clone will not reproduce it
    stale          working != recorded: the file changed after the run, so the run
                   cannot be re-derived from this checkout at all

Exits non-zero when anything is not reproducible, so it can be used as a
release gate rather than as a report nobody reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.experiments.h3_runs import SOURCE_FILES


def normalize(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def committed_blob(relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", "HEAD:" + relative], cwd=ROOT, capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return normalize(result.stdout)


def first_record(root: Path) -> dict:
    runs = sorted((root / "runs").glob("*.json"))
    runs = [p for p in runs if not p.name.endswith((".intent.json", ".pending.json"))]
    if not runs:
        raise SystemExit("no run records under " + str(root))
    for path in runs:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            return record
    raise SystemExit("no complete run records under " + str(root))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="a result root holding run records")
    ap.add_argument("--record", default=None, help="an explicit run record")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.record:
        record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    else:
        from river_graph.experiments.h3_training import load_protocol

        protocol = load_protocol(ROOT / "configs/h3a_v1.json")
        root = ROOT / (args.root or protocol["result_root"])
        record = first_record(root)
    recorded = record["config"]["source_sha256"]

    rows = []
    counts = {"reproducible": 0, "uncommitted": 0, "stale": 0, "missing": 0}
    for relative in SOURCE_FILES:
        want = recorded.get(relative)
        path = ROOT / relative
        if not path.is_file():
            status, working, committed = "missing", None, None
        else:
            working = normalize(path.read_bytes())
            committed = committed_blob(relative)
            if working != want:
                status = "stale"
            elif committed == want:
                status = "reproducible"
            else:
                status = "uncommitted"
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "path": relative,
                "recorded": want,
                "working": working,
                "committed": committed,
                "status": status,
            }
        )

    width = max(len(row["path"]) for row in rows)
    print(f"{'source'.ljust(width)}  status        recorded / working / committed")
    for row in rows:
        print(
            f"{row['path'].ljust(width)}  {row['status']:<12}  "
            f"{str(row['recorded'])[:12]} / {str(row['working'])[:12]} / "
            f"{str(row['committed'])[:12]}"
        )
    print()
    print(json.dumps(counts, ensure_ascii=False))
    if counts["uncommitted"]:
        print(
            "\nThese sources match the run but are NOT committed, so a fresh "
            "checkout cannot reproduce the run's identity:\n  "
            + "\n  ".join(r["path"] for r in rows if r["status"] == "uncommitted")
        )
    if counts["stale"] or counts["missing"]:
        print(
            "\nThese sources no longer match the run at all; the run can only be "
            "re-derived from an archived copy of the sources:\n  "
            + "\n  ".join(r["path"] for r in rows
                          if r["status"] in ("stale", "missing"))
        )
    if args.json_out:
        out = ROOT / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"counts": counts, "sources": rows},
                                  indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print("\nwrote " + str(out))
    return 0 if counts["reproducible"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
