"""Copy the research assets to an independent location and prove they restore.

    python scripts/backup_assets.py --target D:/river_graph_backup_YYYYMMDD
    python scripts/backup_assets.py --target ... --verify
    python scripts/backup_assets.py --target ... --restore-check 5

Why this exists: a temporary git worktree with NTFS junctions was removed with
"git worktree remove --force", which followed the junctions and emptied the live
data/ and .venv/ directories. Nothing under version control was lost, but the
exact input snapshot the frozen experiments were produced from was, and that
made those runs impossible to replay. "Git lost nothing" is not the same as
"the research assets survived".

What is captured
    data/        downloaded caches and every processed dataset build
    experiments/ run records, checkpoints, predictions, decisions, tables
    src/ scripts/ tests/ configs/ docs/  the actual sources that ran
    cache/       the NHDPlus VAA cache the graph build needs
    the small root files (pyproject.toml, uv.lock, .gitattributes, ...)

What is skipped: .git (already versioned), .venv (rebuildable from uv.lock),
bytecode caches, and the local-only presentation/PDF workspaces.

The manifest records path, size and sha256 for every copied file, plus a digest
over the manifest itself. --verify re-hashes the backup and fails on any
mismatch. --restore-check additionally copies a few files back into a temporary
directory and compares them, so the backup is proven restorable rather than
merely present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("data", "experiments", "src", "scripts", "tests", "configs", "docs",
                "cache")
INCLUDE_FILES = ("pyproject.toml", "uv.lock", ".gitattributes", ".gitignore",
                 "README.md")
SKIP_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
    "_pdf_work", "deck_build", "qa_ppt", ".git", ".venv",
}
SKIP_SUFFIXES = (".pyc", ".pyo", ".pptx", ".tmp")
MANIFEST_NAME = "backup_manifest.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def iter_files(base: Path):
    for current, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES):
                continue
            yield Path(current) / name


def sources() -> list[Path]:
    found = []
    for name in INCLUDE_DIRS:
        base = ROOT / name
        if base.is_dir():
            found.extend(iter_files(base))
    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.is_file():
            found.append(path)
    return sorted(found)


def copy_assets(target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    started = time.perf_counter()
    files = sources()
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        if index % 200 == 0:
            print(f"  copied {index}/{len(files)}", flush=True)
    manifest = {
        "created_at": time.time(),
        "source_root": str(ROOT),
        "target": str(target),
        "files": len(entries),
        "bytes": sum(item["bytes"] for item in entries),
        "entries": entries,
        "seconds": time.perf_counter() - started,
    }
    manifest["digest"] = manifest_digest(entries)
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def manifest_digest(entries: list[dict]) -> str:
    payload = json.dumps(
        sorted(((e["path"], e["bytes"], e["sha256"]) for e in entries),
               key=lambda item: item[0]),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(target: Path) -> dict:
    path = target / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit("no manifest at " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def verify(target: Path) -> int:
    manifest = load_manifest(target)
    entries = manifest["entries"]
    if manifest_digest(entries) != manifest.get("digest"):
        raise SystemExit("manifest digest does not match its own entries")
    missing, corrupt = [], []
    for item in entries:
        path = target / item["path"]
        if not path.is_file():
            missing.append(item["path"])
            continue
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            corrupt.append(item["path"])
    print(f"verified {len(entries)} files: missing={len(missing)} "
          f"corrupt={len(corrupt)}")
    for path in (missing + corrupt)[:10]:
        print("  problem:", path)
    return 1 if missing or corrupt else 0


def restore_check(target: Path, count: int) -> int:
    manifest = load_manifest(target)
    staged = target.parent / (target.name + "__restore_check")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    rng = random.Random(20260916)
    sample = rng.sample(manifest["entries"], min(count, len(manifest["entries"])))
    problems = 0
    for item in sample:
        source = target / item["path"]
        destination = staged / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        status = "ok" if digest == item["sha256"] else "MISMATCH"
        if status != "ok":
            problems += 1
        print(f"  restore {status}: {item['path']}")
    print(f"restore check staged in {staged}; mismatches={problems}")
    if problems == 0:
        shutil.rmtree(staged)
        print("restore check passed; temporary copy removed")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--restore-check", type=int, default=0)
    args = ap.parse_args()
    target = Path(args.target)
    if args.verify or args.restore_check:
        if not target.is_dir():
            raise SystemExit("target does not exist: " + str(target))
        code = verify(target)
        if args.restore_check:
            code |= restore_check(target, args.restore_check)
        return code
    if target.resolve() == ROOT.resolve() or ROOT.resolve() in target.resolve().parents:
        raise SystemExit("refusing to back up into the repository itself")
    manifest = copy_assets(target)
    print(f"backed up {manifest['files']} files, "
          f"{manifest['bytes'] / 1e6:.1f} MB in {manifest['seconds']:.1f}s")
    print("manifest digest:", manifest["digest"][:16])
    print("manifest:", target / MANIFEST_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
