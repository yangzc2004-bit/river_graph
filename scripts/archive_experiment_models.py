"""Archive staged experiment models without changing frozen research artifacts.

Run with ``uv run python scripts/archive_experiment_models.py --help``.
Inventory and packing are read-only with respect to source models and Git.
Uploads require the GitHub CLI; local eviction is deliberately a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
KEEP_LOCAL_NAMES = {"model.joblib", "source_attention.joblib", "ensemble.joblib"}


def command(args: list[str], *, root: Path | None = None, data: bytes | None = None) -> bytes:
    return subprocess.check_output(args, cwd=root, input=data)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root / Path(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative}")
    return target


def index_entries(root: Path) -> dict[str, tuple[str, str]]:
    entries = {}
    for item in command(["git", "ls-files", "--stage", "-z"], root=root).split(b"\0"):
        if not item:
            continue
        metadata, name = item.decode().split("\t", 1)
        mode, oid, stage = metadata.split()
        if stage != "0":
            raise ValueError("Resolve index conflicts before archiving models")
        entries[name] = (mode, oid)
    return entries


def inventory(root: Path, output: Path, work_dir: Path, repository: str) -> dict:
    """Hash every staged, newly added model and checkpoint incremental progress."""
    work_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work_dir / "inventory_progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    entries = index_entries(root)
    added = command(
        ["git", "diff", "--cached", "--diff-filter=A", "--name-only", "-z", "--", "experiments"],
        root=root,
    ).split(b"\0")
    names = sorted(item.decode() for item in added if item.endswith(b".joblib"))
    if not names:
        raise ValueError("No newly staged experiment .joblib models to archive")
    models = []
    for number, name in enumerate(names, 1):
        path = safe_path(root, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular model file: {name}")
        before = path.stat()
        mode, oid = entries[name]
        fingerprint = [before.st_size, before.st_mtime_ns, before.st_ino, oid]
        cached = previous.get(name)
        if cached and cached["fingerprint"] == fingerprint:
            digest = cached["sha256"]
        else:
            sha = hashlib.sha256()
            blob_sha = hashlib.sha1(f"blob {before.st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda stream=stream: stream.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
                    blob_sha.update(chunk)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != tuple(fingerprint[:3]):
                raise ValueError(f"Model changed while hashing: {name}")
            if blob_sha.hexdigest() != oid:
                raise ValueError(f"Worktree model differs from its staged bytes: {name}")
            digest = sha.hexdigest()
            previous[name] = {"fingerprint": fingerprint, "sha256": digest}
            atomic_json(progress_path, previous)
        models.append({
            "path": name, "size_bytes": before.st_size, "sha256": digest,
            "git_blob_oid": oid, "git_mode": mode,
            "keep_local": path.name in KEEP_LOCAL_NAMES,
            "mtime_ns": before.st_mtime_ns,
        })
        if number % 25 == 0 or number == len(names):
            print(f"Hashed {number}/{len(names)} models", flush=True)
    refs = []
    for line in command(
        ["git", "for-each-ref", "--format=%(refname) %(objecttype) %(objectname)"], root=root
    ).decode().splitlines():
        name, kind, oid = line.split()
        refs.append({"ref": name, "object_type": kind, "oid": oid})
    index_path = Path(command(["git", "rev-parse", "--git-path", "index"], root=root).decode().strip())
    if not index_path.is_absolute():
        index_path = root / index_path
    shutil.copy2(index_path, work_dir / "git-index.before")
    atomic_json(work_dir / "git-refs.before.json", refs)
    unique = {model["sha256"]: model["size_bytes"] for model in models}
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "source_head": command(["git", "rev-parse", "HEAD"], root=root).decode().strip(),
        "source_index_sha256": sha256_file(index_path),
        "source_refs": refs,
        "models": models,
        "summary": {
            "models": len(models), "unique_contents": len(unique),
            "file_bytes": sum(model["size_bytes"] for model in models),
            "unique_bytes": sum(unique.values()),
            "keep_local_models": sum(model["keep_local"] for model in models),
            "keep_local_bytes": sum(model["size_bytes"] for model in models if model["keep_local"]),
            "evictable_file_bytes": sum(model["size_bytes"] for model in models if not model["keep_local"]),
        },
    }
    atomic_json(output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    for model in manifest["models"]:
        safe_path(Path("/archive-root"), model["path"])
        if len(model["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in model["sha256"]):
            raise ValueError("Invalid model content hash")
    return manifest


def volume_plan(manifest: dict, max_bytes: int) -> list[list[dict]]:
    unique = {}
    for model in manifest["models"]:
        unique.setdefault(model["sha256"], model)
    groups, current, size = [], [], 10240
    for model in sorted(unique.values(), key=lambda item: item["sha256"]):
        needed = 512 + ((model["size_bytes"] + 511) // 512) * 512
        if needed + 10240 > max_bytes:
            raise ValueError(f"A model is too large for the configured volume size: {model['path']}")
        if current and size + needed > max_bytes:
            groups.append(current)
            current, size = [], 10240
        current.append(model)
        size += needed
    if current:
        groups.append(current)
    return groups


def verify_volume(path: Path, models: list[dict]) -> None:
    expected = {f"objects/{item['sha256']}.joblib": item for item in models}
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.name not in expected or not member.isfile() or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            item = expected[member.name]
            if member.size != item["size_bytes"]:
                raise ValueError(f"Archive member has wrong size: {member.name}")
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive member bytes: {member.name}")
            with stream:
                for chunk in iter(lambda stream=stream: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive is missing model contents")


def pack_volume(root: Path, target: Path, models: list[dict]) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
        for model in models:
            source = safe_path(root, model["path"])
            if source.is_symlink() or source.stat().st_size != model["size_bytes"]:
                raise ValueError(f"Source model changed: {model['path']}")
            member = tarfile.TarInfo(f"objects/{model['sha256']}.joblib")
            member.size = model["size_bytes"]
            member.mode = 0o644
            with source.open("rb") as stream:
                archive.addfile(member, stream)
    verify_volume(temporary, models)
    temporary.replace(target)
    return {
        "name": target.name, "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target), "contents": [item["sha256"] for item in models],
    }


def gh_json(arguments: list[str]) -> dict | list:
    return json.loads(command(["gh", *arguments]))


def release_assets(repository: str, tag: str) -> tuple[dict, dict[str, dict]]:
    # Draft tags may not exist as Git refs yet. The release listing exposes
    # authenticated drafts even when the by-tag endpoint returns 404.
    release_pages = gh_json([
        "api", "--paginate", "--slurp", f"repos/{repository}/releases?per_page=100",
    ])
    matching = [item for page in release_pages for item in page if item["tag_name"] == tag]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one accessible archive release for tag: {tag}")
    release = matching[0]
    pages = gh_json(["api", "--paginate", "--slurp", f"repos/{repository}/releases/{release['id']}/assets?per_page=100"])
    assets = {asset["name"]: asset for page in pages for asset in page}
    return release, assets


def verify_asset(asset: dict, info: dict) -> None:
    if (asset.get("state") != "uploaded" or asset.get("size") != info["size_bytes"]
            or asset.get("digest") != f"sha256:{info['sha256']}"):
        raise ValueError(f"Remote asset failed size/checksum verification: {info['name']}")


def upload(root: Path, manifest_path: Path, work_dir: Path, tag: str, max_bytes: int, workers: int = 4) -> None:
    manifest = load_manifest(manifest_path)
    repository = manifest["repository"]
    command(["gh", "auth", "status", "--hostname", "github.com"])
    repo = gh_json(["api", f"repos/{repository}"])
    if not repo.get("permissions", {}).get("push"):
        raise ValueError("The authenticated GitHub account cannot upload repository releases")
    work_dir.mkdir(parents=True, exist_ok=True)
    exists = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repository],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not exists:
        notes = work_dir / "release_notes.txt"
        notes.write_text(
            "Archived experiment model binaries with original paths and SHA256 checksums.\n"
            f"Workspace source HEAD: {manifest['source_head']}.\n"
            "The release tag identifies this archive; original run provenance stays unchanged.\n"
            "Use scripts/archive_experiment_models.py restore to recover model files.\n"
        )
        command([
            "gh", "release", "create", tag, "--repo", repository, "--draft",
            "--target", repo["default_branch"], "--title", tag, "--notes-file", str(notes),
        ])
    parts_path = work_dir / "archive_parts.json"
    saved = json.loads(parts_path.read_text()) if parts_path.exists() else {
        "schema_version": SCHEMA_VERSION, "manifest_sha256": sha256_file(manifest_path),
        "repository": repository, "tag": tag, "parts": [], "max_volume_bytes": max_bytes,
        "verified_parts": [],
    }
    if (saved["manifest_sha256"] != sha256_file(manifest_path) or saved["tag"] != tag
            or saved.get("max_volume_bytes", max_bytes) != max_bytes):
        raise ValueError("Existing upload state belongs to a different manifest/release")
    records = {part["name"]: part for part in saved["parts"]}
    groups = volume_plan(manifest, max_bytes)
    if len(groups) + 3 > 1000:
        raise ValueError("Archive would exceed the release asset-count limit")
    state_lock = threading.Lock()
    release, _ = release_assets(repository, tag)
    release_id = release["id"]

    def assets_now() -> dict[str, dict]:
        pages = gh_json([
            "api", "--paginate", "--slurp", f"repos/{repository}/releases/{release_id}/assets?per_page=100",
        ])
        return {asset["name"]: asset for page in pages for asset in page}

    def upload_one(number: int, models: list[dict]) -> None:
        name = f"models-{number:04d}.tar"
        target = work_dir / name
        assets = assets_now()
        info = records.get(name)
        if name in assets and info and assets[name]["state"] == "uploaded":
            verify_asset(assets[name], info)
        else:
            if name in assets and not info:
                raise ValueError(f"Remote asset exists without a matching local receipt: {name}")
            info = pack_volume(root, target, models)
            with state_lock:
                records[name] = info
                saved["parts"] = sorted(records.values(), key=lambda part: part["name"])
                atomic_json(parts_path, saved)
            for attempt in range(1, 6):
                assets = assets_now()
                if name in assets:
                    asset = assets[name]
                    if asset["state"] == "uploaded":
                        verify_asset(asset, info)
                        break
                    if asset["state"] != "starter" or asset["size"] != info["size_bytes"]:
                        raise ValueError(f"Unexpected unfinished remote asset: {name}")
                    # Only a failed, checksum-unverified upload from this journal
                    # is removed. No completed remote asset is overwritten.
                    command(["gh", "api", "--method", "DELETE", f"repos/{repository}/releases/assets/{asset['id']}"])
                print(f"Uploading volume {number}/{len(groups)}: {name} (attempt {attempt})", flush=True)
                try:
                    command(["gh", "release", "upload", tag, str(target), "--repo", repository])
                    assets = assets_now()
                    verify_asset(assets[name], info)
                    break
                except (subprocess.CalledProcessError, KeyError):
                    if attempt == 5:
                        raise
                    print(f"Retrying interrupted upload: {name}", flush=True)
                    time.sleep(min(5 * attempt, 30))
            else:
                raise ValueError(f"Upload could not be verified: {name}")
        with state_lock:
            saved.setdefault("verified_parts", [])
            if name not in saved["verified_parts"]:
                saved["verified_parts"].append(name)
                saved["verified_parts"].sort()
            atomic_json(parts_path, saved)
        print(f"Verified remote volume {number}/{len(groups)}: {name}", flush=True)
        # Only the reproducible temporary tar is removed; source models stay untouched.
        if target.exists():
            if sha256_file(target) != info["sha256"]:
                raise ValueError(f"Temporary archive changed: {name}")
            target.unlink()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        futures = [pool.submit(upload_one, number, models) for number, models in enumerate(groups, 1)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    for source, name in ((manifest_path, "model_archive_manifest.json"), (parts_path, "archive_parts.json")):
        _, assets = release_assets(repository, tag)
        info = {"name": name, "size_bytes": source.stat().st_size, "sha256": sha256_file(source)}
        if name in assets and assets[name]["state"] == "starter":
            if assets[name]["size"] != info["size_bytes"]:
                raise ValueError(f"Unexpected unfinished metadata asset: {name}")
            command(["gh", "api", "--method", "DELETE", f"repos/{repository}/releases/assets/{assets[name]['id']}"])
            del assets[name]
        if name not in assets:
            copy = work_dir / name
            if source.resolve() != copy.resolve():
                shutil.copy2(source, copy)
            command(["gh", "release", "upload", tag, str(copy), "--repo", repository])
            _, assets = release_assets(repository, tag)
        verify_asset(assets[name], info)
    command(["gh", "release", "edit", tag, "--draft=false", "--repo", repository])
    release, assets = release_assets(repository, tag)
    if release["draft"]:
        raise ValueError("Archive release was not published")
    atomic_json(work_dir / "upload_receipt.json", {
        "schema_version": SCHEMA_VERSION, "repository": repository, "tag": tag,
        "release_url": release["html_url"], "manifest_sha256": sha256_file(manifest_path),
        "parts_sha256": sha256_file(parts_path), "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_models_deleted": False,
    })
    print(f"All archive assets verified: {release['html_url']}", flush=True)


def restore_from_volume(volume: Path, models: list[dict], destination: Path) -> None:
    by_hash = {}
    for model in models:
        by_hash.setdefault(model["sha256"], []).append(model)
    with tarfile.open(volume, "r:") as archive:
        for digest, aliases in by_hash.items():
            member = archive.getmember(f"objects/{digest}.joblib")
            if not member.isfile() or member.size != aliases[0]["size_bytes"]:
                raise ValueError("Invalid model member")
            for model in aliases:
                target = safe_path(destination, model["path"])
                if target.exists():
                    if target.is_symlink() or sha256_file(target) != digest:
                        raise ValueError(f"Refusing to overwrite a different existing file: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing model member")
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    try:
                        with stream:
                            shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    if sha256_file(temporary) != digest:
                        raise ValueError(f"Restored model checksum mismatch: {model['path']}")
                    temporary.chmod(int(model["git_mode"], 8) & 0o777)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)


def restore(manifest_path: Path, parts_path: Path, destination: Path, work_dir: Path, prefix: str) -> None:
    manifest = load_manifest(manifest_path)
    parts = json.loads(parts_path.read_text())
    if parts["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Archive part index belongs to a different manifest")
    selected = [model for model in manifest["models"] if model["path"].startswith(prefix)]
    if not selected:
        raise ValueError("No archived models match the requested path prefix")
    work_dir.mkdir(parents=True, exist_ok=True)
    hashes = {model["sha256"] for model in selected}
    restored = set()
    for part in parts["parts"]:
        wanted = hashes.intersection(part["contents"])
        if not wanted:
            continue
        target = work_dir / part["name"]
        if not target.exists():
            command([
                "gh", "release", "download", parts["tag"], "--repo", parts["repository"],
                "--pattern", part["name"], "--dir", str(work_dir),
            ])
        if target.stat().st_size != part["size_bytes"] or sha256_file(target) != part["sha256"]:
            raise ValueError(f"Downloaded archive checksum mismatch: {target}")
        restore_from_volume(target, [model for model in selected if model["sha256"] in wanted], destination)
        restored.update(wanted)
        target.unlink()
    if restored != hashes:
        raise ValueError("Archive index is missing required model contents")
    print(f"Restored and verified {len(selected)} model files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="action", required=True)
    inspect = subparsers.add_parser("inventory", help="Hash newly staged models; do not change them")
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--work-dir", type=Path, required=True)
    inspect.add_argument("--repository", required=True)
    pack = subparsers.add_parser("pack-one", help="Pack and fully verify one local volume")
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--work-dir", type=Path, required=True)
    pack.add_argument("--volume", type=int, default=1)
    pack.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    send = subparsers.add_parser("upload", help="Resume verified release uploads; keep source models")
    send.add_argument("--manifest", type=Path, required=True)
    send.add_argument("--work-dir", type=Path, required=True)
    send.add_argument("--tag", required=True)
    send.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    recover = subparsers.add_parser("restore", help="Restore and verify models by original path prefix")
    recover.add_argument("--manifest", type=Path, required=True)
    recover.add_argument("--parts", type=Path, required=True)
    recover.add_argument("--destination", type=Path, required=True)
    recover.add_argument("--work-dir", type=Path, required=True)
    recover.add_argument("--prefix", required=True)
    args = parser.parse_args()
    try:
        if args.action == "inventory":
            inventory(args.root, args.manifest, args.work_dir, args.repository)
        elif args.action == "pack-one":
            groups = volume_plan(load_manifest(args.manifest), args.max_bytes)
            if not 1 <= args.volume <= len(groups):
                raise ValueError(f"Volume must be between 1 and {len(groups)}")
            info = pack_volume(args.root, args.work_dir / f"models-{args.volume:04d}.tar", groups[args.volume - 1])
            atomic_json(args.work_dir / f"models-{args.volume:04d}.receipt.json", info)
            print(json.dumps(info, indent=2))
        elif args.action == "upload":
            upload(args.root, args.manifest, args.work_dir, args.tag, args.max_bytes)
        elif args.action == "restore":
            restore(args.manifest, args.parts, args.destination, args.work_dir, args.prefix)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Archive stopped safely: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
