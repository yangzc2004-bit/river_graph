"""Finish a verified model archive and reclaim its local storage.

The worker can wait for an existing uploader, resume interrupted uploads, check
remote hashes, test restoration, back up checkpoint trees, and only then evict
unchanged models. Main-branch history and frozen research artifacts are kept.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import archive_experiment_models as archive
except ModuleNotFoundError:
    import archive_experiment_models as archive


def git(root: Path, arguments: list[str], data: bytes | None = None, index: Path | None = None) -> bytes:
    env = dict(os.environ)
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index.resolve())
    attempts = 5 if any(action in arguments for action in ("push", "fetch", "ls-remote")) else 1
    for attempt in range(1, attempts + 1):
        try:
            return subprocess.check_output(
                ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", *arguments],
                cwd=root, env=env, input=data,
            )
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"Retrying Git network operation ({attempt}/{attempts})", flush=True)
            time.sleep(30)
    raise RuntimeError("Git command did not produce a result")


def entries(root: Path, index: Path | None = None) -> dict[str, tuple[str, str]]:
    result = {}
    for item in git(root, ["ls-files", "--stage", "-z"], index=index).split(b"\0"):
        if not item:
            continue
        metadata, name = item.decode().split("\t", 1)
        mode, oid, stage = metadata.split()
        if stage != "0":
            raise ValueError("The Git index has unresolved conflicts")
        result[name] = (mode, oid)
    return result


def assert_original_index(root: Path, manifest: dict, index_backup: Path) -> None:
    baseline = entries(root, index_backup)
    current = entries(root)
    model_names = {model["path"] for model in manifest["models"]}
    for name, value in current.items():
        if name in model_names:
            if baseline.get(name) != value:
                raise ValueError(f"Staged model changed since archive inventory: {name}")
        elif baseline.get(name) != value:
            raise ValueError(f"The user changed the staged index during the upload: {name}")
    if {name: value for name, value in baseline.items() if name not in model_names} != {
        name: value for name, value in current.items() if name not in model_names
    }:
        raise ValueError("The user changed staged non-model files during the upload")
    if git(root, ["rev-parse", "HEAD"]).decode().strip() != manifest["source_head"]:
        raise ValueError("HEAD changed during the upload; review before finalizing")


def assert_archived_model_index(root: Path, manifest: dict, index_backup: Path) -> None:
    """Allow independent code work while protecting the archived model versions."""
    baseline = entries(root, index_backup)
    current = entries(root)
    tracked = {item.decode() for item in git(root, ["ls-tree", "-r", "--name-only", "-z", "HEAD"]).split(b"\0") if item}
    git(root, ["merge-base", "--is-ancestor", manifest["source_head"], "HEAD"])
    for model in manifest["models"]:
        name = model["path"]
        if name in tracked:
            raise ValueError(f"An archived model was committed during the upload: {name}")
        if name in current and current[name] != baseline.get(name):
            raise ValueError(f"Staged model changed since archive inventory: {name}")


def detach_archived_models(root: Path, manifest: dict, work_dir: Path) -> None:
    """Unstage only the inventoried model versions, retaining every local file."""
    assert_archived_model_index(root, manifest, work_dir / "git-index.before")
    names = b"".join(model["path"].encode() + b"\0" for model in manifest["models"])
    git(root, ["update-index", "--force-remove", "-z", "--stdin"], names)


def checkpoint_backup(root: Path, manifest: dict) -> dict:
    """Store exact tree bytes; non-model blobs stay referenced by sanitized trees."""
    archived = {model["git_blob_oid"] for model in manifest["models"]}
    refs, tree_oids = [], set()
    for line in git(root, ["for-each-ref", "--format=%(refname) %(objecttype) %(objectname)", "refs/codex/"]).decode().splitlines():
        name, kind, oid = line.split()
        if kind != "tree":
            continue
        removed, children = [], {oid}
        for item in git(root, ["ls-tree", "-r", "-t", "-z", oid]).split(b"\0"):
            if not item:
                continue
            metadata, path = item.decode().split("\t", 1)
            mode, object_type, child = metadata.split()
            if object_type == "tree":
                children.add(child)
            elif path.endswith(".joblib") and child in archived:
                removed.append({"path": path, "git_mode": mode, "oid": child})
        if not removed:
            continue
        log_path = Path(git(root, ["rev-parse", "--git-path", f"logs/{name}"]).decode().strip())
        if not log_path.is_absolute():
            log_path = root / log_path
        refs.append({
            "ref": name, "original_oid": oid, "removed_models": removed,
            "reflog_base64": base64.b64encode(log_path.read_bytes()).decode() if log_path.exists() else None,
        })
        tree_oids.update(children)
    tree_objects = {}
    if tree_oids:
        raw = git(root, ["cat-file", "--batch"], ("\n".join(sorted(tree_oids)) + "\n").encode())
        position = 0
        while position < len(raw):
            end = raw.index(b"\n", position)
            oid, kind, length = raw[position:end].decode().split()
            if kind != "tree":
                raise ValueError("Checkpoint backup contains an unexpected object type")
            size = int(length)
            content = raw[end + 1:end + 1 + size]
            if hashlib.sha1(f"tree {size}\0".encode() + content).hexdigest() != oid:
                raise ValueError("Checkpoint tree failed identity verification")
            tree_objects[oid] = base64.b64encode(content).decode()
            position = end + size + 2
    return {
        "schema_version": 1, "source_head": manifest["source_head"],
        "model_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "refs": refs, "tree_objects_base64": tree_objects,
    }


def sanitize_checkpoints(root: Path, backup: dict, work_dir: Path) -> list[dict]:
    result = []
    for record in backup["refs"]:
        ref = record["ref"]
        if not ref.startswith("refs/codex/"):
            raise ValueError("Refusing to change a non-checkpoint Git ref")
        current = git(root, ["rev-parse", ref]).decode().strip()
        scratch = work_dir / "checkpoint-index"
        scratch.unlink(missing_ok=True)
        git(root, ["read-tree", record["original_oid"]], index=scratch)
        paths = b"".join(item["path"].encode() + b"\0" for item in record["removed_models"])
        git(root, ["update-index", "--force-remove", "-z", "--stdin"], paths, index=scratch)
        replacement = git(root, ["write-tree"], index=scratch).decode().strip()
        if current not in {record["original_oid"], replacement}:
            raise ValueError(f"Checkpoint changed after its verified backup: {ref}")
        if current != replacement:
            git(root, ["update-ref", ref, replacement, current])
        # Reflogs for these backed-up tool trees must not retain archived blobs.
        log_path = Path(git(root, ["rev-parse", "--git-path", f"logs/{ref}"]).decode().strip())
        if not log_path.is_absolute():
            log_path = root / log_path
        if log_path.exists():
            git(root, ["reflog", "expire", "--expire=now", "--expire-unreachable=now", ref])
        result.append({"ref": ref, "original_oid": record["original_oid"], "sanitized_oid": replacement})
    return result


def restore_checkpoint_trees(root: Path, backup: dict, manifest: dict) -> None:
    """Recreate original checkpoint refs after models have been restored."""
    model_by_oid = {model["git_blob_oid"]: model for model in manifest["models"]}
    needed = {item["oid"] for record in backup["refs"] for item in record["removed_models"]}
    for oid in needed:
        exists = subprocess.run(
            ["git", "cat-file", "-e", oid], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
        if exists:
            continue
        model = model_by_oid[oid]
        path = archive.safe_path(root, model["path"])
        if archive.sha256_file(path) != model["sha256"]:
            raise ValueError(f"Restore the original model bytes first: {model['path']}")
        actual = git(root, ["hash-object", "-w", "--no-filters", "--", str(path)]).decode().strip()
        if actual != oid:
            raise ValueError("Restored Git model object has the wrong identity")
    for oid, encoded in backup["tree_objects_base64"].items():
        content = base64.b64decode(encoded, validate=True)
        actual = git(root, ["hash-object", "-w", "-t", "tree", "--stdin"], content).decode().strip()
        if actual != oid:
            raise ValueError("Restored checkpoint tree has the wrong identity")
    for record in backup["refs"]:
        if not record["ref"].startswith("refs/codex/"):
            raise ValueError("Invalid checkpoint ref")
        git(root, ["update-ref", record["ref"], record["original_oid"]])
        if record.get("reflog_base64") is not None:
            log_path = Path(git(root, ["rev-parse", "--git-path", f"logs/{record['ref']}"]).decode().strip())
            if not log_path.is_absolute():
                log_path = root / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(base64.b64decode(record["reflog_base64"], validate=True))
    git(root, ["fsck", "--connectivity-only", "--no-dangling"])


def evict_models(root: Path, manifest: dict, work_dir: Path) -> dict:
    assert_archived_model_index(root, manifest, work_dir / "git-index.before")
    progress_path = work_dir / "eviction_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"removed": {}}
    # Check every remaining file before making any change to index or models.
    for model in manifest["models"]:
        path = archive.safe_path(root, model["path"])
        if not path.exists():
            if model["path"] not in progress["removed"]:
                raise ValueError(f"A source model disappeared outside this archive job: {model['path']}")
            continue
        if path.is_symlink() or path.stat().st_size != model["size_bytes"] or archive.sha256_file(path) != model["sha256"]:
            raise ValueError(f"A source model changed; no eviction performed: {model['path']}")
    names = b"".join(model["path"].encode() + b"\0" for model in manifest["models"])
    git(root, ["update-index", "--force-remove", "-z", "--stdin"], names)
    for model in manifest["models"]:
        if model["keep_local"]:
            continue
        path = archive.safe_path(root, model["path"])
        # Record intent before unlink so a crash can resume without guessing.
        progress["removed"][model["path"]] = model["size_bytes"]
        archive.atomic_json(progress_path, progress)
        path.unlink(missing_ok=True)
    return {"removed_files": len(progress["removed"]), "removed_file_bytes": sum(progress["removed"].values())}


def ensure_asset(repository: str, tag: str, path: Path) -> dict:
    info = {"name": path.name, "size_bytes": path.stat().st_size, "sha256": archive.sha256_file(path)}
    _, assets = archive.release_assets(repository, tag)
    if path.name not in assets:
        archive.command(["gh", "release", "upload", tag, str(path), "--repo", repository])
        _, assets = archive.release_assets(repository, tag)
    archive.verify_asset(assets[path.name], info)
    return {**info, "download_url": assets[path.name]["browser_download_url"]}


def verify_remote_archive(manifest_path: Path, work_dir: Path, tag: str) -> dict:
    manifest = archive.load_manifest(manifest_path)
    parts = json.loads((work_dir / "archive_parts.json").read_text())
    if parts["manifest_sha256"] != archive.sha256_file(manifest_path) or parts["tag"] != tag:
        raise ValueError("Upload receipt does not match this archive")
    release, assets = archive.release_assets(manifest["repository"], tag)
    expected_hashes = {model["sha256"] for model in manifest["models"]}
    recorded = [digest for part in parts["parts"] for digest in part["contents"]]
    if len(recorded) != len(set(recorded)) or set(recorded) != expected_hashes or release["draft"]:
        raise ValueError("The archive release is incomplete or contains duplicate model contents")
    for part in parts["parts"]:
        archive.verify_asset(assets[part["name"]], part)
    for path, name in ((manifest_path, "model_archive_manifest.json"), (work_dir / "archive_parts.json", "archive_parts.json")):
        archive.verify_asset(assets[name], {
            "name": name, "size_bytes": path.stat().st_size, "sha256": archive.sha256_file(path),
        })
    return release


def verify_real_model_restore(root: Path, manifest_path: Path, work_dir: Path) -> dict:
    manifest = archive.load_manifest(manifest_path)
    choices = [model for model in manifest["models"] if Path(model["path"]).name == "forests.joblib"]
    selected = min(choices, key=lambda model: model["size_bytes"])
    destination = work_dir / "restore-check"
    archive.restore(
        manifest_path, work_dir / "archive_parts.json", destination,
        work_dir / "restore-downloads", selected["path"],
    )
    original = archive.safe_path(root, selected["path"])
    restored = archive.safe_path(destination, selected["path"])
    import joblib
    import numpy as np

    def estimators(value):
        import dataclasses

        if hasattr(value, "predict") and hasattr(value, "n_features_in_"):
            return [value]
        if isinstance(value, dict):
            return [model for item in value.values() for model in estimators(item)]
        if isinstance(value, (list, tuple)):
            return [model for item in value for model in estimators(item)]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [model for field in dataclasses.fields(value) for model in estimators(getattr(value, field.name))]
        return []

    before, after = estimators(joblib.load(original)), estimators(joblib.load(restored))
    if not before or len(before) != len(after):
        raise ValueError("Restored forest bundle could not be validated with actual model predictions")
    for source, recovered in zip(before, after, strict=True):
        if source.n_features_in_ != recovered.n_features_in_:
            raise ValueError("Restored model feature dimensions differ")
        if hasattr(source, "n_jobs"):
            source.n_jobs = recovered.n_jobs = 1
        sample = np.zeros((2, int(source.n_features_in_)), dtype=np.float32)
        np.testing.assert_array_equal(source.predict(sample), recovered.predict(sample))
    return {"model_path": selected["path"], "sha256": selected["sha256"], "estimators_checked": len(before)}


def process_matches(pid: int) -> bool:
    if pid <= 0:
        return False
    check = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    return check.returncode == 0 and "archive_experiment_models.py upload" in check.stdout


def directory_bytes(path: Path) -> int:
    return int(archive.command(["du", "-sk", str(path)]).decode().split()[0]) * 1024


def archive_git_records(root: Path, work_dir: Path, tag: str, manifest: dict | None = None) -> list[str]:
    own_paths = [
        ".gitignore", "scripts/archive_experiment_models.py", "scripts/finalize_model_archive.py",
        "tests/test_model_archiving.py", "tests/test_model_archive_finalization.py",
        "experiments/analysis/model_archive_20261009",
    ]
    existing = [path for path in own_paths if (root / path).exists()]
    git(root, ["add", "--", *existing])
    target = entries(root)
    names = sorted(item.decode() for item in git(root, ["diff", "--cached", "--name-only", "-z"]).split(b"\0") if item)
    if manifest is not None:
        baseline = entries(root, work_dir / "git-index.before")
        original_names = {
            item.decode() for item in git(
                root, ["diff", "--cached", manifest["source_head"], "--name-only", "-z"],
                index=work_dir / "git-index.before",
            ).split(b"\0") if item
        }
        def owned(name):
            return any(name == path or name.startswith(path + "/") for path in own_paths)

        selected = [name for name in names if owned(name) or (name in original_names and target.get(name) == baseline.get(name))]
        selected_names = set(selected)
        archive.atomic_json(work_dir / "preserved_concurrent_staging.json", {
            "paths_left_staged": [name for name in names if name not in selected_names],
        })
        names = selected
    ids = {target[name][1] for name in names if name in target}
    sizes = {}
    if ids:
        for line in git(root, ["cat-file", "--batch-check=%(objectname) %(objectsize)"], ("\n".join(ids) + "\n").encode()).decode().splitlines():
            oid, size = line.split()
            sizes[oid] = int(size)
    names.sort(key=lambda name: (not name.startswith(("src/", "scripts/", "tests/", "docs/", "pyproject.toml", ".gitignore")), name))
    groups, current, size = [], [], 0
    for name in names:
        amount = sizes.get(target[name][1], 0) if name in target else 0
        if amount > 512 * 1024 * 1024:
            raise ValueError(f"A remaining Git artifact needs separate review before pushing: {name}")
        if current and size + amount > 512 * 1024 * 1024:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += amount
    if current:
        groups.append(current)
    commits_path = work_dir / "record_commits.json"
    commits = json.loads(commits_path.read_text()) if commits_path.exists() else []
    for number, group in enumerate(groups, 1):
        scratch = work_dir / "commit-index"
        scratch.unlink(missing_ok=True)
        git(root, ["read-tree", "HEAD"], index=scratch)
        changes = []
        for name in group:
            mode, oid = target.get(name, ("0", "0" * 40))
            changes.append(f"{mode} {oid}\t{name}".encode() + b"\0")
        git(root, ["update-index", "-z", "--index-info"], b"".join(changes), index=scratch)
        message = f"Preserve experiment records after model archival ({number}/{len(groups)})"
        git(root, ["commit", "-m", message, "-m", f"Model binaries are preserved in GitHub Release {tag}; original predictions and provenance bytes are retained."], index=scratch)
        commits.append(git(root, ["rev-parse", "HEAD"]).decode().strip())
        archive.atomic_json(work_dir / "record_commits.json", commits)
    return commits


def preload_commit_blobs(root: Path, auth: list[str], base: str, target: str, max_bytes: int) -> str:
    """Push large commits' blobs through a temporary branch, preserving their hashes."""
    ref = f"refs/heads/archive-upload-buffer-{target[:12]}"
    existing = git(root, [*auth, "ls-remote", "--heads", "origin", ref]).decode().strip()
    parent = base
    if existing:
        parent = existing.split()[0]
        git(root, [*auth, "fetch", "--no-tags", "origin", ref])
        message = git(root, ["show", "-s", "--format=%s", parent]).decode()
        if not message.startswith(f"Temporary upload buffer for original commit {target} ("):
            raise ValueError(f"An unrelated remote branch uses the upload buffer name: {ref}")
        git(root, ["merge-base", "--is-ancestor", base, parent])
    else:
        local = subprocess.run(["git", "show-ref", "--verify", "--quiet", ref], cwd=root, check=False)
        if local.returncode == 0:
            message = git(root, ["show", "-s", "--format=%s", ref]).decode()
            if not message.startswith(f"Temporary upload buffer for original commit {target} ("):
                raise ValueError(f"An unrelated local branch uses the upload buffer name: {ref}")
    ids = {line.split()[0] for line in git(root, ["rev-list", "--objects", target, f"^{base}", f"^{parent}"]).decode().splitlines()}
    blobs = []
    if ids:
        for line in git(root, ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"], ("\n".join(sorted(ids)) + "\n").encode()).decode().splitlines():
            oid, kind, size = line.split()
            if kind == "blob":
                if int(size) > 100 * 1024 * 1024:
                    raise ValueError(f"A historical Git blob exceeds GitHub's file limit: {oid}")
                blobs.append((oid, int(size)))
    groups, group, amount = [], [], 0
    for oid, size in blobs:
        if group and amount + size > max_bytes:
            groups.append(group)
            group, amount = [], 0
        group.append(oid)
        amount += size
    if group:
        groups.append(group)
    # Use the original paths. Git's tree traversal may resend shared blobs when
    # they are reachable only under unrelated paths in an excluded commit.
    pending = {oid for oid, _ in blobs}
    paths_by_oid = {}
    for item in git(root, ["ls-tree", "-r", "-z", target]).split(b"\0"):
        if not item:
            continue
        metadata, name = item.decode().split("\t", 1)
        _mode, kind, oid = metadata.split()
        if kind == "blob" and oid in pending:
            paths_by_oid.setdefault(oid, []).append(name)
    if pending != set(paths_by_oid):
        raise ValueError("The large commit contains historical blobs absent from its final tree")
    scratch = root / ".git" / "archive-upload-buffer-index"
    try:
        for number, group in enumerate(groups, 1):
            scratch.unlink(missing_ok=True)
            git(root, ["read-tree", target], index=scratch)
            pending.difference_update(group)
            names = b"".join(name.encode() + b"\0" for oid in pending for name in paths_by_oid[oid])
            if names:
                git(root, ["update-index", "--force-remove", "-z", "--stdin"], names, index=scratch)
            tree = git(root, ["write-tree"], index=scratch).decode().strip()
            message = f"Temporary upload buffer for original commit {target} ({number}/{len(groups)})\n"
            commit = git(root, ["commit-tree", tree, "-p", parent], message.encode()).decode().strip()
            git(root, ["update-ref", ref, commit])
            git(root, [*auth, "push", "origin", f"{commit}:{ref}"])
            parent = commit
    finally:
        scratch.unlink(missing_ok=True)
    return ref


def push_original_with_buffer(root: Path, auth: list[str], target: str, ref: str) -> None:
    """Include the buffer update so its old tree bounds the outgoing pack."""
    remote_line = git(root, [*auth, "ls-remote", "--heads", "origin", ref]).decode().strip()
    if not remote_line:
        raise ValueError("The temporary upload branch disappeared")
    parent = remote_line.split()[0]
    tree = git(root, ["rev-parse", f"{parent}^{{tree}}"]).decode().strip()
    message = f"Temporary upload buffer for original commit {target} (ready)\n"
    ready = git(root, ["commit-tree", tree, "-p", parent], message.encode()).decode().strip()
    git(root, ["update-ref", ref, ready])
    git(root, [*auth, "push", "origin", f"{target}:refs/heads/main", f"{ready}:{ref}"])


def cleanup_completed_upload_buffers(root: Path, auth: list[str], main: str) -> None:
    lines = git(root, [*auth, "ls-remote", "--heads", "origin", "refs/heads/archive-upload-buffer-*"]).decode().splitlines()
    prefix = "Temporary upload buffer for original commit "
    for line in lines:
        oid, ref = line.split()
        git(root, [*auth, "fetch", "--no-tags", "origin", ref])
        message = git(root, ["show", "-s", "--format=%s", oid]).decode().strip()
        target = message[len(prefix):].split(" ", 1)[0] if message.startswith(prefix) else ""
        if len(target) != 40 or any(character not in "0123456789abcdef" for character in target):
            continue
        if ref != f"refs/heads/archive-upload-buffer-{target[:12]}":
            continue
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", target, main], cwd=root, check=False)
        if ancestor.returncode == 0:
            git(root, [*auth, "push", "origin", f":{ref}"])
            git(root, ["update-ref", "-d", ref])


def push_records_in_batches(root: Path, max_raw_bytes: int = 1024 * 1024 * 1024) -> None:
    # Use the authenticated CLI for this task without changing global Git helpers.
    auth = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    remote_line = git(root, [*auth, "ls-remote", "--heads", "origin", "main"]).decode().strip()
    if not remote_line:
        raise ValueError("The remote main branch could not be identified")
    remote = remote_line.split()[0]
    git(root, [*auth, "fetch", "--no-tags", "origin", "main"])
    git(root, ["merge-base", "--is-ancestor", remote, "HEAD"])
    cleanup_completed_upload_buffers(root, auth, remote)
    commits = git(root, ["rev-list", "--reverse", "--first-parent", f"{remote}..HEAD"]).decode().splitlines()
    all_objects = {line.split()[0] for line in git(root, ["rev-list", "--objects", f"{remote}..HEAD"]).decode().splitlines()}
    sizes = {}
    if all_objects:
        for line in git(root, ["cat-file", "--batch-check=%(objectname) %(objectsize)"], ("\n".join(all_objects) + "\n").encode()).decode().splitlines():
            oid, size = line.split()
            sizes[oid] = int(size)
    position = 0
    while position < len(commits):
        low, high, best = position, len(commits) - 1, None
        while low <= high:
            middle = (low + high) // 2
            objects = {line.split()[0] for line in git(root, ["rev-list", "--objects", f"{remote}..{commits[middle]}"]).decode().splitlines()}
            amount = sum(sizes[oid] for oid in objects)
            if amount <= max_raw_bytes:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best is None:
            target = commits[position]
            buffer_ref = preload_commit_blobs(root, auth, remote, target, max_raw_bytes // 2)
            push_original_with_buffer(root, auth, target, buffer_ref)
            git(root, [*auth, "push", "origin", f":{buffer_ref}"])
            git(root, ["update-ref", "-d", buffer_ref])
            remote, position = target, position + 1
            continue
        target = commits[best]
        git(root, [*auth, "push", "origin", f"{target}:refs/heads/main"])
        remote, position = target, best + 1
    git(root, [*auth, "fetch", "--no-tags", "origin", "main"])
    if git(root, ["rev-parse", "HEAD"]).strip() != git(root, ["rev-parse", "origin/main"]).strip():
        raise ValueError("Remote main does not match the completed local archive commits")


def run_worker(root: Path, manifest_path: Path, work_dir: Path, tag: str, uploader_pid: int) -> None:
    import fcntl

    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = work_dir / "execution_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "started_at": datetime.now(timezone.utc).isoformat(), "source_models_deleted": False,
            "git_bytes_before": directory_bytes(root / ".git"),
            "free_bytes_before": os.statvfs(root).f_bavail * os.statvfs(root).f_frsize,
        }

        def update(phase: str, **fields) -> None:
            if phase != "needs_attention":
                state.pop("error", None)
                state.pop("error_type", None)
            state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
            archive.atomic_json(state_path, state)
            print(f"{state['updated_at']} {phase}", flush=True)

        try:
            manifest = archive.load_manifest(manifest_path)
            detach_archived_models(root, manifest, work_dir)
            update("uploading", repository=manifest["repository"], tag=tag)
            while process_matches(uploader_pid):
                progress = work_dir / "archive_parts.json"
                if progress.exists():
                    data = json.loads(progress.read_text())
                    update("uploading", prepared_volumes=len(data["parts"]),
                           verified_volumes=len(data.get("verified_parts", [])))
                time.sleep(30)
            if not (work_dir / "upload_receipt.json").exists():
                for attempt in range(1, 21):
                    try:
                        archive.upload(root, manifest_path, work_dir, tag, archive.DEFAULT_MAX_BYTES)
                        break
                    except subprocess.CalledProcessError:
                        if attempt == 20:
                            raise
                        update("retrying_network_upload", attempt=attempt)
                        time.sleep(60)
            update("verifying_remote_archive")
            release = verify_remote_archive(manifest_path, work_dir, tag)
            report_dir = manifest_path.parent
            if not state.get("restore_verified"):
                update("testing_restoration", release_url=release["html_url"])
                restore_result = verify_real_model_restore(root, manifest_path, work_dir)
                archive.atomic_json(report_dir / "restore_verification.json", restore_result)
                update("restoration_verified", restore_verified=True)
            if not state.get("checkpoint_backup_verified"):
                assert_archived_model_index(root, manifest, work_dir / "git-index.before")
                update("backing_up_git_checkpoints")
                backup = checkpoint_backup(root, manifest)
                compressed = gzip.compress(json.dumps(backup, sort_keys=True).encode(), mtime=0)
                digest = hashlib.sha256(compressed).hexdigest()
                backup_path = report_dir / f"git-checkpoint-trees-{digest[:16]}.json.gz"
                backup_path.write_bytes(compressed)
                receipt = ensure_asset(manifest["repository"], tag, backup_path)
                archive.atomic_json(report_dir / "git_checkpoint_backup_receipt.json", receipt)
                update("checkpoint_backup_verified", checkpoint_backup_verified=True,
                       checkpoint_backup_path=str(backup_path.resolve()))
            backup = json.loads(gzip.decompress(Path(state["checkpoint_backup_path"]).read_bytes()))
            if not state.get("eviction_complete"):
                update("evicting_verified_models")
                eviction = evict_models(root, manifest, work_dir)
                update("models_evicted", eviction_complete=True, source_models_deleted=True, **eviction)
            if not state.get("checkpoints_sanitized"):
                update("sanitizing_backed_up_checkpoints")
                changed = sanitize_checkpoints(root, backup, work_dir)
                archive.atomic_json(report_dir / "sanitized_checkpoints.json", changed)
                git(root, ["fsck", "--connectivity-only", "--no-dangling"])
                update("checkpoints_sanitized", checkpoints_sanitized=True)
            if not state.get("git_gc_complete"):
                update("reclaiming_git_objects")
                git(root, ["-c", "pack.threads=2", "-c", "pack.windowMemory=128m", "gc", "--prune=2.hours.ago"])
                git(root, ["fsck", "--connectivity-only", "--no-dangling"])
                after = directory_bytes(root / ".git")
                update("git_objects_reclaimed", git_gc_complete=True, git_bytes_after=after,
                       git_bytes_reclaimed=state["git_bytes_before"] - after)
            if not state.get("records_saved"):
                update("saving_archive_records")
                archive.atomic_json(report_dir / "archive_parts.json", json.loads((work_dir / "archive_parts.json").read_text()))
                archive.atomic_json(report_dir / "upload_receipt.json", json.loads((work_dir / "upload_receipt.json").read_text()))
                commits = archive_git_records(root, work_dir, tag, manifest)
                update("records_saved", records_saved=True, record_commits=len(commits),
                       records_head=git(root, ["rev-parse", "HEAD"]).decode().strip())
            git(root, ["merge-base", "--is-ancestor", manifest["source_head"], "HEAD"])
            if git(root, ["rev-parse", "HEAD"]).decode().strip() != state["records_head"]:
                raise ValueError("HEAD changed after saving archive records; review before pushing")
            update("pushing_git_records", main_history_preserved=True)
            push_records_in_batches(root)
            free_after = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            update("complete", free_bytes_after=free_after,
                   filesystem_free_bytes_increase=free_after - state["free_bytes_before"],
                   local_head=git(root, ["rev-parse", "HEAD"]).decode().strip(),
                   remote_head=git(root, ["rev-parse", "origin/main"]).decode().strip())
        except BaseException as exc:
            update("needs_attention", error_type=type(exc).__name__, error=str(exc))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--uploader-pid", type=int, default=0)
    parser.add_argument("--restore-checkpoints", type=Path)
    args = parser.parse_args()
    if args.restore_checkpoints:
        backup = json.loads(gzip.decompress(args.restore_checkpoints.read_bytes()))
        restore_checkpoint_trees(args.root, backup, archive.load_manifest(args.manifest))
    else:
        run_worker(args.root, args.manifest, args.work_dir, args.tag, args.uploader_pid)


if __name__ == "__main__":
    main()
