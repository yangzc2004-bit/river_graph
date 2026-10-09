"""Integrity and recovery checks for experiment-model archiving."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import archive_experiment_models as archiving


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=root)


@pytest.fixture
def staged_models(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Archive Test")
    git(root, "config", "user.email", "archive-test@example.invalid")
    (root / "README.md").write_text("Existing history\n")
    git(root, "add", "README.md")
    git(root, "commit", "-qm", "Initial history")
    models = {
        "experiments/example/runs/seed42/forest.joblib": b"model\0" * 173,
        "experiments/example/runs/seed43/forest.joblib": b"model\0" * 173,
        "experiments/example/exports/model.joblib": b"deploy\0" * 79,
    }
    for name, content in models.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "experiments/example/mask.npz").write_bytes(b"keep mask")
    git(root, "add", "experiments")
    manifest_path = tmp_path / "manifest.json"
    before = git(root, "diff", "--cached", "--raw")
    manifest = archiving.inventory(root, manifest_path, tmp_path / "work", "owner/repo")
    assert git(root, "diff", "--cached", "--raw") == before
    return root, manifest_path, manifest, models


def test_deduplicated_archive_restores_original_paths_and_keeps_sources(staged_models, tmp_path):
    root, manifest_path, manifest, sources = staged_models
    assert manifest["summary"]["models"] == 3
    assert manifest["summary"]["unique_contents"] == 2
    assert manifest["summary"]["keep_local_models"] == 1
    assert len(archiving.load_manifest(manifest_path)["models"]) == 3
    groups = archiving.volume_plan(manifest, 32 * 1024)
    assert len(groups) == 1
    volume = tmp_path / "models.tar"
    info = archiving.pack_volume(root, volume, groups[0])
    assert info["size_bytes"] <= 32 * 1024
    assert info["sha256"] == hashlib.sha256(volume.read_bytes()).hexdigest()
    destination = tmp_path / "restored"
    archiving.restore_from_volume(volume, manifest["models"], destination)
    for name, content in sources.items():
        assert (destination / name).read_bytes() == content
        assert (root / name).read_bytes() == content
    assert (root / "experiments/example/mask.npz").read_bytes() == b"keep mask"
    archiving.restore_from_volume(volume, manifest["models"], destination)


def test_changed_model_cannot_produce_a_verified_archive(staged_models, tmp_path):
    root, _, manifest, sources = staged_models
    name = next(iter(sources))
    (root / name).write_bytes(b"x" * len(sources[name]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        archiving.pack_volume(root, tmp_path / "models.tar", archiving.volume_plan(manifest, 32 * 1024)[0])
    assert not (tmp_path / "models.tar").exists()
    assert (root / name).exists()


def test_corrupt_archive_is_rejected(staged_models, tmp_path):
    root, _, manifest, _ = staged_models
    models = archiving.volume_plan(manifest, 32 * 1024)[0]
    volume = tmp_path / "models.tar"
    archiving.pack_volume(root, volume, models)
    with tarfile.open(volume) as archive:
        offset = archive.getmembers()[0].offset_data
    with volume.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"X")
    with pytest.raises(ValueError, match="checksum mismatch"):
        archiving.verify_volume(volume, models)


def test_restore_rejects_existing_different_file_and_path_escape(staged_models, tmp_path):
    root, _, manifest, sources = staged_models
    volume = tmp_path / "models.tar"
    archiving.pack_volume(root, volume, archiving.volume_plan(manifest, 32 * 1024)[0])
    destination = tmp_path / "restored"
    name = next(iter(sources))
    target = destination / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"new local work")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        archiving.restore_from_volume(volume, manifest["models"], destination)
    assert target.read_bytes() == b"new local work"
    with pytest.raises(ValueError, match="Unsafe relative path"):
        archiving.safe_path(destination, "../../outside.joblib")
    (destination / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes destination"):
        archiving.safe_path(destination, "linked/outside.joblib")


def test_remote_asset_requires_server_checksum_and_uploaded_state():
    info = {"name": "models.tar", "size_bytes": 123, "sha256": "a" * 64}
    asset = {"state": "uploaded", "size": 123, "digest": "sha256:" + "a" * 64}
    archiving.verify_asset(asset, info)
    for bad in ({**asset, "digest": None}, {**asset, "size": 122}, {**asset, "state": "starter"}):
        with pytest.raises(ValueError, match="Remote asset failed"):
            archiving.verify_asset(bad, info)


def test_restore_manifest_rejects_unsafe_paths(staged_models, tmp_path):
    _, _, manifest, _ = staged_models
    manifest["models"][0]["path"] = "../outside.joblib"
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe relative path"):
        archiving.load_manifest(path)
