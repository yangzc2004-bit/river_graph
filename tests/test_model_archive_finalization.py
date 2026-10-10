"""Verify eviction gates and exact restoration of sanitized checkpoint trees."""

import json
import random
import subprocess
import sys

import pytest

from scripts import archive_experiment_models as archive
from scripts import finalize_model_archive as finalizer


@pytest.fixture
def archived_workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    git("init", "-q")
    git("config", "user.name", "Archive Test")
    git("config", "user.email", "archive-test@example.invalid")
    (root / "README.md").write_text("Keep committed history\n")
    git("add", "README.md")
    git("commit", "-qm", "Initial history")
    head = git("rev-parse", "HEAD").decode().strip()
    models = {
        "experiments/example/runs/seed42/forest.joblib": b"forest\0" * 83,
        "experiments/example/runs/seed43/forest.joblib": b"other\0" * 91,
        "experiments/example/exports/model.joblib": b"deploy\0" * 57,
    }
    for name, content in models.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "experiments/example/mask.npz").write_bytes(b"frozen mask bytes")
    git("add", "experiments")
    tree = git("write-tree").decode().strip()
    ref = "refs/codex/turn-diffs/checkpoints/archive-test"
    git("update-ref", ref, tree)
    work = tmp_path / "work"
    manifest = archive.inventory(root, tmp_path / "manifest.json", work, "owner/repo")
    return root, work, manifest, models, git, head, ref, tree


def test_sanitized_checkpoint_is_fully_recoverable_after_gc(archived_workspace, tmp_path):
    root, work, manifest, models, git, head, ref, tree = archived_workspace
    backup = finalizer.checkpoint_backup(root, manifest)
    assert backup["refs"][0]["original_oid"] == tree
    assert tree in backup["tree_objects_base64"]
    volume = tmp_path / "models.tar"
    archive.pack_volume(root, volume, archive.volume_plan(manifest, 32 * 1024)[0])
    removed = finalizer.evict_models(root, manifest, work)
    assert removed["removed_files"] == 2
    assert (root / "experiments/example/exports/model.joblib").exists()
    assert (root / "experiments/example/mask.npz").read_bytes() == b"frozen mask bytes"
    changes = finalizer.sanitize_checkpoints(root, backup, work)
    assert git("rev-parse", "HEAD").decode().strip() == head
    assert git("rev-parse", ref).decode().strip() == changes[0]["sanitized_oid"]
    assert b"joblib" not in git("ls-tree", "-r", ref)
    assert git("diff", "--cached", "--name-only").decode().splitlines() == ["experiments/example/mask.npz"]
    finalizer.sanitize_checkpoints(root, backup, work)  # Safe retry before reclamation.
    git("gc", "--prune=now")
    git("fsck", "--connectivity-only", "--no-dangling")
    archive.restore_from_volume(volume, manifest["models"], root)
    finalizer.restore_checkpoint_trees(root, backup, manifest)
    assert git("rev-parse", ref).decode().strip() == tree
    assert git("rev-parse", "HEAD").decode().strip() == head
    for name, content in models.items():
        assert (root / name).read_bytes() == content


def test_changed_model_blocks_all_eviction(archived_workspace):
    root, work, manifest, models, git, _, _, _ = archived_workspace
    original_index = git("diff", "--cached", "--raw")
    name = next(iter(models))
    (root / name).write_bytes(b"changed local work")
    with pytest.raises(ValueError, match="source model changed"):
        finalizer.evict_models(root, manifest, work)
    assert git("diff", "--cached", "--raw") == original_index
    assert all((root / path).exists() for path in models)


def test_new_staged_work_is_preserved_during_eviction(archived_workspace):
    root, work, manifest, models, git, _, _, _ = archived_workspace
    (root / "new-user-work.py").write_text("value = 42\n")
    git("add", "new-user-work.py")
    finalizer.evict_models(root, manifest, work)
    assert (root / "new-user-work.py").read_text() == "value = 42\n"
    assert "new-user-work.py" in git("diff", "--cached", "--name-only").decode().splitlines()
    assert all((root / path).exists() == (path.endswith("/model.joblib")) for path in models)


def test_eviction_resumes_after_journal_was_written_before_unlink(archived_workspace):
    root, work, manifest, models, _, _, _, _ = archived_workspace
    name = next(iter(models))
    (work / "eviction_progress.json").write_text(json.dumps({"removed": {name: len(models[name])}}))
    result = finalizer.evict_models(root, manifest, work)
    assert result["removed_files"] == 2
    assert not (root / name).exists()
    finalizer.evict_models(root, manifest, work)


def test_manifest_volume_size_can_bound_upload_retries(archived_workspace):
    _, _, manifest, _, _, _, _, _ = archived_workspace
    for group in archive.volume_plan(manifest, 16384):
        assert 10240 + sum(512 + ((item["size_bytes"] + 511) // 512) * 512 for item in group) <= 16384


def test_archive_records_keep_unstaged_work_and_push_original_history(archived_workspace, tmp_path):
    root, work, manifest, _, git, head, _, _ = archived_workspace
    git("branch", "-M", "main")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git("remote", "add", "origin", str(remote))
    git("push", "-q", "origin", "main")
    finalizer.evict_models(root, manifest, work)
    (root / "README.md").write_text("User work created after inventory\n")
    owned_script = root / "scripts/archive_experiment_models.py"
    owned_script.parent.mkdir()
    owned_script.write_text("# Administrative archive tool\n")

    (root / "new-user-work.py").write_text("value = 42\n")
    git("add", "new-user-work.py")
    commits = finalizer.archive_git_records(root, work, "test-archive", manifest)
    assert len(commits) == 1
    assert git("show", f"{commits[0]}:README.md") == b"Keep committed history\n"
    assert (root / "README.md").read_text() == "User work created after inventory\n"
    assert git("diff", "--cached", "--name-only").decode().splitlines() == ["new-user-work.py"]
    assert git("diff", "--name-only").decode().splitlines() == ["README.md"]
    assert b"joblib" not in git("ls-tree", "-r", "HEAD")
    git("merge-base", "--is-ancestor", head, "HEAD")

    finalizer.push_records_in_batches(root)
    assert git("rev-parse", "origin/main") == git("rev-parse", "HEAD")
    assert finalizer.archive_git_records(root, work, "test-archive", manifest) == commits


def test_new_commit_can_advance_without_committing_archive_models(archived_workspace):
    root, work, manifest, models, git, _, _, _ = archived_workspace
    finalizer.detach_archived_models(root, manifest, work)
    assert all((root / path).exists() for path in models)
    git("commit", "-qm", "Save frozen mask")
    finalizer.evict_models(root, manifest, work)
    assert (root / "experiments/example/mask.npz").read_bytes() == b"frozen mask bytes"


def test_newly_committed_archive_model_blocks_eviction(archived_workspace):
    root, work, manifest, models, git, _, _, _ = archived_workspace
    git("commit", "-qm", "Accidentally commit archived model files")
    with pytest.raises(ValueError, match="model was committed"):
        finalizer.evict_models(root, manifest, work)
    assert all((root / path).exists() for path in models)


def test_large_existing_commit_push_uses_temporary_branch_without_rewriting(archived_workspace, tmp_path):
    root, _, _, _, git, head, _, _ = archived_workspace
    git("branch", "-M", "main")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git("remote", "add", "origin", str(remote))
    git("push", "-q", "origin", "main")
    # Enforce an actual incoming-pack limit, including incompressible blobs.
    # This catches preloading at unrelated paths that still resends the blobs.
    subprocess.run(["git", "--git-dir", str(remote), "config", "receive.unpackLimit", "1"], check=True)
    hook = remote / "hooks/pre-receive"
    hook.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "quarantine = os.environ.get('GIT_QUARANTINE_PATH')\n"
        "packs = pathlib.Path(quarantine).glob('pack/*.pack') if quarantine else []\n"
        "sys.exit(1 if sum(p.stat().st_size for p in packs) > 4096 else 0)\n"
    )
    hook.chmod(0o755)
    generator = random.Random(42)
    contents = {}
    for number in range(8):
        contents[number] = generator.randbytes(1000)
        (root / f"artifact-{number}.bin").write_bytes(contents[number])
    git("add", ".")
    git("commit", "-qm", "Existing large artifact commit")
    original = git("rev-parse", "HEAD")

    finalizer.push_records_in_batches(root, max_raw_bytes=2048)
    assert git("rev-parse", "HEAD") == original
    assert git("rev-parse", "origin/main") == original
    git("merge-base", "--is-ancestor", head, "HEAD")
    assert b"archive-upload-buffer" not in git("ls-remote", "--heads", "origin")
    assert git("for-each-ref", "refs/heads/archive-upload-buffer-") == b""
    stored = subprocess.check_output(["git", "--git-dir", str(remote), "show", "main:artifact-7.bin"])
    assert stored == contents[7]
