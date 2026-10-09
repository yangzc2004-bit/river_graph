"""Which stored prediction files an analysis is allowed to read (A0 policy).

Prediction file names are NOT unique across the two batches: the conflicted
``G0_gcn_none__e1_r20_seed42.parquet`` exists in both
``experiments/predictions/`` (overwritten by a different run) and
``experiments/predictions_historical/`` (the recovered, frozen-table-matching
copy). Selecting by file name alone therefore lets a deliberately excluded file
back into the analysis, and it silently double-counts the test cells of that
(mask, station) set.

This module keys the allow-list by (batch directory, file name, sha256) so that
exactly one source is chosen per (model, mask) and a file whose content changed
since the manifest was frozen is reported instead of used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CURRENT_DIR = Path("experiments/predictions")
HISTORICAL_DIR = Path("experiments/predictions_historical")
MANIFEST = Path("experiments/frozen_results/prediction_manifest_20260912.json")

# Directory name in the manifest -> path on disk.
BATCH_DIRS = {
    "predictions": CURRENT_DIR,
    "predictions_historical": HISTORICAL_DIR,
}


class PredictionSourceError(RuntimeError):
    """Raised when the on-disk predictions disagree with the frozen manifest."""


@dataclass(frozen=True)
class PredictionSource:
    """One verified, uniquely selected prediction file."""

    path: Path
    batch: str
    model: str
    mask: str
    sha256: str

    @property
    def key(self) -> str:
        return f"{self.model}__{self.mask}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def model_mask_of(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        raise PredictionSourceError(f"unexpected prediction file name: {path.name}")
    model, mask = stem.split("__", 1)
    return model, mask


def load_manifest(path: Path = MANIFEST) -> dict:
    if not path.exists():
        raise PredictionSourceError(
            f"manifest {path} not found; run scripts/build_prediction_manifest.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources(
    manifest_path: Path = MANIFEST,
    verify_hashes: bool = True,
) -> tuple[list[PredictionSource], dict]:
    """Return the verified allow-list plus a report of what was excluded.

    Every selected file must appear in the manifest with a matching sha256. A
    file present on disk but absent from the manifest is never selected, even
    if its name matches an allowed one.
    """
    manifest = load_manifest(manifest_path)

    allowed: dict[tuple[str, str], dict] = {}
    for entry in manifest["predictions"]:
        if not entry.get("used_for_phase_a"):
            continue
        allowed[(entry["batch"], entry["file"])] = entry
    for entry in manifest.get("historical_recovered", []):
        allowed[(entry["batch"], entry["file"])] = entry

    excluded: dict[tuple[str, str], dict] = {}
    for group in ("conflicts", "zero_coverage"):
        for entry in manifest.get(group, []):
            if group == "conflicts" and entry.get("used_for_phase_a", False):
                continue
            excluded[(entry.get("batch", "predictions"), entry["file"])] = entry

    sources: list[PredictionSource] = []
    report = {
        "allowed_entries": len(allowed),
        "selected": 0,
        "excluded_entries": len(excluded),
        "missing": [],
        "hash_changed": [],
        "pin_mismatch": [],
        "duplicate_keys": [],
        "unmanifested_on_disk": [],
    }
    seen_keys: dict[str, str] = {}

    for (batch, name), entry in sorted(allowed.items()):
        path = BATCH_DIRS.get(batch, Path(batch)) / name
        if not path.exists():
            report["missing"].append(f"{batch}/{name}")
            continue
        digest = sha256_file(path)
        # The pin is authoritative: it is carried forward across manifest
        # rebuilds, so editing `sha256` by hand cannot re-admit replaced bytes.
        expected = entry.get("pinned_sha256") or entry.get("sha256")
        pinned = entry.get("pinned_sha256")
        if verify_hashes and pinned and digest != pinned:
            report["pin_mismatch"].append(
                f"{batch}/{name}: pinned {pinned[:12]} != disk {digest[:12]}")
            continue
        if verify_hashes and expected and digest != expected:
            report["hash_changed"].append(
                f"{batch}/{name}: manifest {expected[:12]} != disk {digest[:12]}")
            continue
        model, mask = model_mask_of(path)
        key = f"{model}__{mask}"
        if key in seen_keys:
            report["duplicate_keys"].append(
                f"{key}: already selected from {seen_keys[key]}, also present in "
                f"{batch}/{name}")
            continue
        seen_keys[key] = f"{batch}/{name}"
        sources.append(PredictionSource(path=path, batch=batch, model=model,
                                        mask=mask, sha256=digest))

    # Files on disk that no entry covers: report them, never use them.
    known_names = {name for (_batch, name) in allowed} | \
                  {name for (_batch, name) in excluded}
    for batch, directory in BATCH_DIRS.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.parquet")):
            if path.name not in known_names:
                report["unmanifested_on_disk"].append(f"{batch}/{path.name}")

    report["selected"] = len(sources)
    return sources, report


def describe_report(report: dict) -> str:
    lines = [
        f"allowed entries   : {report['allowed_entries']}",
        f"selected files    : {report['selected']}",
        f"excluded entries  : {report['excluded_entries']}",
    ]
    for key in ("missing", "hash_changed", "pin_mismatch", "duplicate_keys",
                "unmanifested_on_disk"):
        values = report[key]
        lines.append(f"{key:18}: {len(values)}")
        for value in values:
            lines.append(f"    - {value}")
    return "\n".join(lines)
