"""Record inspectable preparation provenance; this is not a training freeze."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prepare_aqua_oc_data_v1 import OUTPUT, ROOT, digest, write_json


def main() -> None:
    # Compact a legacy probe inventory without changing its access measurements.
    path = OUTPUT / "archive_access_probes.json"
    probes = json.loads(path.read_text())
    for probe in probes:
        if "files" in probe:
            files = probe.pop("files")
            probe["file_count"] = len(files)
            probe["implementation_files"] = [name for name in files if name.endswith((".py", ".ipynb"))]
    write_json(path, probes)
    code = ["scripts/prepare_aqua_oc_data_v1.py", "scripts/probe_aqua_oc_sources_v1.py",
            "scripts/probe_aqua_oc_archives_v1.py", "scripts/audit_aqua_oc_data_v1.py",
            "scripts/record_aqua_oc_preparation_v1.py", "tests/test_aqua_oc_acquisition_v1.py",
            "docs/aqua_oc_reproduction_status_20261011.md"]
    artifacts = [str(path.relative_to(ROOT)) for path in sorted(OUTPUT.glob("*.json"))
                 if path.name != "preparation_manifest.json"]
    data = ["cache/aqua_oc_reproduction_v1/raw/grqa/DOC_GRQA.csv",
            "cache/aqua_oc_reproduction_v1/raw/grqa/README.txt",
            "cache/aqua_oc_reproduction_v1/local_doc_samples.parquet",
            "cache/aqua_oc_reproduction_v1/frozen_comparison_cells.parquet",
            "cache/aqua_oc_reproduction_v1/pilot_landsat_blue.npy",
            "cache/doc_domain_work_v1/sources/aqua_oc_2025.pdf",
            "cache/doc_domain_work_v1/sources/aqua_oc_supplement.docx",
            "experiments/phase4_transfer/doc_domain_work_v1/protocol.json"]
    hashes = {name: digest(ROOT / name) for name in code + artifacts + data}
    receipt = json.loads((OUTPUT / "data_quality_audit.json").read_text())
    if receipt["training_started"] or receipt["complete_aqua_oc_reproduction"]:
        raise ValueError("Preparation audit has an incorrect training/completion claim")
    if receipt["monthly_cells_missing_in_current_raw"] or receipt["monthly_cells_difference_over_1e_minus_4"]:
        raise ValueError("Daily records do not reconcile with the frozen monthly truth")
    for row in json.loads((OUTPUT / "download_receipts.json").read_text()):
        if digest(ROOT / row["file"]) != row["sha256"]:
            raise ValueError("Verified download differs from its receipt")
    manifest = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "stage": "data_acquisition_and_quality_audit_only",
        "training_started": False, "complete_aqua_oc_reproduction": False,
        "training_protocol_frozen": False,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "workspace_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "generator_script_sha256": digest(Path(__file__)), "sha256": hashes,
        "canonical_dataset_sha256": receipt["dataset_sha256"],
        "unresolved": ["Authorized ERA5-Land access", "AquaSat release acquisition",
                       "Original cohort/filter configuration and DOC model implementation details",
                       "Optical matchup coverage and geographical alias exclusions",
                       "Common monthly estimand and frozen comparison protocol"],
        "verification": "Downloaded-file SHA256, provider MD5 and canonical monthly reconciliation verified by scripts. Test output reported separately.",
    }
    write_json(OUTPUT / "preparation_manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "hashed_files": len(hashes),
                      "training_started": False, "reconciliation": "passed",
                      "manifest": str((OUTPUT / "preparation_manifest.json").relative_to(ROOT))}))


if __name__ == "__main__":
    main()
