"""Prepare pinned cache-local packages and source PDFs for the DOC benchmark."""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/doc_domain_work_v1"
PACKAGES = ("pypdf==6.1.1", "pygam==0.10.1", "xgboost==3.1.0",
            "progressbar2==4.5.0", "python-utils==3.9.1")
SOURCES = {
    "aqua_oc_2025.pdf": "https://gsprs-pku.github.io/articles/Tian_2025_ISPRS.pdf",
    "aqua_oc_supplement.docx": "https://authors.library.caltech.edu/records/b74cg-58e81/files/1-s2.0-S0924271625000334-mmc1.docx?download=1",
}


def main():
    marker = CACHE / "dependency_setup.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        for name, expected in record["critical_hashes"].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("prepared dependency changed; inspect before replacing")
        print(json.dumps({"stage": "already_prepared", "packages": PACKAGES}))
        return
    deps = CACHE / "python_deps"
    if not deps.exists():
        subprocess.run(["uv", "pip", "install", "--target", str(deps), "--no-deps", *PACKAGES], check=True)
    sources = CACHE / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (sources / name).exists():
            urllib.request.urlretrieve(url, sources / name)
    critical = {str((sources / name).relative_to(ROOT)): sha256_file(sources / name) for name in SOURCES}
    patch = None
    if sys.platform == "darwin":
        library = deps / "xgboost/lib/libxgboost.dylib"
        omp = Path(torch.__file__).parent / "lib/libomp.dylib"
        linkage = subprocess.check_output(["otool", "-L", str(library)], text=True)
        before = sha256_file(library)
        if "@rpath/libomp.dylib" in linkage:
            subprocess.run(["install_name_tool", "-change", "@rpath/libomp.dylib", str(omp), str(library)], check=True)
            subprocess.run(["codesign", "--force", "--sign", "-", str(library)], check=True)
        patch = {"kind": "macOS OpenMP loader path only; XGBoost algorithm code unchanged",
            "library_before_setup_sha256": before, "library_after_setup_sha256": sha256_file(library),
            "openmp_path": str(omp), "openmp_sha256": sha256_file(omp),
            "already_patched_on_initial_setup": "@rpath/libomp.dylib" not in linkage}
        critical[str(library.relative_to(ROOT))] = sha256_file(library)
        critical[str(omp.relative_to(ROOT))] = sha256_file(omp)
    record = {"packages": PACKAGES, "sources": SOURCES, "platform_patch": patch, "critical_hashes": critical}
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": "prepared", "packages": PACKAGES, "platform_patch": patch}))


if __name__ == "__main__":
    main()
