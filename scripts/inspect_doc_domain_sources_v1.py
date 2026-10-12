"""Extract primary-source methods and inspect isolated optional dependencies."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from river_graph.experiments.doc_domain_work import optional_dependencies

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cache/doc_domain_work_v1/python_deps"))


def main():
    optional_dependencies()
    for name in ("pypdf", "pygam", "xgboost", "scipy"):
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"))
    from pypdf import PdfReader

    source = ROOT / "cache/doc_domain_work_v1/sources/aqua_oc_2025.pdf"
    destination = source.with_suffix(".txt")
    destination.write_text("\n".join(page.extract_text() for page in PdfReader(source).pages))
    print("Extracted paper:", destination)
    supplement = source.with_name("aqua_oc_supplement.docx")
    if supplement.exists():
        with ZipFile(supplement) as archive:
            xml = ElementTree.fromstring(archive.read("word/document.xml"))
        supplement.with_suffix(".txt").write_text("\n".join(
            element.text or "" for element in xml.iter()
            if element.tag.endswith("}t")))


if __name__ == "__main__":
    main()
