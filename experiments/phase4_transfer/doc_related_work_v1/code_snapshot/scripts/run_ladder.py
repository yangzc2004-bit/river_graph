"""Frozen DOC related-work training entry point.

The runner freezes data, masks, official backbones and source-only selection.
Prediction provenance is checked on resumption; historical entry points remain
unchanged in the original repository.
"""
from run_doc_related_work_v1 import main

if __name__ == "__main__":
    main()
