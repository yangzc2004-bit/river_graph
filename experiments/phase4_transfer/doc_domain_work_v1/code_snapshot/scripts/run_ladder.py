"""Versioned training entry point for the matched-input DOC development benchmark.

The benchmark freezes its protocol before fitting and refuses changed prediction
provenance on resumption. The original repository run_ladder is unchanged.
"""
from run_doc_domain_work_v1 import main

if __name__ == "__main__":
    main()
