"""Launch separate-river training with the established local runtime helper."""
from pathlib import Path

import start_doc_unified_transfer_v2 as launcher

REPOSITORY = Path(__file__).resolve().parents[1]
launcher.OUTPUT = REPOSITORY/"experiments/phase4_transfer/doc_per_river_v1"
launcher.WORKSPACE = REPOSITORY/"cache/doc_per_river_v1/runtime"
launcher.DISPATCHER = ('"""Unchanged DOC model, trained separately per river."""\n'
    'from run_doc_per_river_v1 import main\n\nif __name__ == "__main__":\n    main()\n')

if __name__ == "__main__":
    launcher.main()
