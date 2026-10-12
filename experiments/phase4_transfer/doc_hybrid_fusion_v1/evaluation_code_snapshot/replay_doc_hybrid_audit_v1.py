"""Load the pinned baseline dependency path, then run the frozen verifier."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import run_doc_dynamic_reconstruction_v1 as base
import run_doc_hybrid_fusion_v1 as study
from run_doc_fusion_benchmark_development_v1 import utc_now
from run_doc_river_architecture_comparison_v1 import digest, write_json

from river_graph.experiments.provenance import sha256_file


def main():
    root = study.ROOT.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    dependency_path = base.optional_dependencies()
    import xgboost
    repair = {'script_sha256': sha256_file(__file__), 'recorded_at': utc_now(),
              'dependency_path': str(dependency_path), 'xgboost_version': xgboost.__version__,
              'frozen_verifier_sha256': sha256_file('scripts/verify_doc_hybrid_fusion_v1.py'),
              'reason': 'Frozen audit entry omitted optional_dependencies before deserializing the cached XGBoost controls. Load the identical pinned parent dependency path, then execute the unchanged frozen verifier.',
              'training_or_predictions_modified': False, 'selection_modified': False}
    assert repair['frozen_verifier_sha256'] == protocol['runtime_snapshot']['scripts/verify_doc_hybrid_fusion_v1.py']
    write_json(root / 'verification_environment_repair.json', repair)
    sys.argv = ['scripts/verify_doc_hybrid_fusion_v1.py', '--root', str(root)]
    runpy.run_path('scripts/verify_doc_hybrid_fusion_v1.py', run_name='__main__')
    assert json.loads((root / 'analysis/independent_audit.json').read_text())['passed']
    write_json(root / 'batch_complete.json', {'protocol_hash': digest(protocol), 'tasks': 96,
               'new_fits': 24, 'completed_at': utc_now(), 'audit_passed': True,
               'verification_environment_repair_sha256': sha256_file(root / 'verification_environment_repair.json')})
    print(json.dumps({'stage': 'complete', 'tasks': 96, 'new_fits': 24}))


if __name__ == '__main__':
    main()
