"""Restore only the detached process-scoped macOS sleep inhibitor."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from river_graph.experiments.provenance import sha256_file

ROOT = Path('experiments/phase4_transfer/doc_hybrid_fusion_v1')


def main():
    launch = json.loads((ROOT / 'launch.json').read_text())
    os.kill(launch['pid'], 0)
    original = launch.get('sleep_inhibitor_pid')
    if original:
        try:
            os.kill(original, 0)
        except ProcessLookupError:
            pass
        else:
            print(json.dumps({'inhibitor_alive': original}))
            return
    proc = subprocess.Popen(['caffeinate', '-i', '-w', str(launch['pid'])],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    record = {'training_pid': launch['pid'], 'sleep_inhibitor_pid': proc.pid,
              'script_sha256': sha256_file(__file__), 'scientific_settings_changed': False,
              'reason': 'Launcher-group sleep inhibitor exited; restore only the detached process-scoped inhibitor. No system preference or model/mask/runtime is changed.'}
    (ROOT / 'sleep_support.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
