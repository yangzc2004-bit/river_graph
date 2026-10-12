"""Prepare the immutable local runtime and launch via its run_ladder."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / 'experiments/phase4_transfer/doc_hybrid_fusion_v1'
WORKSPACE = REPOSITORY / 'cache/doc_hybrid_fusion_v1/runtime'
DISPATCHER = '"""Frozen hybrid RiverDOCNet entry."""\nfrom run_doc_hybrid_fusion_v1 import main\n\nif __name__ == "__main__":\n    main()\n'


def prepare():
    marker = WORKSPACE / 'workspace.json'
    if marker.exists():
        assert (WORKSPACE / 'scripts/run_ladder.py').read_text() == DISPATCHER
        return
    assert not WORKSPACE.exists(), 'Unfinished runtime: inspect before continuing'
    WORKSPACE.mkdir(parents=True)
    for directory in ('src', 'scripts', 'tests', 'docs'):
        shutil.copytree(REPOSITORY / directory, WORKSPACE / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(REPOSITORY / name, WORKSPACE / name)
    for directory in ('data', 'experiments', '.venv', 'cache'):
        (WORKSPACE / directory).symlink_to(REPOSITORY / directory, target_is_directory=True)
    (WORKSPACE / 'scripts/run_ladder.py').write_text(DISPATCHER)
    marker.write_text(json.dumps({'repository': str(REPOSITORY)}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-time', type=int, default=3)
    args = parser.parse_args()
    prepare()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(WORKSPACE / 'src')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        environment[name] = '1'
    command = ['uv', 'run', '--no-sync', 'python', 'scripts/run_ladder.py', '--repository', str(REPOSITORY),
               '--root', str(OUTPUT), '--workers', str(args.workers), '--max-time', str(args.max_time)]
    subprocess.run([*command, '--freeze-only'], cwd=WORKSPACE, env=environment, check=True)
    if args.freeze_only:
        return
    pid_path = OUTPUT / 'training.pid'
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            print(json.dumps({'stage': 'already_running'}))
            return
    with (OUTPUT / 'training.log').open('a') as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(process.pid) + '\n')
    inhibitor = None
    if shutil.which('caffeinate'):
        inhibitor = subprocess.Popen(['caffeinate', '-i', '-w', str(process.pid)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (OUTPUT / 'launch.json').write_text(json.dumps({'pid': process.pid, 'command': command,
             'workspace': str(WORKSPACE), 'sleep_inhibitor_pid': inhibitor.pid if inhibitor else None}, indent=2) + '\n')
    print(json.dumps({'stage': 'started', 'pid': process.pid, 'new_fits': 24}))


if __name__ == '__main__':
    main()
