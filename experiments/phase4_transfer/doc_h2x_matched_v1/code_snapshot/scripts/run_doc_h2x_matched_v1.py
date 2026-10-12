"""Historical H2X with the current frozen label-free DOC information budget."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_doc_dynamic_reconstruction_v1 as base
import run_doc_hybrid_fusion_v1 as parent_study
import torch
from run_doc_fusion_benchmark_development_v1 import (
    bind,
    save_predictions,
    utc_now,
    verify,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    role_labels,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.doc_fusion_benchmark import metrics
from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_dynamic_reconstruction import (
    change_metrics,
    label_scales,
    reconstruction_loss,
)
from river_graph.models.h2x_matched_doc import MatchedInputH2X

ROOT = Path('experiments/phase4_transfer/doc_h2x_matched_v1')
PARENT = parent_study.ROOT
NEW_ARM = 'h2x_matched'
ARMS = (*parent_study.ARMS, NEW_ARM)
NEW_CONFIG = {'env_dim': 15, 'lookback': 12, 'hidden': 64, 'env_emb': 32,
              'layers': 2, 'dropout': .1}
CODE = (*parent_study.CODE, 'src/river_graph/models/hydro.py',
        'src/river_graph/models/h2x_matched_doc.py',
        'scripts/run_doc_h2x_matched_v1.py', 'scripts/start_doc_h2x_matched_v1.py',
        'scripts/verify_doc_h2x_matched_v1.py', 'tests/test_doc_h2x_matched_v1.py',
        'docs/doc_h2x_matched_plan_20261012.md')


def tasks():
    result = []
    for old in parent_study.tasks():
        result.append(old)
        if old['arm'] == 'monthly_log':
            result.append({**old, 'arm': NEW_ARM,
                           'id': old['id'].replace('monthly_log', NEW_ARM)})
    return result


def directory(root, task):
    return root / 'fits' / task['id'] if task['arm'] == NEW_ARM else parent_study.directory(PARENT, task)


def freeze(root, repository, workers, max_time):
    root.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT / 'protocol.json').read_text())
    assert json.loads((PARENT / 'analysis/independent_audit.json').read_text())['passed']
    assert sha256_file(base.DATASET) == parent['dataset_sha256']
    assert sha256_file(base.NODES) == parent['nodes_sha256']
    for name in parent_study.CODE:
        if name != 'scripts/run_ladder.py':
            assert sha256_file(name) == parent['runtime_snapshot'][name], name
    masks, dependencies = {}, {}
    for task in tasks():
        name = task['mask']
        source = PARENT / 'masks' / f'{name}.npz'
        assert sha256_file(source) == parent['mask_sha256'][name]
        target = root / 'masks' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
        masks[name] = sha256_file(target)
        if task['arm'] != NEW_ARM:
            d = directory(root, task)
            verify(d, 'selection_complete.json')
            verify(d, 'test_complete.json')
            names = set(json.loads((d / 'selection_complete.json').read_text())['files'])
            names.update(json.loads((d / 'test_complete.json').read_text())['files'])
            names.update(('selection_complete.json', 'test_complete.json'))
            dependencies[task['id']] = {n: sha256_file(d / n) for n in sorted(names)}
    with torch.random.fork_rng(devices=[]):
        parameters = sum(p.numel() for p in MatchedInputH2X(**NEW_CONFIG).parameters())
    protocol = {'experiment': ROOT.name, 'parent': str(PARENT), 'tasks': tasks(),
                'arms': list(ARMS), 'new_arm': NEW_ARM, 'model_config': NEW_CONFIG,
                'dataset': str(base.DATASET), 'dataset_sha256': parent['dataset_sha256'],
                'nodes': str(base.NODES), 'nodes_sha256': parent['nodes_sha256'],
                'settings': parent['settings'], 'mask_sha256': masks,
                'workers': workers, 'max_simultaneous_time': max_time,
                'runtime_snapshot': {n: sha256_file(n) for n in CODE},
                'baseline_dependencies': dependencies,
                'parent_protocol_sha256': sha256_file(PARENT / 'protocol.json'),
                'parent_audit_sha256': sha256_file(PARENT / 'analysis/independent_audit.json'),
                'repository': str(repository), 'joint_primary': parent['joint_primary'],
                'uses_doc_inputs': False, 'uses_daily_features': False,
                'uses_original_nhd_edge_attributes': False,
                'edge_gate_inputs': 'concatenated existing normalized environmental features of upstream and downstream endpoints; no extra measurements',
                'node_inputs': 'same 113 local environmental/history/season values as trees; twelve-month driver history flattened, not recurrent',
                'checkpoint_selection': parent['checkpoint_selection'],
                'candidate_selection': 'one predeclared architecture; all validation selections sealed before new test scoring',
                'parameters': {**parent['parameters'], NEW_ARM: parameters},
                'adaptation': 'unmodified released TransportGCNImputer: env embedding32, hidden64, two directed gated layers; replace old DOC channel with same flattened driver history and original edge attributes with existing endpoint environment',
                'scope': parent['scope'], 'aggregation': parent['aggregation'],
                'versions': parent['versions'],
                'original_dispatcher_sha256': sha256_file(repository / 'scripts/run_ladder.py')}
    if (root / 'protocol.json').exists():
        assert json.loads((root / 'protocol.json').read_text()) == protocol, 'Frozen protocol changed'
    else:
        write_json(root / 'protocol.json', protocol)
        for name in CODE:
            dst = root / 'code_snapshot' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(name).read_bytes())
        write_json(root / 'manifest.json', {'protocol_hash': digest(protocol), 'frozen_at': utc_now(),
                   'code_commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
                   'workspace_dirty': bool(subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True)),
                   'generator_script_sha256': sha256_file(__file__), 'runtime_snapshot_hash': digest(protocol['runtime_snapshot']),
                   'dataset_sha256': protocol['dataset_sha256'], 'nodes_sha256': protocol['nodes_sha256'],
                   'mask_sha256': masks, 'parent_protocol_sha256': protocol['parent_protocol_sha256']})
    return protocol


def identity(protocol, task):
    return {'experiment': ROOT.name, **task, 'protocol_hash': digest(protocol),
            'dataset_sha256': protocol['dataset_sha256'], 'mask_sha256': protocol['mask_sha256'][task['mask']],
            'runtime_snapshot_hash': digest(protocol['runtime_snapshot'])}


def context(root, task, data):
    # Monthly controls and hybrid receive exactly the same tensor. The eight
    # unused daily channels remain zero, as in the audited original control.
    zeros = np.zeros((*data['y'].shape, 8), np.float32)
    return base.context(root, {**task, 'arm': 'monthly_log'}, data, zeros)


def make_model(protocol):
    return MatchedInputH2X(**protocol['model_config'])


def train(ctx, task, data, protocol, progress):
    """Same update schedule, normalization and loss as the audited control."""
    settings = protocol['settings']
    torch.manual_seed(task['seed'])
    model = make_model(protocol)
    train_graph, val_graph = ctx['graphs']['train'], ctx['graphs']['val']
    train_y = role_labels(data, ctx['split'], 'train', train_graph['rows'].numpy())
    val_y = role_labels(data, ctx['split'], 'val', val_graph['rows'].numpy())
    scales = label_scales(train_y)
    active = torch.where(torch.isfinite(train_y).any(1))[0].numpy()
    batches = int(np.ceil(len(active) / settings['batch_months']))
    denominators = {'cells': scales['training_cells'] / batches,
                    'pairs': max(scales['training_adjacent_pairs'] / batches, 1.)}
    previous_y = torch.cat([torch.full_like(train_y[:1], float('nan')), train_y[:-1]])
    val_months = torch.where(torch.isfinite(val_y).any(1))[0].numpy()
    val_truth = val_y[val_months].numpy()
    valid = np.isfinite(val_truth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    rng = np.random.default_rng(task['seed'])
    rows = train_graph['rows']
    best, stale, state, trace, started = float('inf'), 0, None, [], time.monotonic()
    for epoch in range(1, settings['max_epochs'] + 1):
        model.train()
        permutation, total = rng.permutation(active), 0.
        for start in range(0, len(permutation), settings['batch_months']):
            current = permutation[start:start + settings['batch_months']]
            previous = np.maximum(current - 1, 0)
            union = np.union1d(current, previous)
            optimizer.zero_grad(set_to_none=True)
            all_z = model(ctx['inputs']['windows'][union][:, rows], ctx['inputs']['environment'][rows],
                          ctx['inputs']['season'][union], train_graph)
            z = all_z[np.searchsorted(union, current)]
            previous_z = all_z[np.searchsorted(union, previous)]
            value = reconstruction_loss(z, train_y[current], previous_z, previous_y[current], scales, denominators, 'log')
            if not torch.isfinite(value):
                raise ValueError('Nonfinite training objective')
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise ValueError('Nonfinite gradient')
            optimizer.step()
            total += float(value.detach())
        pred = native_prediction(predict(model, ctx['inputs'], val_graph, val_months, settings['batch_months']),
                                 {'mean': scales['log_mean'], 'std': scales['log_std']})
        score = float(np.abs(pred[valid] - val_truth[valid]).mean())
        if score < best - 1e-6:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        row = {'epoch': epoch, 'objective_sum': total, 'validation_mae': score,
               'best_epoch': best_epoch, 'elapsed_seconds': time.monotonic() - started}
        trace.append(row)
        if epoch == 1 or epoch % 5 == 0:
            progress(row)
        if epoch >= settings['min_epochs'] and stale >= settings['patience']:
            break
    model.load_state_dict(state)
    return model, {'best_epoch': best_epoch, 'epochs_run': epoch, 'validation_mae': best,
                   'training_seconds': time.monotonic() - started,
                   'parameters': sum(p.numel() for p in model.parameters()), 'scales': scales}, trace


def fit(root, task, protocol, data):
    d, config = directory(root, task), identity(protocol, task)
    if (d / 'selection_complete.json').exists():
        verify(d, 'selection_complete.json', config)
        return
    d.mkdir(parents=True, exist_ok=True)
    ctx = context(root, task, data)
    progress = lambda row: write_json(root / f"progress_{task['id']}.json", {'stage': 'training', **task, **row})
    model, info, trace = train(ctx, task, data, protocol, progress)
    prediction = base.neural_cells(model, ctx, ctx['split']['val'], info['scales'], protocol['settings']['batch_months'])
    part = base.make_frame(data, task, ctx['split']['val'], prediction)
    torch.save(model.state_dict(), d / 'model.pt')
    write_json(d / 'config.json', config)
    write_json(d / 'normalization.json', ctx['inputs']['normalization'])
    write_json(d / 'info.json', info)
    write_json(d / 'trace.json', trace)
    write_json(d / 'selection.json', {'test_used': False, 'validation_metrics': {**metrics(part), **change_metrics(part)}})
    save_predictions(d, 'validation_predictions', part, config)
    bind(d, 'selection_complete.json', config, ['config.json', 'normalization.json', 'info.json', 'trace.json',
         'model.pt', 'selection.json', 'validation_predictions.parquet', 'validation_predictions.meta.json'])
    write_json(root / f"progress_{task['id']}.json", {'stage': 'selection_complete', **task, **info})
    print(json.dumps({'stage': 'selection_complete', 'task': task['id'], **info}), flush=True)


def score(root, task, protocol, data):
    assert (root / 'all_selections.json').exists(), 'Every validation choice must be sealed first'
    d, config = directory(root, task), identity(protocol, task)
    if (d / 'test_complete.json').exists():
        verify(d, 'test_complete.json', config)
        return
    verify(d, 'selection_complete.json', config)
    ctx = context(root, task, data)
    info = json.loads((d / 'info.json').read_text())
    model = make_model(protocol)
    model.load_state_dict(torch.load(d / 'model.pt', weights_only=True, map_location='cpu'))
    prediction = base.neural_cells(model, ctx, ctx['split']['test'], info['scales'], protocol['settings']['batch_months'])
    save_predictions(d, 'predictions', base.make_frame(data, task, ctx['split']['test'], prediction), config)
    bind(d, 'test_complete.json', config, ['predictions.parquet', 'predictions.meta.json'])


def analyze(root, protocol):
    all_predictions = pd.concat([pd.read_parquet(directory(root, t) / 'predictions.parquet') for t in protocol['tasks']], ignore_index=True)
    records = [{'track': track, 'gap_months': int(gap), 'seed': int(seed), 'model_name': arm,
                **metrics(part), **change_metrics(part)} for (track, gap, seed, arm), part in
               all_predictions.groupby(['track', 'gap_months', 'seed', 'model_name'])]
    by_seed = pd.DataFrame(records)
    summary = by_seed.groupby(['track', 'gap_months', 'model_name'], as_index=False).mean(numeric_only=True)
    regions = [{'region': region, 'seed': int(seed), 'model_name': arm, **metrics(part), **change_metrics(part)}
               for (region, seed, arm), part in all_predictions[all_predictions.track == 'space'].groupby(['target_huc4', 'seed', 'model_name'])]
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    by_seed.to_csv(out / 'by_seed.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(regions).to_csv(out / 'by_region_seed.csv', index=False)
    overview = by_seed[by_seed.track == 'time'].groupby('model_name', as_index=False).mean(numeric_only=True)
    overview.to_csv(out / 'temporal_overview.csv', index=False)
    bind(out, 'complete.json', {'protocol_hash': digest(protocol)},
         ['by_seed.csv', 'summary.csv', 'by_region_seed.csv', 'temporal_overview.csv'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repository', type=Path, default=Path.cwd())
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-time', type=int, default=3)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--task')
    args = parser.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text()) if args.task else freeze(args.root, args.repository, args.workers, args.max_time)
    if args.freeze_only:
        print(json.dumps({'stage': 'frozen', 'comparison_tasks': len(tasks()), 'new_fits': 24}))
        return
    torch.set_num_threads(1)
    data = torch.load(base.DATASET, weights_only=False, map_location='cpu')
    if args.task:
        task = next(t for t in protocol['tasks'] if t['id'] == args.task)
        assert task['arm'] == NEW_ARM
        with threadpool_limits(limits=1):
            fit(args.root, task, protocol, data)
        return
    pending = [t for t in protocol['tasks'] if t['arm'] == NEW_ARM]
    pending.sort(key=lambda t: (t['track'] != 'time', t['gap_months'], t['id']))
    running = []
    logs = args.root / 'worker_logs'
    logs.mkdir(exist_ok=True)
    while pending or running:
        kept = []
        for task, process, log in running:
            if process.poll() is None:
                kept.append((task, process, log))
            else:
                log.close()
                if process.returncode:
                    raise RuntimeError(f"Fit failed: {task['id']}")
                verify(directory(args.root, task), 'selection_complete.json', identity(protocol, task))
        running = kept
        while pending and len(running) < protocol['workers']:
            active_time = sum(t['track'] == 'time' for t, _, _ in running)
            i = next((i for i, t in enumerate(pending) if t['track'] != 'time' or active_time < protocol['max_simultaneous_time']), None)
            if i is None:
                break
            task = pending.pop(i)
            log = (logs / f"{task['id']}.log").open('a')
            process = subprocess.Popen([os.sys.executable, 'scripts/run_ladder.py', '--root', str(args.root), '--task', task['id']], stdout=log, stderr=subprocess.STDOUT)
            running.append((task, process, log))
        time.sleep(.5)
    write_json(args.root / 'all_selections.json', {'protocol_hash': digest(protocol), 'test_used_for_selection': False,
               'architecture_predeclared': NEW_CONFIG, 'frozen_at': utc_now(),
               'new_fit_receipts': {t['id']: sha256_file(directory(args.root, t) / 'selection_complete.json')
                                    for t in protocol['tasks'] if t['arm'] == NEW_ARM},
               'baseline_dependencies_hash': digest(protocol['baseline_dependencies'])})
    with threadpool_limits(limits=1):
        for task in protocol['tasks']:
            if task['arm'] == NEW_ARM:
                score(args.root, task, protocol, data)
    analyze(args.root, protocol)
    subprocess.run([os.sys.executable, 'scripts/verify_doc_h2x_matched_v1.py', '--root', str(args.root)], check=True)
    write_json(args.root / 'batch_complete.json', {'protocol_hash': digest(protocol), 'tasks': len(protocol['tasks']), 'new_fits': 24, 'completed_at': utc_now(), 'audit_passed': True})
    print(json.dumps({'stage': 'complete', 'tasks': len(protocol['tasks']), 'new_fits': 24}), flush=True)


if __name__ == '__main__':
    main()
