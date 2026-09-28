"""Compare completed mechanism arms on paired station-month test cells.

Example: --arm m2_learned=.../m2 --arm m2_static=.../m2
Each supplied directory must contain runs/<run>/{meta.json,full_grid.parquet}.
Unfinished configurations are reported and excluded; complete configurations
must have unique analyte/mask/seed identities within each arm.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import config_hash, sha256_file


def load_arm(root: Path) -> tuple[dict, list[dict]]:
    runs, inventory = {}, []
    for path in sorted((root / "runs").glob("*")):
        meta_file, product = path / "meta.json", path / "full_grid.parquet"
        if not meta_file.is_file() or not product.is_file():
            inventory.append({"run_dir": str(path), "status": "incomplete"})
            continue
        meta = json.loads(meta_file.read_text())
        cfg = meta["config"]
        if sha256_file(product) != meta["full_grid_sha256"]:
            raise ValueError(f"prediction content differs from sidecar: {product}")
        recorded_hash = meta.get("config_hash", cfg.get("config_hash"))
        version = meta.get("config_hash_version", cfg.get("config_hash_version", 3))
        identity = "verified"
        if recorded_hash != config_hash(cfg, version=version):
            # The original graph-upgrade runner declared V4 but wrote V3.
            # Identify that historical defect explicitly; never relabel it
            # as a fully verified V4 artifact.
            if version == 4 and recorded_hash == config_hash(cfg, version=3):
                identity = "historical_v4_declared_v3_written"
            else:
                raise ValueError(f"configuration cannot be reproduced: {meta_file}")
        full = pd.read_parquet(product)
        for column in ("analyte", "mask", "seed"):
            if full[column].nunique() != 1:
                raise ValueError(f"mixed {column} values: {product}")
        key = (str(full.analyte.iloc[0]), str(full['mask'].iloc[0]), int(full.seed.iloc[0]))
        if key in runs:
            raise ValueError(f"duplicate task within arm: {key}")
        test = full[full['split'] == 'test'].copy()
        test.station = test.station.astype(str)
        test.month = test.month.astype(str)
        test = test.set_index(['station', 'month']).sort_index()
        if test.index.has_duplicates or test.empty:
            raise ValueError(f"empty/duplicate query cells: {product}")
        if not np.isfinite(test[['y_true', 'y_pred']].to_numpy()).all():
            raise ValueError(f"nonfinite test values: {product}")
        threshold = float(full.loc[full['split'] == 'train', 'y_true'].quantile(.9))
        record = {"run_dir": str(path), "status": "complete", "identity": identity,
                  "analyte": key[0], "mask": key[1], "seed": key[2],
                  "max_epochs": cfg.get('max_epochs'), "patience": cfg.get('patience'),
                  "n": len(test), "q90_threshold": threshold,
                  "mae": float((test.y_true - test.y_pred).abs().mean())}
        inventory.append(record)
        runs[key] = (test, cfg, record)
    return runs, inventory


def paired_errors(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if not left.index.equals(right.index):
        raise ValueError('query station-month cells differ')
    if not np.array_equal(left.y_true.to_numpy(), right.y_true.to_numpy()):
        raise ValueError('query truth differs')
    return pd.DataFrame({
        'a': (left.y_true - left.y_pred).abs(),
        'b': (right.y_true - right.y_pred).abs(),
    }, index=left.index)


def station_interval(errors: pd.DataFrame, repeats: int = 2000) -> tuple[float, float]:
    """Station cluster interval; each station retains its query months.

    errors has already been averaged across matched model seeds, so seeds
    never increase the effective ecological sample size.
    """
    grouped = errors.groupby(level='station').agg(['sum', 'count'])
    if len(grouped) < 2:
        return float('nan'), float('nan')
    differences = grouped['b']['sum'].to_numpy() - grouped['a']['sum'].to_numpy()
    counts = grouped['a']['count'].to_numpy()
    rng = np.random.default_rng(42)
    draws = rng.integers(len(grouped), size=(repeats, len(grouped)))
    delta = differences[draws].sum(1) / counts[draws].sum(1)
    return tuple(float(v) for v in np.quantile(delta, [.025, .975]))


def compare(arms: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows, summaries = [], []
    for a_name, b_name in itertools.combinations(arms, 2):
        left, right = arms[a_name], arms[b_name]
        shared = sorted(set(left) & set(right))
        groups = {}
        for key in shared:
            a, ac, ar = left[key]
            b, bc, br = right[key]
            for field in ('dataset_sha256', 'mask_sha256'):
                if not ac.get(field) or ac[field] != bc.get(field):
                    raise ValueError(f'{key}: unequal or missing {field}')
            errors = paired_errors(a, b)
            groups.setdefault(key[:2], []).append(errors)
            mean_a, mean_b = errors.mean().to_numpy()
            seed_rows.append({
                'a': a_name, 'b': b_name, 'analyte': key[0], 'mask': key[1],
                'seed': key[2], 'n': len(errors), 'mae_a': mean_a, 'mae_b': mean_b,
                'improvement_a_vs_b_pct': 100 * (mean_b - mean_a) / mean_b,
                'same_epoch_cap': ac.get('max_epochs') == bc.get('max_epochs'),
                'same_patience': ac.get('patience') == bc.get('patience'),
                'identity_a': ar['identity'], 'identity_b': br['identity'],
            })
        for (analyte, mask), errors_by_seed in groups.items():
            if any(not e.index.equals(errors_by_seed[0].index) for e in errors_by_seed):
                raise ValueError('query cells changed with training seed')
            averaged = sum(errors_by_seed) / len(errors_by_seed)
            ma, mb = averaged.mean().to_numpy()
            lo, hi = station_interval(averaged)
            summaries.append({
                'a': a_name, 'b': b_name, 'analyte': analyte, 'mask': mask,
                'paired_seeds': len(errors_by_seed), 'unique_query_cells': len(averaged),
                'stations': averaged.index.get_level_values('station').nunique(),
                'mae_a': ma, 'mae_b': mb, 'delta_b_minus_a': mb - ma,
                'improvement_a_vs_b_pct': 100 * (mb - ma) / mb,
                'station_ci_low': lo, 'station_ci_high': hi,
                'seeds_a_better': sum(e.a.mean() < e.b.mean() for e in errors_by_seed),
            })
    return pd.DataFrame(seed_rows), pd.DataFrame(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', action='append', required=True, help='label=path')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    arms, inventory = {}, []
    for item in args.arm:
        name, path = item.split('=', 1)
        if name in arms:
            raise ValueError(f'duplicate arm label: {name}')
        arms[name], rows = load_arm(Path(path))
        inventory.extend({'arm': name, **row} for row in rows)
    seeds, summary = compare(arms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(args.out_dir / 'run_inventory.csv', index=False)
    seeds.to_csv(args.out_dir / 'paired_seed_metrics.csv', index=False)
    summary.to_csv(args.out_dir / 'paired_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
