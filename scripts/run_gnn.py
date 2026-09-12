"""Run the GCN model (topology variants) over benchmark masks.

    python scripts/run_gnn.py --only e1_r20_seed42 --variants river
    python scripts/run_gnn.py                       # all masks x variants

Cache behaviour (R1 fix; see docs/run_gnn_prediction_storage.md)
---------------------------------------------------------------
For every (model, mask) pair the runner inspects what is actually on disk:

``complete``          metrics + predictions -> skip (no training)
``metrics_only``      the historical R1 trap: the metrics JSON has an entry but
                      the per-cell parquet does not exist. Nothing is silently
                      declared successful:
                        * ``--rebuild-predictions``  re-runs that mask once so
                          metrics and predictions come from the same run
                        * ``--save-predictions``     reports the gap and exits
                          non-zero at the end, without training
                        * neither                    metrics are reused as-is
``predictions_only``  metrics are rebuilt from the stored test rows
``absent``            train once, then store metrics (and predictions when
                      requested)

A stored prediction whose provenance sidecar records a different config hash
is a different run reusing the same name; that is refused unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics
from river_graph.experiments.predictions import (
    CacheState,
    PredictionConflictError,
    cache_state,
    prediction_path,
    read_meta,
    save_predictions,
)
from river_graph.experiments.provenance import build_meta, config_hash, describe
from river_graph.models.gcn import GCNDocModel

RESULTS = Path("experiments/results")
MASKS_DIR = Path("experiments/masks")
PRED_DIR = Path("experiments/predictions")


def model_name_for(args: argparse.Namespace, variant: str) -> str:
    """Result-name prefix for one run (``--model-name`` wins when given)."""
    if args.model_name:
        prefix = args.model_name
    elif args.tag:
        prefix = args.tag
    elif args.arch == "gcn":
        prefix = "G0_gcn"
    elif args.arch == "transport_enc":
        prefix = "H2X_transport_enc"
    elif args.arch == "transport":
        prefix = "H2_transport"
    elif args.share_weights or args.edge_dropout or args.wd:
        prefix = "H15_directed"
    else:
        prefix = "H1_directed"
    return f"{prefix}_{variant}" + (f"_s{args.seed}" if args.seed else "")


def run_params(args: argparse.Namespace, model_name: str) -> dict:
    """Everything that determines the produced numbers (feeds the config hash)."""
    return {
        "script": "scripts/run_gnn.py",
        "model_name": model_name,
        "tag": args.tag,
        "architecture": args.arch,
        "variant": args.variant,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.wd,
        "edge_dropout": args.edge_dropout,
        "share_weights": args.share_weights,
        "env_groups": args.env_groups,
        "env_encoder": args.arch == "transport_enc",
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _metrics_from_predictions(parquet: Path) -> dict:
    """Rebuild the metric record from stored test rows (no training)."""
    df = pd.read_parquet(parquet)
    test = df[df["split"] == "test"]
    return metrics(test["y_true"].values, test["y_pred"].values)


@dataclass
class RunReport:
    """Outcome of one runner invocation, suitable for assertions in tests."""

    trained: int = 0
    skipped: int = 0
    rebuilt_from_predictions: int = 0
    missing_predictions: list[str] = field(default_factory=list)
    identity_mismatch: list[str] = field(default_factory=list)
    metric_drift: list[str] = field(default_factory=list)
    legacy_predictions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"trained={self.trained} skipped={self.skipped} "
                f"metrics_rebuilt={self.rebuilt_from_predictions} "
                f"missing_predictions={len(self.missing_predictions)} "
                f"identity_mismatch={len(self.identity_mismatch)} "
                f"metric_drift={len(self.metric_drift)} "
                f"legacy_predictions={len(self.legacy_predictions)}")


def run(
    args: argparse.Namespace,
    model_factory: Callable[..., object] | None = None,
    results_dir: Path = RESULTS,
    masks_dir: Path = MASKS_DIR,
    pred_dir: Path | None = None,
) -> RunReport:
    """Execute the requested runs. ``model_factory`` is injectable for tests."""
    factory = model_factory or GCNDocModel
    report = RunReport()
    store = Path(pred_dir) if pred_dir is not None else PRED_DIR
    pred_kwargs = {"out_dir": store}

    dataset = load_dataset(args.dataset)
    names = sorted(p.stem for p in masks_dir.glob("*.npz"))
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    print(f"masks: {names}; variants: {args.variants}")
    if not names:
        print("no masks matched; nothing to do")
        return report

    results_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        args.variant = variant
        mname = model_name_for(args, variant)
        params = run_params(args, mname)
        jpath = results_dir / f"{mname}.json"
        merged = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
        model = None  # built lazily so a fully cached run trains nothing

        for mask_name in names:
            tag = f"{mname} @ {mask_name}"
            state = cache_state(mname, mask_name, results_dir, store,
                                metrics=merged)
            parquet = prediction_path(mname, mask_name, store)

            # identity check: a different config must not pass as this result
            meta = read_meta(mname, mask_name, **pred_kwargs) if parquet.exists() else None
            if parquet.exists() and meta is not None:
                if meta.get("config_hash") != config_hash(params) and not args.force:
                    report.identity_mismatch.append(f"{mname}__{mask_name}")
                    print(f"{tag}: IDENTITY MISMATCH -- stored prediction has a "
                          f"different configuration; refusing to reuse it. "
                          f"{describe(meta)}. Use --force to replace it.")
                    continue
            elif parquet.exists():
                report.legacy_predictions.append(f"{mname}__{mask_name}")

            if state is CacheState.COMPLETE and not args.force:
                report.skipped += 1
                print(f"{tag}: cached, skip")
                continue

            if state is CacheState.METRICS_ONLY and not args.rebuild_predictions and not args.force:
                if args.save_predictions:
                    report.missing_predictions.append(f"{mname}__{mask_name}")
                    print(f"{tag}: PREDICTIONS MISSING -- metrics are cached but "
                          f"the per-cell parquet does not exist. Re-run with "
                          f"--rebuild-predictions to regenerate it, or pass "
                          f"--save-predictions on a fresh run.")
                else:
                    report.skipped += 1
                    print(f"{tag}: cached metrics, skip")
                continue

            if state is CacheState.PREDICTIONS_ONLY and not args.force:
                m = _metrics_from_predictions(parquet)
                merged[mask_name] = m
                _atomic_write_json(jpath, merged)
                report.rebuilt_from_predictions += 1
                print(f"{tag}: metrics rebuilt from stored predictions "
                      f"MAE={m['mae']:.3f} R2={m['r2']:.3f} (n={m['n']})")
                continue

            split = load_mask(mask_name, masks_dir)
            if model is None:
                model = factory(
                    variant=variant, lr=args.lr, architecture=args.arch,
                    seed=args.seed, env_groups=args.env_groups,
                    env_encoder=args.arch == "transport_enc",
                    share_weights=args.share_weights,
                    edge_dropout=args.edge_dropout, weight_decay=args.wd,
                )
            pred = model.fit_predict(dataset, split)
            y = dataset["y"].numpy()
            m = metrics(y.ravel()[split["test"]], pred.ravel()[split["test"]])
            report.trained += 1

            if args.save_predictions:
                meta_payload = build_meta(
                    model_name=mname, mask_name=mask_name,
                    dataset_path=args.dataset, split=split, params=params,
                    results_path=jpath, masks_dir=masks_dir,
                )
                try:
                    out, _mpath = save_predictions(
                        pred, dataset, split, mname, mask_name,
                        Path(args.dataset).stem, meta=meta_payload,
                        force=args.force, **pred_kwargs,
                    )
                except PredictionConflictError as exc:
                    report.identity_mismatch.append(f"{mname}__{mask_name}")
                    print(f"{tag}: IDENTITY MISMATCH -- {exc}")
                    continue
                print(f"{tag}: predictions -> {out.name} ({describe(meta_payload)})")

            merged[mask_name] = m
            _atomic_write_json(jpath, merged)
            print(f"{tag}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} "
                  f"R2={m['r2']:.3f} (n={m['n']})", flush=True)

    rebuild_flat_csv(results_dir)
    return report


def rebuild_flat_csv(results_dir: Path = RESULTS) -> None:
    """Rebuild the flat CSV from all per-model JSONs (never overwrite blindly:
    it is derived from the JSONs, which remain the source of truth)."""
    rows = []
    for jpath in sorted(results_dir.glob("G*_*.json")) + sorted(results_dir.glob("H*_*.json")):
        for mask_name, m in json.loads(jpath.read_text(encoding="utf-8")).items():
            rows.append({"model": jpath.stem, "mask": mask_name, **m})
    out = results_dir / "gnn.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


def verify_predictions(
    results_dir: Path = RESULTS,
    pred_dir: Path = PRED_DIR,
    only: str | None = None,
) -> RunReport:
    """Report every cached metric that has no stored prediction (no training).

    The audit counterpart of the save path: it answers "which results would be
    silently incomplete if we claimed the cache was whole?".
    """
    report = RunReport()
    for jpath in sorted(results_dir.glob("*.json")):
        model = jpath.stem
        if only and not model.startswith(only):
            continue
        try:
            merged = json.loads(jpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report.identity_mismatch.append(f"{model} (unreadable JSON)")
            continue
        for mask_name in merged:
            state = cache_state(model, mask_name, results_dir, pred_dir,
                                metrics=merged)
            if state is CacheState.METRICS_ONLY:
                report.missing_predictions.append(f"{model}__{mask_name}")
            elif state is CacheState.COMPLETE:
                report.skipped += 1
            elif state is CacheState.PREDICTIONS_ONLY:
                report.rebuilt_from_predictions += 1
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="mask-name prefix filter")
    ap.add_argument("--variants", nargs="+", default=["river", "random", "none"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", default="gcn",
                    choices=["gcn", "directed", "transport", "transport_enc"],
                    help="encoder: plain GCN (G0), directed (H1), "
                         "transport-gated (H2), transport + ecological "
                         "context encoder (M6)")
    ap.add_argument("--dataset", default="data/processed/mississippi_graph_v02.pt",
                    help="dataset .pt (v03 adds edge_attr/regime, v04 adds "
                         "StreamCat ecological context)")
    ap.add_argument("--tag", default=None,
                    help="result-name prefix override, e.g. H2E for v04 runs")
    ap.add_argument("--model-name", default=None,
                    help="full result-name prefix override (alias of --tag; "
                         "use it to give a new configuration its own name "
                         "instead of overwriting a historical result)")
    ap.add_argument("--share-weights", action="store_true",
                    help="H1.5: share relation weights + direction embedding")
    ap.add_argument("--edge-dropout", type=float, default=0.0,
                    help="H1.5: per-edge dropout prob during training")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--seed", type=int, default=0, help="model training seed")
    ap.add_argument("--save-predictions", action="store_true",
                    help="also store per-cell predictions to "
                         "experiments/predictions/ (parquet, resumable)")
    ap.add_argument("--rebuild-predictions", action="store_true",
                    help="explicitly re-run masks whose metrics are cached but "
                         "whose predictions are missing")
    ap.add_argument("--force", action="store_true",
                    help="retrain and overwrite cached predictions even when a "
                         "stored provenance sidecar records another config")
    ap.add_argument("--env-groups", nargs="*", default=None,
                    choices=["hydro", "landcover", "climate", "soil", "topo"],
                    help="subset of v04 regime groups (default: all)")
    ap.add_argument("--verify-predictions", action="store_true",
                    help="audit mode: report every cached metric that has no "
                         "stored prediction, then exit (no training at all)")
    args = ap.parse_args()

    if args.verify_predictions:
        report = verify_predictions(only=args.only)
        print(f"prediction coverage audit: {report.summary()}")
        print(f"  metrics with predictions : {report.skipped}")
        print(f"  predictions without metrics: {report.rebuilt_from_predictions}")
        print(f"  metrics WITHOUT predictions: {len(report.missing_predictions)}")
        for name in report.missing_predictions:
            print(f"    - {name}")
        if report.missing_predictions:
            print()
            print("These results cannot be used for per-cell diagnostics. "
                  "Regenerate with --rebuild-predictions (that trains).")
        raise SystemExit(1 if report.missing_predictions else 0)

    report = run(args)
    print()
    print(f"run summary: {report.summary()}")
    if report.missing_predictions:
        print("predictions missing for:")
        for name in report.missing_predictions:
            print(f"  - {name}")
        print("these are NOT saved; rerun with --rebuild-predictions "
              "(no training happens without that flag)")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
